from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from dotenv import load_dotenv
from github import Github, GithubException
from github.ContentFile import ContentFile
from github.Issue import Issue
from github.PullRequest import PullRequest
from github.Repository import Repository
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from core_engine import AgentState, CodeHealerEngine, DockerSandboxExecutor, LangChainCoder


DEMO_TEST_CODE = """from solution import safe_divide


def test_normal_division() -> None:
    assert safe_divide(10, 2) == 5


def test_division_by_zero_returns_zero() -> None:
    assert safe_divide(10, 0) == 0.0
"""


@dataclass(frozen=True)
class IssuePayload:
    number: int
    title: str
    body: str

    @property
    def task_description(self) -> str:
        return f"GitHub Issue #{self.number}: {self.title}\n\n{self.body}".strip()


@dataclass(frozen=True)
class CodeFile:
    path: str
    content: str


@dataclass(frozen=True)
class RetrievedCodeFile:
    file_path: str
    content: str


class GitHubConnector:
    """封装 CodeHealer 需要的 GitHub 仓库、Issue、文件和 PR 操作。"""

    def __init__(self, *, token: str, repo_full_name: str) -> None:
        if not token:
            raise ValueError("GITHUB_TOKEN 不能为空。")
        if "/" not in repo_full_name:
            raise ValueError("TARGET_REPO 必须形如 owner/repo，例如 octocat/test-repo。")

        self._client = Github(token)
        try:
            self._repo: Repository = self._client.get_repo(repo_full_name)
        except GithubException as exc:
            raise RuntimeError(
                f"无法访问仓库 {repo_full_name}，请检查 TARGET_REPO 和 GITHUB_TOKEN 权限。"
            ) from exc

    @property
    def default_branch(self) -> str:
        """优先使用仓库配置的默认分支，通常是 main 或 master。"""

        branch = self._repo.default_branch
        if branch:
            return branch

        for candidate in ("main", "master"):
            try:
                self._repo.get_branch(candidate)
                return candidate
            except GithubException:
                continue

        raise RuntimeError("无法识别仓库默认分支，请确认仓库存在 main 或 master 分支。")

    def get_issue(self, issue_number: int) -> IssuePayload:
        """读取指定 Issue 的标题和正文，作为修复任务描述。"""

        try:
            issue: Issue = self._repo.get_issue(number=issue_number)
        except GithubException as exc:
            raise RuntimeError(f"读取 Issue #{issue_number} 失败。") from exc

        return IssuePayload(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
        )

    def get_next_open_issue(
        self,
        *,
        label: str,
        skip_labels: Iterable[str] = ("codehealer-pr-created", "codehealer-failed"),
    ) -> IssuePayload:
        """Return the oldest open issue with the given label.

        The label is an explicit opt-in gate. CodeHealer should not blindly
        attempt every open issue in a real repository.
        """

        try:
            issues = self._repo.get_issues(
                state="open",
                labels=[label],
                sort="created",
                direction="asc",
            )
        except GithubException as exc:
            raise RuntimeError(f"读取带有标签 {label!r} 的 Issue 失败。") from exc

        skipped = set(skip_labels)
        for issue in issues:
            if issue.pull_request is not None:
                continue

            issue_labels = {item.name for item in issue.labels}
            if issue_labels & skipped:
                continue

            return self._issue_to_payload(issue)

        raise RuntimeError(f"没有找到带有标签 {label!r} 的待处理 open Issue。")

    def comment_on_issue(self, issue_number: int, body: str) -> None:
        """在 Issue 下追加一条运行结果评论；失败时不影响修复主流程。"""

        try:
            issue = self._repo.get_issue(number=issue_number)
            issue.create_comment(body)
        except GithubException as exc:
            print(f"[Issue] 添加评论失败，已跳过: Issue #{issue_number} ({exc.status})")

    def add_issue_labels(self, issue_number: int, *labels: str) -> None:
        """给 Issue 打标签；标签不存在或权限不足时只打印提醒，不中断主流程。"""

        clean_labels = [label for label in labels if label]
        if not clean_labels:
            return

        try:
            issue = self._repo.get_issue(number=issue_number)
            issue.add_to_labels(*clean_labels)
        except GithubException as exc:
            print(f"[Issue] 添加标签失败，已跳过: {clean_labels} ({exc.status})")

    def get_file_content(self, file_path: str, *, ref: Optional[str] = None) -> str:
        """读取仓库中指定文件的文本内容。"""

        try:
            content = self._repo.get_contents(file_path, ref=ref or self.default_branch)
        except GithubException as exc:
            raise RuntimeError(f"读取文件 {file_path} 失败。") from exc

        if isinstance(content, list):
            raise RuntimeError(f"{file_path} 是目录，不是可修复的单个文件。")
        return self._decode_content(content)

    def iter_python_files(self) -> Iterable[CodeFile]:
        """遍历默认分支下的 Python 源码文件，忽略测试目录和测试文件。"""

        yield from self._walk_python_files(path="", ref=self.default_branch)

    def create_pr(
        self,
        *,
        file_path: str,
        new_content: str,
        commit_message: str,
        branch_name: str,
        pr_title: str,
        pr_body: str,
    ) -> PullRequest:
        """基于默认分支创建修复分支，提交文件更新，并发起 PR。"""

        base_branch = self.default_branch
        branch_name = self._create_unique_branch(
            requested_branch_name=branch_name,
            base_branch=base_branch,
        )

        try:
            current_file = self._repo.get_contents(file_path, ref=branch_name)
        except GithubException as exc:
            raise RuntimeError(f"在新分支读取待更新文件 {file_path} 失败。") from exc

        if isinstance(current_file, list):
            raise RuntimeError(f"{file_path} 是目录，无法作为普通文件更新。")

        try:
            self._repo.update_file(
                path=file_path,
                message=commit_message,
                content=new_content,
                sha=current_file.sha,
                branch=branch_name,
            )
        except GithubException as exc:
            raise RuntimeError(f"提交文件更新失败：{file_path}。") from exc

        full_pr_body = (
            f"{pr_body.strip()}\n\n"
            "---\n"
            "此 Pull Request 由 CodeHealer 自动生成。\n"
            "合并前请人工复核补丁内容和沙箱验证结果。"
        )

        try:
            return self._repo.create_pull(
                title=pr_title,
                body=full_pr_body,
                head=branch_name,
                base=base_branch,
            )
        except GithubException as exc:
            raise RuntimeError(f"创建 Pull Request 失败，修复分支已创建：{branch_name}") from exc

    def _walk_python_files(self, *, path: str, ref: str) -> Iterable[CodeFile]:
        try:
            contents = self._repo.get_contents(path, ref=ref)
        except GithubException as exc:
            raise RuntimeError(f"遍历仓库路径 {path or '/'} 失败。") from exc

        entries = contents if isinstance(contents, list) else [contents]
        for entry in entries:
            if entry.type == "dir":
                if self._should_ignore_directory(entry.path):
                    continue
                yield from self._walk_python_files(path=entry.path, ref=ref)
                continue

            if entry.type != "file" or not entry.path.endswith(".py"):
                continue
            if self._should_ignore_python_file(entry.path):
                continue

            yield CodeFile(path=entry.path, content=self._decode_content(entry))

    def _create_unique_branch(self, *, requested_branch_name: str, base_branch: str) -> str:
        """创建分支；如果名字冲突，自动追加时间戳和递增后缀。"""

        safe_base = self._sanitize_branch_name(requested_branch_name)
        base_ref = self._repo.get_git_ref(f"heads/{base_branch}")
        base_sha = base_ref.object.sha

        candidate_names = [safe_base]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        candidate_names.extend(f"{safe_base}-{timestamp}-{index}" for index in range(1, 6))

        last_error: Optional[GithubException] = None
        for candidate in candidate_names:
            try:
                self._repo.create_git_ref(ref=f"refs/heads/{candidate}", sha=base_sha)
                return candidate
            except GithubException as exc:
                if exc.status == 422:
                    last_error = exc
                    continue
                raise RuntimeError(f"创建分支 {candidate} 失败。") from exc

        raise RuntimeError(f"分支名冲突过多，无法创建修复分支：{safe_base}") from last_error

    @staticmethod
    def _decode_content(content: ContentFile) -> str:
        if content.decoded_content is None:
            raise RuntimeError(f"文件 {content.path} 内容为空或无法解码。")
        return content.decoded_content.decode("utf-8")

    @staticmethod
    def _issue_to_payload(issue: Issue) -> IssuePayload:
        return IssuePayload(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
        )

    @staticmethod
    def _should_ignore_directory(path: str) -> bool:
        parts = {part.lower() for part in path.split("/")}
        return bool(parts & {"test", "tests", "__pycache__", ".venv", "venv"})

    @staticmethod
    def _should_ignore_python_file(path: str) -> bool:
        filename = path.rsplit("/", 1)[-1].lower()
        return filename.startswith("test_") or filename.endswith("_test.py")

    @staticmethod
    def _sanitize_branch_name(branch_name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "-", branch_name.strip())
        cleaned = re.sub(r"/+", "/", cleaned).strip("/.-")
        return cleaned or "codehealer/fix"


class CustomAliyunEmbeddings(Embeddings):
    """阿里云通义千问 OpenAI 兼容 Embedding 适配器。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str],
        model: str = "text-embedding-v1",
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY 不能为空，Embedding 服务需要该密钥。")
        if not base_url:
            raise ValueError("OPENAI_API_BASE 不能为空，阿里云 Embedding 需要兼容接口地址。")

        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量，并确保 input 是干净的 list[str]。"""

        clean_texts = self._normalize_texts(texts)
        response = self._client.embeddings.create(
            model=self._model,
            input=clean_texts,
        )
        sorted_items = sorted(
            response.data,
            key=lambda item: item.index if item.index is not None else 0,
        )
        return [list(item.embedding) for item in sorted_items]

    def embed_query(self, text: str) -> list[float]:
        """生成查询向量，供 Chroma similarity_search 使用。"""

        return self.embed_documents([text])[0]

    @staticmethod
    def _normalize_texts(texts: list[str]) -> list[str]:
        clean_texts: list[str] = []
        for text in texts:
            if text is None:
                normalized = ""
            elif isinstance(text, str):
                normalized = text
            else:
                normalized = str(text)

            clean_texts.append(normalized if normalized.strip() else " ")

        if not clean_texts:
            raise ValueError("Embedding 输入为空，无法生成向量。")
        return clean_texts


class CodebaseRetriever:
    """基于 ChromaDB 的代码级 RAG 检索器。"""

    def __init__(self, *, connector: GitHubConnector) -> None:
        self._connector = connector
        self._code_files = list(connector.iter_python_files())
        self._file_content_by_path = {
            code_file.path: code_file.content for code_file in self._code_files
        }
        self._vector_store: Optional[Chroma] = None

        if not self._code_files:
            raise RuntimeError("仓库默认分支下没有可检索的 Python 源码文件。")

    def build_vector_store(self) -> None:
        """对代码分块并构建进程内 Chroma 向量库。"""

        documents = [
            Document(
                page_content=code_file.content,
                metadata={"file_path": code_file.path},
            )
            for code_file in self._code_files
        ]

        splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=1200,
            chunk_overlap=160,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise RuntimeError("代码分块结果为空，无法构建向量库。")

        embeddings = CustomAliyunEmbeddings(
            model="text-embedding-v1",
            api_key=_get_required_env("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="codehealer_runtime_codebase",
        )

    def search_relevant_file(self, issue_text: str) -> RetrievedCodeFile:
        """根据 Issue 描述检索 Top-1 代码块，并返回所属文件的完整内容。"""

        if self._vector_store is None:
            raise RuntimeError("向量库尚未构建，请先调用 build_vector_store()。")

        results = self._vector_store.similarity_search(issue_text, k=1)
        if not results:
            raise RuntimeError("RAG 未检索到相关代码文件。")

        file_path = results[0].metadata.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise RuntimeError("检索结果缺少 file_path metadata。")

        content = self._file_content_by_path.get(file_path)
        if content is None:
            raise RuntimeError(f"检索到文件 {file_path}，但无法找到完整文件内容。")

        return RetrievedCodeFile(file_path=file_path, content=content)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请在 .env 中配置。")
    return value


def _get_optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None

    value = value.strip()
    return value or None


def _select_issue(connector: GitHubConnector) -> IssuePayload:
    issue_number = _get_optional_env("TARGET_ISSUE_NUMBER")
    if issue_number is not None:
        print(f"[Issue] 使用手动指定的 Issue: #{issue_number}")
        return connector.get_issue(int(issue_number))

    issue_label = os.getenv("TARGET_ISSUE_LABEL", "codehealer")
    print(f"[Issue] 自动扫描带有标签 {issue_label!r} 的 open Issue")
    issue = connector.get_next_open_issue(label=issue_label)
    print(f"[Issue] 自动领取 Issue #{issue.number}: {issue.title}")
    return issue


def _build_llm() -> ChatOpenAI:
    openai_api_key = _get_required_env("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    llm_model_name = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

    return ChatOpenAI(
        model=llm_model_name,
        api_key=openai_api_key,
        base_url=openai_api_base,
        temperature=0.1,
    )


def main() -> None:
    load_dotenv(override=True)

    github_token = _get_required_env("GITHUB_TOKEN")
    target_repo = _get_required_env("TARGET_REPO")

    connector = GitHubConnector(token=github_token, repo_full_name=target_repo)
    issue = _select_issue(connector)

    retriever = CodebaseRetriever(connector=connector)
    retriever.build_vector_store()
    retrieved = retriever.search_relevant_file(issue.task_description)
    target_file_path = retrieved.file_path
    target_code = retrieved.content

    print(f"[RAG] 自动检索定位到疑似 Bug 文件: {target_file_path}")

    initial_state: AgentState = {
        "task_description": issue.task_description,
        "target_code": target_code,
        "test_code": DEMO_TEST_CODE,
        "sandbox_output": "",
        "is_resolved": False,
        "iterations": 0,
    }

    engine = CodeHealerEngine(
        sandbox=DockerSandboxExecutor(timeout_seconds=90),
        coder=LangChainCoder(llm=_build_llm()),
    )
    result = engine.run(initial_state)

    print("=== CodeHealer 执行报告 ===")
    print(f"是否修复成功: {result['is_resolved']}")
    print(f"修复迭代次数: {result['iterations']}")
    print("=== 沙箱输出 ===")
    print(result["sandbox_output"])

    if not result["is_resolved"]:
        failure_comment = (
            "CodeHealer 已尝试自动修复，但沙箱测试仍未通过，因此未创建 PR。\n\n"
            "最后一次沙箱输出：\n"
            "```text\n"
            f"{result['sandbox_output'][:6000]}\n"
            "```"
        )
        connector.comment_on_issue(issue.number, failure_comment)
        connector.add_issue_labels(issue.number, "codehealer-failed")
        print("测试未通过，已停止创建 PR，并已在 Issue 中记录失败原因。")
        return

    branch_name = f"codehealer/issue-{issue.number}-{target_file_path.rsplit('/', 1)[-1]}-fix"
    pr = connector.create_pr(
        file_path=target_file_path,
        new_content=result["target_code"],
        commit_message=f"fix: resolve issue #{issue.number} with CodeHealer",
        branch_name=branch_name,
        pr_title=f"CodeHealer: fix issue #{issue.number}",
        pr_body=(
            f"CodeHealer 根据 Issue #{issue.number} 自动生成了修复。\n\n"
            f"Issue: {issue.title}\n\n"
            f"RAG 定位文件：{target_file_path}\n\n"
            "沙箱验证结果：pytest 已通过。"
        ),
    )
    connector.comment_on_issue(
        issue.number,
        (
            "CodeHealer 已创建候选修复 PR：\n\n"
            f"{pr.html_url}\n\n"
            f"定位文件：`{target_file_path}`\n"
            f"修复迭代次数：{result['iterations']}\n"
            "沙箱验证：pytest passed"
        ),
    )
    connector.add_issue_labels(issue.number, "codehealer-pr-created")
    print(f"已创建 PR: {pr.html_url}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"配置错误：{exc}")
    except RuntimeError as exc:
        print(f"运行失败：{exc}")
    except GithubException as exc:
        print(f"GitHub API 调用失败：{exc.status} {exc.data}")
