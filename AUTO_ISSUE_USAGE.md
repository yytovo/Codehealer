# CodeHealer 自动领取 Issue 模式

当前入口文件为 `run.py`。

## 两种运行方式

手动指定 Issue：

```env
TARGET_REPO=owner/repo
TARGET_ISSUE_NUMBER=3
```

自动领取 Issue：

```env
TARGET_REPO=owner/repo
TARGET_ISSUE_NUMBER=
TARGET_ISSUE_LABEL=codehealer
```

当 `TARGET_ISSUE_NUMBER` 留空时，CodeHealer 会自动扫描目标仓库中带有 `TARGET_ISSUE_LABEL` 标签的 open Issue，并处理最早创建的一条。

## 演示步骤

1. 在目标仓库创建一个 Bug Issue。
2. 给该 Issue 添加 `codehealer` 标签。
3. 确认 `.env` 中 `TARGET_ISSUE_NUMBER` 为空。
4. 运行：

```bash
python run.py
```

5. CodeHealer 会自动领取该 Issue、定位代码、修复、测试并创建 PR。
6. 成功后，CodeHealer 会在 Issue 下评论 PR 链接，并添加 `codehealer-pr-created` 标签。

这个标签机制相当于人工授权入口：不是所有 Issue 都会被 Agent 自动处理，只有明确打上 `codehealer` 标签的问题才会进入自动修复流程。

