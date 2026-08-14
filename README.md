# agent-router

本地 Agent 模型路由网关 — 按任务难度路由到 Ollama Cloud 不同模型。
同时暴露 OpenAI 和 Anthropic 两套协议，兼容 Hermes 和 Claude Code。

## 架构

```
Hermes (OpenAI协议)  ─┐
                      ├─→ agent_router (难度路由) ─→ ollama.com/v1
Claude (Anthropic)  ─┘
```

## 难度路由

| 档位 | 模型 | 场景 |
|---|---|---|
| cheap | `deepseek-v4-flash:0731` | 简单任务（短 prompt / 闲聊） |
| medium | `deepseek-v4-pro:preview` | 常规任务（写代码 / 总结） |
| smart | `glm-5.2` | 复杂任务（重构 / 调试 / 安全审计 / 长输入） |

- 规则打分：关键词 + 长度 + 简单信号压制，分数 <0.7 直判 cheap，≥0.9 直判 smart
- 模糊地带（0.7–0.9）：用 flash 二次 LLM judge（失败保守回退 medium）
- Token 从 `~/.hermes/config.yaml` 的 `api_key` 动态读取，不硬编码

## 端点

| 端点 | 协议 | 用途 |
|---|---|---|
| `GET /health` | — | 健康检查 |
| `GET /v1/models` | — | 模型档位列表 |
| `POST /v1/chat/completions` | OpenAI | Hermes |
| `POST /v1/messages` | Anthropic | Claude Code |

## 部署

```bash
python3 agent_router.py --port 18001   # 手动运行
```

### launchd 自启动（macOS）

1. 安装 `ai.hermes.agent-router.plist.template`（`<USER>` 替换为实际用户名）：
   ```bash
   sed 's|<USER>|'$USER'|g' ai.hermes.agent-router.plist.template \
     > ~/Library/LaunchAgents/ai.hermes.agent-router.plist
   launchctl load ~/Library/LaunchAgents/ai.hermes.agent-router.plist
   ```
2. 重启：`launchctl kickstart -k "gui/$(id -u)/ai.hermes.agent-router"`
3. 日志：`~/.hermes/logs/router.error.log`（⚠️ 注意是 error.log，logging 走 stderr）

### Hermes 客户端配置

`~/.hermes/config.yaml`：

```yaml
model:
  base_url: "http://127.0.0.1:18001/v1"
  provider: "ollama-cloud"
```

**关键**：`~/.hermes/.env` 必须包含（否则 Hermes 连本机端口会走 lmclient 代理被拒 → 502）：

```bash
NO_PROXY=127.0.0.1,localhost,open.feishu.cn,open.larksuite.com,feishu.cn,larksuite.com
no_proxy=127.0.0.1,localhost,open.feishu.cn,open.larksuite.com,feishu.cn,larksuite.com
```

## 故障排查

参见 Claude Code skill `hermes-gateway-502`（`~/.claude/skills/hermes-gateway-502/`）：
- 502 根因：`hermes_bootstrap.py` 把 NO_PROXY 改写成纯飞书域名 → 连 127.0.0.1 走代理被拒
- 快速判定：`router.error.log` 无 `openai收到` + 进程 `no_proxy` 无 127.0.0.1

## 历史

- `capture_proxy.py`（18000）为早期纯转发层，已废弃移除，Hermes 直连 18001
- `cc-switch`（15721）Codex 配置损坏，已弃用，改用 ollama.com 直连
