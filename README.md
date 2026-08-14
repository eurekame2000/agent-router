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

## 显式模型路由

除难度自动路由外，可通过请求的 `model` 字段强制指定档位：

| model 别名 | 行为 |
|---|---|
| `auto` / 未知名 | 难度自动路由（默认） |
| `flash` / `cheap` / `deepseek-v4-flash:0731` | 固定 → flash |
| `pro` / `medium` / `deepseek-v4-pro:preview` | 固定 → pro |
| `smart` / `glm` / `glm-5.2` | 固定 → smart |

`GET /v1/models` 返回 `auto / flash / pro / smart` 四项，Hermes 的 `/mode` picker 直接选用。

### Hermes 显式切换（通过 custom provider）

在 `~/.hermes/config.yaml` 的 `custom_providers` 加 4 个指向网关的 provider：

```yaml
custom_providers:
  - name: "Router-auto"
    model: "auto"
    base_url: "http://127.0.0.1:18001/v1"
    api_key: "dummy"      # 网关转发时用自己的 token，忽略此字段
  - name: "Router-flash"
    model: "flash"
    base_url: "http://127.0.0.1:18001/v1"
    api_key: "dummy"
  - name: "Router-pro"
    model: "pro"
    base_url: "http://127.0.0.1:18001/v1"
    api_key: "dummy"
  - name: "Router-glm"
    model: "smart"
    base_url: "http://127.0.0.1:18001/v1"
    api_key: "dummy"
```

然后 `hermes model` 选对应 provider，或 `/mode` picker 选 auto/flash/pro/smart。

**⚠️ 注意**：
1. `model.default` 必须保持 `auto`——若填真实模型名（如 `deepseek-v4-flash:0731`）会命中别名表导致永远固定档位，破坏难度路由。
2. `hermes config set 'custom_providers' '[...]'` 会把数组存成 JSON 字符串而非原生 YAML 列表——直接编辑 config.yaml 或手动改回列表格式。

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
