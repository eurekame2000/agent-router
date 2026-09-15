# GitHub 开源 LLM 路由/网关方案调研报告

日期: 2026-09-15
目的: 为 agent-router 的 ② escalation/latch 重构 + ③ 阈值校准寻找可借鉴设计
**结论: 不引入任何库, 只吸收算法思想**（理由见第 4 节）

---

## 1. 候选筛选（GitHub API, 按 star 排序）

| 仓库 | ★ | 语言 | 定位 | 相关性 |
|---|---|---|---|---|
| bytedance/deer-flow | 82k | Python | SuperAgent harness | ✗ 应用层, 非路由 |
| **BerriAI/litellm** | 58.8k | Python+Rust | AI Gateway, 100+ LLM | **✓✓ 熔断/冷却已对齐** |
| QuantumNous/new-api | 48k | Go | 模型聚合分发 | ✗ 计费/分发为主 |
| **Portkey-AI/gateway** | 33k | TS | AI Gateway + guardrails | **✓ 条件路由/嵌套 fallback** |
| tensorzero/tensorzero | 11.7k | Rust | LLMOps + gateway | ✓ 重试退避 |
| coaidev/coai | 9.3k | TS | 多租户方案 | ✗ |
| maximhq/bifrost | 8.1k | Go | 企业网关 | ✓ 路由规则表 |
| katanemo/plano | 7.1k | Rust | 智能数据面 | ✓ 语义路由 |
| tbphp/gpt-load | 6.8k | Go | 多凭证网关 | ✗ |

最终精读 3 个: **LiteLLM**（熔断/延迟策略）、**Portkey**（条件路由/嵌套 fallback）、**TensorZero**（退避）。

---

## 2. 逐个可借鉴点

### 2.1 LiteLLM — 熔断/冷却（**已对齐, 无需改**）

源码: `litellm/router_utils/cooldown_handlers.py` + `litellm/constants.py`

其 **v2 冷却逻辑**（`_should_cooldown_deployment:332-404`）判据：

```python
# constants.py:70-76,129-133
DEFAULT_FAILURE_THRESHOLD_PERCENT   = 0.5   # 失败率阈值
DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS = 5   # 最小样本
DEFAULT_COOLDOWN_TIME_SECONDS       = 5     # 冷却时长
SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD = 1000
```

```python
# cooldown_handlers.py:388-399
if exception_status_int == 429 and not is_single_deployment_model_group:
    return True                                    # 429 直接冷却
elif percent_fails == 1.0 and total >= SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD:
    return True                                    # 全败 + 高流量
elif percent_fails > 0.5 and total >= 5 and not is_single_deployment_model_group:
    return True                                    # 失败率门限  ← 我们已实现
```

**我方 agent-router v1.3.0 的熔断参数与此完全一致**：
连续 3 败（`_UPSTREAM_ALLOWED_FAILS`）或 60s 窗口样本≥5 且失败率>50%（门限 B），冷却 5s。
→ **无需改动**，调研只作确认。

**可借鉴但未采纳**：它对 `is_single_deployment_model_group`（池内只剩 1 个模型）
时**跳过失败率冷却**——避免把唯一可用模型也锁死。我方池内最后候选也应如此。

### 2.2 LiteLLM — 延迟路由（**重要发现，已验证**）

源码: `litellm/router_strategy/lowest_latency.py`

**核心: 延迟必须按输出 token 归一化**（`lowest_latency.py:295-305`）：

```python
completion_tokens = _usage.completion_tokens
response_seconds  = response_ms  # 已归一化为秒
normalized_value  = safe_divide_seconds(response_seconds, completion_tokens)
final_value = float(normalized_value)   # ← 每 token 秒数 = 1/吞吐
```

并**保留最近 N 个样本的滑动窗口**（`max_latency_list_size=10`）：

```python
if len(latency_list) < max_latency_list_size:
    latency_list.append(final_value)
else:
    latency_list = latency_list[1:] + [final_value]   # FIFO 滑窗
```

超时/失败记 `latency = 1000.0`（惩罚值）。

**这与我方独立分析结论完全吻合**：我此前算出的「glm-5.3-flash 慢 5-10 倍」
经按 output_tokens 归一化后**结论反转为吞吐基本相同**（104.7 vs 117.9 tok/s）。
LiteLLM 用工程实现印证了这个方法论。

**借鉴点**：若后续要做「按实测延迟选池内模型」，必须用 `latency/output_tokens`
而非裸 `latency_ms`，且用 FIFO 滑窗（10 个样本）而非全量均值。
**当前不采纳**——我方池是「有序降级列表」不是「按延迟动态排序」，引入动态排序
会增加不确定性。仅记录为将来的优化选项。

### 2.3 Portkey — 嵌套 fallback + 条件路由

源码: `cookbook/getting-started/resilient-loadbalancing-with-failure-mitigating-fallbacks.md`

**核心设计: `strategy` 可递归嵌套**——targets 里的任意一项本身可以是另一个
strategy，形成树状降级：

```javascript
{ strategy: { mode: 'loadbalance' },
  targets: [
    { virtual_key: ANTHROPIC, weight: 0.5, override_params: {model: 'claude-3-opus'} },
    { strategy: { mode: 'fallback' },        // ← 嵌套 strategy
      weight: 0.5,
      targets: [ {key: OPENAI}, {key: AZURE} ] }
  ]}
```

**条件路由**（`conditional-routing.ts`）：按请求元数据分流，与我们的 `tier` 语义一致：

```javascript
{ strategy: { mode: 'conditional',
    conditions: [
      { query: { 'metadata.user_plan': {$eq: 'paid'} },
        then: { model: 'claude-3-5-sonnet' } },     // 付费→强模型
      { query: { 'metadata.user_plan': {$eq: 'free'} },
        then: { model: 'gpt-4o' } } ] }}
```

**借鉴点**：概念与我方 `MODEL_POOLS[tier]` + 池内降级同构，**我们的设计方向是对的**。
`weight: 0` 可动态摘除 target（等价于我方「熔断后跳过」）。

**未采纳**：loadbalance 权重分流不适配单用户本地路由（无需横向扩容）。

### 2.4 TensorZero — 重试退避

源码: `crates/tensorzero-core/src/utils/retries.rs`

```rust
pub struct RetryConfig { num_retries: usize, max_delay_s: f32 }   // 默认 0 次 / 10s
fn get_backoff(&self) -> ExponentialBuilder {
    ExponentialBuilder::default()
        .with_jitter()                              // ← 抖动防惊群
        .with_max_delay(Duration::from_secs_f32(self.max_delay_s))
        .with_max_times(self.num_retries)
}
func.retry(backoff).when(Error::is_retryable)       // 仅可重试错误重试
```

**借鉴点**: ①`with_jitter()` 指数退避加抖动；②`is_retryable` 显式区分可重试/不可重试
（我方 401/403 不应重试，429/5xx 应重试——已实现）；③默认 `num_retries: 0` 保守。

---

## 3. 对我方 agent-router 的具体建议

| 项 | 现状 | 建议 | 优先级 |
|---|---|---|---|
| 熔断参数 | 3 败 / 50% ≥5 样本 / 冷却 5s | **与 LiteLLM v2 一致, 保持** | — |
| 池内最后候选 | 可能被锁死 | 借鉴 `is_single_deployment_model_group`：最后候选不熔断 | P2 |
| 延迟归一化 | 曾误判 glm 慢 5-10x | 若做延迟路由, 必须 `latency/output_tokens` | P2（暂不做） |
| 重试退避抖动 | 无 | 借鉴 `with_jitter()` | P2 |
| **难度判定** | **关键词 score + 阈值 0.8** | **← 真正的问题, 见下** | **P0** |
| escalation/latch | 无 | Portkey 嵌套 strategy 概念可参考 | P1 |

**注意**: 调研中**没有任何方案**解决「关键词难度评分」的问题——LiteLLM/Portkey/
TensorZero 都假设**外部指定模型或元数据**，不做 prompt 难度推断。
→ 难度判定必须自研，这是 ②/③ 的核心。

---

## 4. 为什么「不引库」

| 方案 | 阻塞原因 |
|---|---|
| litellm | 58k★ 但引入 Python 全栈网关依赖（含 Redis、proxy 框架），我方只需 2000 行单文件；其熔断逻辑我们已独立实现且参数一致 |
| Portkey gateway | TypeScript 服务，需独立部署 Node 进程；条件路由能力我方已用 `tier` 覆盖 |
| tensorzero | Rust 编译期依赖重；仅退避算法可借鉴（30 行可自实现） |
| Switchyard (此前调研) | 算法最对口（escalation/decision），但要求 Python≥3.12 且 decision API 未上 PyPI |
| RouteLLM | 已停更 |

**结论**: 所有候选的能力点，我方要么已实现、要么可用 <50 行自研。
引入完整网关会带来 Redis/进程/依赖链成本，与「单文件嵌入式路由」定位冲突。

---

## 5. 附: 调研中确认的 ③ 阈值校准结论

用带标签样本（28 条: 10 trivial / 18 nontrivial）+ 阈值权重联合搜索 + 留一交叉验证：

| 实现 | 准确率 |
|---|---|
| 现行 `MEDIUM_THRESHOLD=0.8` | 71% |
| 同公式 `threshold≈0.5` | **82~93%** |
| 最优权重组合（in-sample 上界） | 93%（无实质提升） |
| 留一交叉验证（泛化） | 93% |

**结论**: 关键词公式本身够用，**问题是阈值 0.8 定得过高**——8 个错例全是
「真任务被误判为 trivial → 走 cheap 池」：
`解释一下什么是快速排序(0.65)` / `重构这个类(0.7)` / `设计高并发秒杀架构(0.7)` /
`给项目加登录接口(0.5)` 等。

将阈值降到 0.5 后错误从 8 降到 5，且剩余错误方向变成「闲聊误走 judge」（更安全，
因为 judge 本身是廉价的分类调用，误判只会多点一次模型调用而非降低质量）。

> 详细缺陷分析见 `agent-router-路由缺陷分析.md`（P0-1 auto 路由 medium 不可达）
