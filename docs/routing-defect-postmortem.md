# agent-router 难度路由缺陷分析（③ 阈值校准的产出）

日期: 2026-09-15
状态: **已定位 2 个 P0 缺陷 + 1 个架构缺陷，待修复方案确认**

---

## 结论先行

原计划是「校准 `MEDIUM_THRESHOLD=0.8`」。实测后发现：**阈值不是问题，路由逻辑本身有 bug**。
校准数据（本次新增 `auto:<tier>` 埋点）尚未积累，但代码审查 + e2e 实测已经钉死缺陷。

---

## P0-1: auto 路由下 judge 的 `medium` 判定必然被篡改为 `medium_hard`

**决定性证据（e2e 实测，2026-09-15 17:55）**

| 请求 | 记录 `request_model` | 实际落地 `model` | 期望 | 判定 |
|---|---|---|---|---|
| `auto` + 简单(score<0.8) | `auto:cheap` | deepseek-v4.1-flash | cheap 池首 | ✅ |
| `pro` + 简单(score<0.8) | `pro` | deepseek-v4.1-flash | medium 池首 | ✅ |
| `auto` + 中等难(score=0.95) | `auto:medium` | **glm-5.3-flash** | medium 池首 | ❌ |

第三条是 smoking gun：**记录档位是 `medium`，落地模型却是 `medium_hard` 池首**
（`MODEL_POOLS["medium"][0]` = `deepseek-v4.1-flash`）。
档位标签与实际执行池**不一致**，且日志 `difficulty: score=1.00 > 0.8 → medium_hard` 佐证。

**根因**: `resolve_pool()` 与 `route_by_difficulty()` 对同一个 `difficulty_score` 语义冲突。

```python
# route_by_difficulty()  ← 决定 tier
score = difficulty_score(prompt)
if score < LLM_JUDGE_LOW(0.8):  return "cheap"      # 快路径, 不调 judge
return await llm_judge(prompt, token)               # 仅 score>=0.8 才调 judge

# resolve_pool()  ← 决定用哪个池
if tier == "medium" and difficulty_score(prompt) > MEDIUM_THRESHOLD(0.8):
    return MODEL_POOLS["medium_hard"]               # 覆盖 judge 的判定
```

judge **只在 `score >= 0.8` 时被调用** → 它若返回 `medium`，则 `resolve_pool` 里
`score > 0.8` **必然成立** → 立刻被篡改为 `medium_hard`。

**精确影响**: 在 auto 路由下，`MODEL_POOLS["medium"]` 不可达——它要求
`tier=="medium"` 且 `score<=0.8`，但 `tier=="medium"` 在 auto 下只能来自 judge，
而 judge 的调用前提就是 `score>=0.8`。两个条件互斥。

唯一能进入 medium 池的路径是**显式传 `model=pro`**（绕过 judge 直接得 tier=medium）
且 prompt 恰好 `score<=0.8`。

**影响面**: medium_hard 池首 glm-5.3-flash 承担 36% 流量（DB 实测），
其中相当部分是 judge 本想判 `medium` 而被篡改上来的。

---

## P0-2: `difficulty_score` 与真实难度反相关

用 28 条带标签样本评测现行实现（`/tmp/calib2.py`）: **准确率 71%**。

误判样本（真任务被判为「简单」→ 直接走 cheap 池，跳过 judge）:

| prompt | score | 问题 |
|---|---|---|
| 给这个项目加一个登录接口 | 0.50 | 无关键词命中 |
| 解释一下什么是快速排序 | 0.65 | 只有「解释」1 个中等词 |
| 总结一下这篇文章的要点 | 0.65 | 同上 |
| 重构这个类，把职责拆分清楚 | 0.70 | 只加 1 个复杂词 |
| 设计一个高并发秒杀系统的架构 | 0.70 | 只加 1 个复杂词 |
| 帮我优化一下这段查询的性能 | 0.70 | 只加 1 个复杂词 |
| 帮我看看这个报错是什么原因 | 0.65 | 无关键词命中 |

反例（简单任务被判为「难」→ 走 medium_hard）:

| prompt | score | 问题 |
|---|---|---|
| 写代码实现一个函数 | 0.95 | 命中「写代码/实现/代码/函数」多个近义词 |
| 对比一下 PostgreSQL 和 MySQL 的区别 | 0.95 | 命中「对比/区别」+ 长度 |
| 请实现一个二分查找算法并写单元测试 | 0.95 | 命中多个中等词 |

**根因**: score 是**关键词计数**且关键词高度重叠（「写代码」「代码」「函数」互相重复计数），
与真实语义难度解耦。

**网格搜索结论**: 尝试 base/keyword权重/长度加成的组合调参，最优仅 **46%**，
比基线 71% 更差 → **关键词计数是弱分类器，调参救不了，需换判定方式**。

---

## P1: 判定链架构缺陷

判定分散在三处，语义重叠且互相干扰:

1. `route_by_difficulty()` — score<0.8 → cheap；否则调 LLM judge
2. `resolve_pool()` — tier=='medium' 且 score>0.8 → medium_hard
3. `llm_judge()` — 用 flash 判 cheap/medium/smart

同一个 score 被用于「是否需要 judge」和「用哪个池」两个不同决策。
且 `LLM_JUDGE_LOW` 与 `MEDIUM_THRESHOLD` 都是 0.8，纯属巧合耦合。

日志证据: `llm_judge` 历史失败率可观（2026-08-30 ~ 09-08 共 19 次
ConnectError/ReadTimeout/500），全部 `回退 medium` —— 在旧逻辑下这意味着
回退 tier=medium 后又因 score>0.8 升到 medium_hard。

---

## 延迟数据澄清（推翻此前结论）

此前认为「glm-5.3-flash 平均 22-72s，比 deepseek 慢 5-10 倍」。**经核对是辛普森悖论**:

| 模型 | 样本 | 平均输出 tok | 平均延迟 | 吞吐 tok/s |
|---|---|---|---|---|
| glm-5.3-flash (全部) | 1165 | — | 22.6s | — |
| glm-5.3-flash (输出500-2000) | — | — | — | **104.7** |
| deepseek-v4.1-flash (输出500-2000) | — | — | — | **117.9** |
| deepseek-v4-flash:0731 (输出500-2000) | — | — | — | **104.9** |

慢请求 avg_out **8635 tok**，快请求仅 **738 tok** → 延迟差异来自**输出长度**，不是模型慢。
同长度区间下三者吞吐几乎相同。**因此「降阈值让流量离开 glm」的动机不成立。**

---

## 建议修复方向（待确认）

1. **P0-1 立即修**: `resolve_pool` 不应重新计算 score 决定子池。应让 tier 直接表达
   `medium` / `medium_hard`（即 `route_by_difficulty` / `llm_judge` 返回四档而非三档），
   消除重复判定。
2. **P0-2 换判定方式**: 关键词计数弃用或降级为「快速否决」信号（只用于
   `EASY_KEYWORDS 命中且极短` 这种高精度场景），主判交 LLM judge。
   同时把 judge 的失败回退从 `medium` 改为更保守策略。
3. **P1 收敛**: 判定逻辑收到单一函数，`LLM_JUDGE_LOW` / `MEDIUM_THRESHOLD` 合并为一个门限。

---

## 待办

- [ ] 用户确认修复方向
- [ ] 实施 P0-1（消除 medium 池不可达）
- [ ] 实施 P0-2（换判定方式或降级关键词为否决信号）
- [ ] judge 失败回退策略复核
- [ ] 埋点数据积累后回看 auto:<tier> 真实分流比
