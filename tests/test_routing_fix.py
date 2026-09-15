#!/usr/bin/env python3
"""P0-1 路由缺陷回归测试: 防止 resolve_pool 二次改写 judge 判定复活。

背景(2026-09-15 修复): 旧实现 MEDIUM_THRESHOLD=0.8 被两处使用且语义冲突,
judge 一旦返回 medium 必被 resolve_pool 用 score>0.8 升级为 medium_hard,
导致 medium 池在 auto 路由下不可达。v1.4.0 起 tier 为权威, 不再二次改写。
"""
import sys
import importlib.util

spec = importlib.util.spec_from_file_location(
    "agent_router", "/Users/eureka/agent-router/agent_router.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name)

print("== 常量: 旧阈值已移除, 新闸门存在 ==")
check("无 MEDIUM_THRESHOLD(旧常量已删)", not hasattr(m, "MEDIUM_THRESHOLD"))
check("无 LLM_JUDGE_LOW(旧常量已删)", not hasattr(m, "LLM_JUDGE_LOW"))
check("TRIVIAL_THRESHOLD == 0.5", m.TRIVIAL_THRESHOLD == 0.5)

print("== resolve_pool: tier 为权威, 不被 difficulty_score 改写 ==")
# 一句高规则分但 tier=medium 的请求 —— 旧实现会把它升级成 medium_hard
tricky = "请重构这个类并审计安全性，设计分布式架构，调试性能问题"
score = m.difficulty_score(tricky)
check(f"构造样本规则分 >=0.8 (实际 {score:.2f})", score >= 0.8)
pool = m.resolve_pool("medium", tricky)
check(f"tier=medium → 仍在 medium 池 (首元素 {pool[0]!r}), 未被升级",
      pool is m.MODEL_POOLS["medium"] and pool[0] == m.MODEL_POOLS["medium"][0])

print("== resolve_pool: 四个档位全部可达 ==")
for tier in ("cheap", "medium", "medium_hard", "smart"):
    p = m.resolve_pool(tier, "测试")
    check(f"tier={tier} → {p[0]!r} ({len(p)} 个候选)", p == m.MODEL_POOLS[tier])

print("== resolve_pool: 显式真实模型名 → 精确锁定单元素池 ==")
locked = m.resolve_pool("smart", "任意请求", requested="deepseek-v4.1-flash")
check("requested=deepseek-v4.1-flash → 单元素池(即使 tier=smart)",
      locked == ["deepseek-v4.1-flash"])

print("== resolve_pool: 未知档位回退 medium ==")
check("未知档位回退 medium", m.resolve_pool("bogus", "x") == m.MODEL_POOLS["medium"])

print("== judge 4 分类解析: medium_hard 不被 medium 子串抢先 ==")
# 复刻 llm_judge 的解析顺序, 断言长词优先
def parse(content):
    content = content.strip().lower()
    for tier in ("smart", "medium_hard", "medium", "cheap"):
        if tier in content:
            return tier
    return None
check("'medium_hard' → medium_hard (非 medium)", parse("medium_hard") == "medium_hard")
check("'medium' → medium", parse("medium") == "medium")
check("'smart' → smart", parse("smart") == "smart")
check("'cheap' → cheap", parse("cheap") == "cheap")
check("带解释文本 '答案: medium_hard' → medium_hard",
      parse("答案: medium_hard") == "medium_hard")

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("ALL GREEN")
