#!/usr/bin/env python3
"""路由不变量回归测试(v1.5.0): tier 为权威, 池内不被二次改写。

历史: P0-1(2026-09-15) 旧实现 MEDIUM_THRESHOLD=0.8 双语义冲突, judge 返回 medium
必被升级为 medium_hard, 导致 medium 池在 auto 路由下不可达。v1.4.0 起 tier 权威。
v1.5.0 进一步删除规则打分(difficulty_score/关键词表), 判定权全交事后评估。
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

print("== 常量: 旧机制已彻底移除 (v1.5.0) ==")
check("无 MEDIUM_THRESHOLD(旧常量已删)", not hasattr(m, "MEDIUM_THRESHOLD"))
check("无 TRIVIAL_THRESHOLD(v1.5.0 判定权移交)", not hasattr(m, "TRIVIAL_THRESHOLD"))
check("无 difficulty_score(规则打分已删)", not hasattr(m, "difficulty_score"))
check("无 EASY_KEYWORDS(关键词表已删)", not hasattr(m, "EASY_KEYWORDS"))
check("无 route_by_difficulty(请求前路由已删)", not hasattr(m, "route_by_difficulty"))
check("无 llm_judge(请求前分类器已删)", not hasattr(m, "llm_judge"))
check("有 judge_escalation(事后评估)", hasattr(m, "judge_escalation"))
check("有 speculative_route(投机路由)", hasattr(m, "speculative_route"))
check("有 apply_escalation_verdict(状态机)", hasattr(m, "apply_escalation_verdict"))

print("== resolve_pool: tier 权威, 任意 prompt 均不改写池 ==")
tricky = "请重构这个类并审计安全性，设计分布式架构，调试性能问题"
for tier in ("cheap", "medium", "medium_hard", "smart"):
    pool = m.resolve_pool(tier, tricky)
    check(f"tier={tier} + 高难 prompt → 仍是 {tier} 池 (首元素 {pool[0]!r})",
          pool is m.MODEL_POOLS[tier] and pool[0] == m.MODEL_POOLS[tier][0])

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

print("== 显式档位不参与 escalation (conv_key=None 语义) ==")
body_explicit = {"model": "smart", "messages": [{"role": "user", "content": "架构设计"}]}
check("explicit smart → resolve_tier 命中", m.resolve_tier("smart") == "smart")
check("auto → resolve_tier 为 None", m.resolve_tier("auto") is None)
check("unknown 模型名 → resolve_tier 为 None", m.resolve_tier("nope-xyz") is None)

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("ALL GREEN")
