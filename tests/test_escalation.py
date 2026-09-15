#!/usr/bin/env python3
"""② escalation/latch 状态机测试 (v1.5.0)。

覆盖: judge 输出解析(think 剥离/长档位优先)、确认计数、锁存、TTL 过期与续期、
fail-open、非 auto 会话跳过、会话键稳定性。
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

def reset():
    m._LATCH.clear()
    m._ESCALATION_CONF.clear()

K = "conv-aaa"

print("== judge 输出解析 ==")
parse = m.parse_escalation_verdict
check("'ok' → ok", parse("ok") == "ok")
check("'OK\\n' → ok (大小写/空白容错)", parse("OK\n") == "ok")
check("'smart' → smart", parse("smart") == "smart")
check("'medium_hard' → medium_hard (不被 medium 截断)", parse("medium_hard") == "medium_hard")
check("带解释 '判定: smart' → smart", parse("判定: smart") == "smart")
check("理由在前, 判定在末尾 → 取最右侧",
      parse("回答基本可用, 但多步推理明显吃力: smart") == "smart")
check("末尾为 ok 时不误取理由里的 medium_hard",
      parse("虽然提到 medium_hard 场景, 本次回答充分: ok") == "ok")
check("'medium' 是合法独立目标档", parse("medium") == "medium")
check("'medium' 出现在 medium_hard 内不被截断", parse("medium_hard") == "medium_hard")
check("ok 独立词保护: 'book' 不命中 ok", parse("book") is None)
check("ok 独立词保护: 'looks good' 不命中 ok", parse("looks good") is None)
check("think 块内的词被剥离 (块内 medium_hard 不算) → 取块外 smart",
      parse("思考: medium_hard 可能不够</" + "thi" + "nk" + ">smart") == "smart")
check("完整 think 块被剥离",
      parse("<" + "thi" + "nk" + ">应该用 medium_hard 吧</" + "thi" + "nk" + ">ok") == "ok")
check("纯 think 无结论 → None (fail-open)", parse("思考中</" + "thi" + "nk" + ">") is None)
check("空字符串 → None", parse("") is None)
check("None 输入 → None", parse(None) is None)

print("== 确认计数: 连续 2 次吃力才锁存 ==")
reset()
a1 = m.apply_escalation_verdict(K, "smart", "cheap")
check(f"第 1 次 smart → 只确认不锁 (返回 {a1!r})", "confirm 1/2" in a1 and K not in m._LATCH)
check("待确认计数存在", m._ESCALATION_CONF.get(K, (0,))[0] == 1)
a2 = m.apply_escalation_verdict(K, "smart", "cheap")
check(f"第 2 次 smart → 锁存 (返回 {a2!r})", "LATCH" in a2 and m._LATCH.get(K, (None,))[0] == "smart")
check("锁存后待确认计数清空", K not in m._ESCALATION_CONF)

print("== ok 判定清零计数(小模型胜任的正面证据) ==")
reset()
m.apply_escalation_verdict(K, "smart", "cheap")
m.apply_escalation_verdict(K, "ok", "cheap")
check("ok 后计数清零", K not in m._ESCALATION_CONF)
m.apply_escalation_verdict(K, "smart", "cheap")
check("清零后需重新累计(不残留)", m._ESCALATION_CONF.get(K, (0,))[0] == 1)

print("== fail-open: None 判定不改状态 ==")
reset()
m.apply_escalation_verdict(K, "smart", "cheap")          # 计数 1
r = m.apply_escalation_verdict(K, None, "cheap")          # judge 挂了
check(f"None → 状态不变 (返回 {r!r})", "fail-open" in r and m._ESCALATION_CONF.get(K, (0,))[0] == 1)
m.apply_escalation_verdict(K, "smart", "cheap")          # 计数 2 → 锁存
check("fail-open 不清零也不消耗, 下一轮仍可锁存", m._LATCH.get(K, (None,))[0] == "smart")

print("== 升级目标必须严格高于当前档位 ==")
reset()
r = m.apply_escalation_verdict(K, "medium_hard", "smart")
check(f"current=smart + target=medium_hard → no-op (返回 {r!r})",
      "no-op" in r and K not in m._ESCALATION_CONF)
r = m.apply_escalation_verdict(K, "smart", "smart")
check("current=smart + target=smart → no-op (不自我升级)", "no-op" in r)
r = m.apply_escalation_verdict(K, "medium", "cheap")
check("current=cheap + target=medium → 有效升级信号",
      m._ESCALATION_CONF.get(K, (0,))[0] == 1)

print("== 非 auto 会话(conv_key=None) 直接跳过 ==")
reset()
r = m.apply_escalation_verdict(None, "smart", "cheap")
check(f"conv_key=None → skip (返回 {r!r})", "skip" in r and not m._LATCH and not m._ESCALATION_CONF)

print("== latch TTL: 过期清除 / 读即续期 ==")
reset()
now = 1_000_000.0
m._LATCH[K] = ("smart", now)
check("TTL 内命中", m.latch_get(K, now + 10) == "smart")
check("读即续期(ts 被刷新)", m._LATCH[K][1] == now + 10)
check("闲置超 TTL → 过期返回 None", m.latch_get(K, now + 10 + m.ESCALATION_LATCH_TTL + 1) is None)
check("过期后条目已删除", K not in m._LATCH)
check("未知会话 → None", m.latch_get("nope", now) is None)
check("conv_key=None → None", m.latch_get(None, now) is None)

print("== 会话键: 稳定性与区分度 ==")
ck = m.conversation_key
b1 = {"messages": [{"role": "user", "content": "帮我重构支付模块"},
                   {"role": "assistant", "content": "好的"},
                   {"role": "user", "content": "继续"}]}
b2 = {"messages": [{"role": "user", "content": "帮我重构支付模块"},
                   {"role": "assistant", "content": "好的"},
                   {"role": "user", "content": "再改一处"},
                   {"role": "assistant", "content": "done"}]}
b3 = {"messages": [{"role": "user", "content": "今天天气怎么样"}]}
check("同一会话(历史增长) → 键稳定", ck(b1) == ck(b2))
check("不同首条消息 → 键不同", ck(b1) != ck(b3))
check("键为 16 位 hex", len(ck(b1)) == 16 and all(ch in "0123456789abcdef" for ch in ck(b1)))
check("纯工具回传(无用户文本) → None",
      ck({"messages": [{"role": "assistant", "content": "x"},
                       {"role": "user", "content": [{"type": "tool_result", "content": "out"}]}]}) is None)
check("多模态文本块可提取",
      ck({"messages": [{"role": "user", "content": [{"type": "text", "text": "看图"}]}]}) is not None)

print("== 投机路由: 未锁存会话 → cheap; 锁存会话 → 锁存档位 ==")
reset()
tier, conv = m.speculative_route(b3)
check(f"新会话 → cheap (实际 {tier!r})", tier == "cheap" and conv == ck(b3))
m._LATCH[conv] = ("smart", m.time.time())
tier2, _ = m.speculative_route(b3)
check(f"锁存会话 → smart (实际 {tier2!r})", tier2 == "smart")
reset()
tier3, conv3 = m.speculative_route({"messages": [{"role": "system", "content": "s"}]})
check("无用户文本 → cheap 且 conv_key=None", tier3 == "cheap" and conv3 is None)

print("== 调度门槛: 空响应不评估 ==")
reset()
m._schedule_escalation_judge("req", "", "cheap", K, "tok")
check("空响应 → 不创建后台任务", len(m._BG_TASKS) == 0)
m._schedule_escalation_judge("req", "x" * 100, "cheap", None, "tok")
check("conv_key=None → 不创建后台任务", len(m._BG_TASKS) == 0)
reset()

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("ALL GREEN")
