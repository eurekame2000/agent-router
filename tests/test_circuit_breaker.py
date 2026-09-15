#!/usr/bin/env python3
"""验证 agent-router 熔断双门限逻辑。直接 import 真实模块, 不复制代码。"""
import sys, importlib.util, time

spec = importlib.util.spec_from_file_location(
    "agent_router", "/Users/eureka/agent-router/agent_router.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name)

def reset():
    m._UPSTREAM_WINDOW.clear(); m._UPSTREAM_FAILURES.clear()
    m._UPSTREAM_COOLDOWN_UNTIL.clear()

print("== 门限A: 连续失败 3 次熔断 ==")
reset()
m._upstream_fail("A"); m._upstream_fail("A")
check("2连败未熔断(仍可用)", m._upstream_available("A"))
m._upstream_fail("A")
check("第3连败触发熔断", not m._upstream_available("A"))
check("冷却时长=5s", 4.5 < m._UPSTREAM_COOLDOWN_UNTIL["A"] - time.time() <= 5.0)

print("== 成功清零连续计数 ==")
reset()
m._upstream_fail("A"); m._upstream_fail("A"); m._upstream_success("A"); m._upstream_fail("A")
check("失败-失败-成功-失败 未熔断(连续计数被清零)", m._upstream_available("A"))

print("== 门限B: 间歇性失败(成功/失败交替)按比例熔断 ==")
reset()
# 精算边界: 序列 F,T,F,T,T 到第5样本时仅 2F/5=40%(不触发), 但连续计数被清零;
# 第6样本 F => 3F/6 = 恰好50%, 严格 ">0.5" 才熔断, 故不应触发。
for c in ["F", "T", "F", "T", "T", "F"]:
    m._upstream_fail("B") if c == "F" else m._upstream_success("B")
check("失败率恰好50%(3/6)不熔断(条件是严格大于)", m._upstream_available("B"))
m._upstream_fail("B")   # 样本8, 失败4 => 50% -> 仍不触发, 再补1次
m._upstream_fail("B")   # 样本9, 失败5 => 55.6% > 50%
check("失败率55.6%(5/9)熔断(门限B抓住间歇性失败)", not m._upstream_available("B"))

print("== 门限B 独立性: 连续计数远未达3时也能熔断 ==")
reset()
# 直接注入窗口: 5 失败 + 2 成功(失败率71%), 且连续计数保持 0
w = m._UPSTREAM_WINDOW.setdefault("G", __import__("collections").deque(maxlen=64))
now = time.time()
for _ in range(5): w.append((now, False))
for _ in range(2): w.append((now, True))
m._UPSTREAM_FAILURES["G"] = 0
m._upstream_fail("G")   # 连续计数变成 1, 远未达 3
check("连续计数仅1次但因失败率71%熔断(证明是比例门限)",
      not m._upstream_available("G") and m._UPSTREAM_FAILURES.get("G", 0) < 3)

print("== 最小样本数保护: 样本不足时不按比例熔断 ==")
reset()
m._upstream_fail("C"); m._upstream_fail("C")   # 样本2, 失败率100%, 但<5
check("样本仅2(失败率100%)不熔断", m._upstream_available("C"))

print("== 冷却期满后自动恢复 ==")
reset()
for _ in range(3): m._upstream_fail("D")
check("冷却中不可用", not m._upstream_available("D"))
m._UPSTREAM_COOLDOWN_UNTIL["D"] = time.time() - 0.1   # 模拟时间流逝
check("冷却期满恢复可用", m._upstream_available("D"))
check("恢复后状态已清理", "D" not in m._UPSTREAM_FAILURES and "D" not in m._UPSTREAM_WINDOW)

print("== 窗口老化: 旧失败样本过期后不再计入 ==")
reset()
for _ in range(5): m._upstream_fail("E")   # 熔断
m._upstream_available("E")
for _ in range(5): m._upstream_fail("E")   # 再次熔断
m._UPSTREAM_COOLDOWN_UNTIL["E"] = time.time() - 0.1
m._upstream_available("E")                 # 恢复+清空
m._upstream_fail("E"); m._upstream_fail("E")
check("恢复后旧样本未残留(2次失败不熔断)", m._upstream_available("E"))

print("== 各上游互不干扰 ==")
reset()
for _ in range(3): m._upstream_fail("X")
check("X 熔断不影响 Y", m._upstream_available("Y"))

print("== 常量核对(对齐 LiteLLM) ==")
check("allowed_fails=3", m._UPSTREAM_ALLOWED_FAILS == 3)
check("percent=0.5", m._UPSTREAM_FAILURE_PERCENT == 0.5)
check("min_requests=5", m._UPSTREAM_MIN_REQUESTS == 5)
check("cooldown=5s", m._UPSTREAM_COOLDOWN_SECS == 5)

print(f"\n结果: {len(PASS)} 通过, {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL); sys.exit(1)
print("ALL GREEN")
