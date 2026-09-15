#!/usr/bin/env python3
"""
本地 Agent 模型路由网关 — 按任务难度选档位, 每档位一个候选模型池(Ollama Cloud)
同时暴露 OpenAI 和 Anthropic 两套协议，兼容 Hermes 和 Claude Code。

架构:
  Hermes (OpenAI协议)  ─┐
                        ├─→ 本网关 (难度路由) ─→ Ollama Cloud
  Claude (Anthropic)  ─┘

档位模型池(2026-09-15, 智谱无额度剔除, 全走 Ollama Cloud):
  cheap  — 简单任务: deepseek-v4.1-flash → deepseek-v4-flash:0731 → gpt-oss:20b
  medium — 常规任务, 按难度分流:
           ≤0.8: deepseek-v4.1-flash → kimi-k2.7-code → minimax-m2.7 → deepseek-v4-flash:0731
           >0.8: glm-5.3-flash → kimi-k2.7-code → minimax-m3
  smart  — 复杂任务: glm-5.3 → kimi-k3 → deepseek-v4-pro:0813 → qwen3.5:397b → minimax-m3 → glm-5.3-flash
  池内降级: 429/5xx/请求异常自动换下一个模型; 每模型独立熔断(连续3败 或 60s内失败率>50%且样本≥5 → 冷却5s)。

难度自动路由(② 投机启动 + 事后升级, 2026-09-16):
  不再在请求前同步调用 judge(省掉每轮 judge 的延迟 + token):
    1) 新会话投机走 cheap 池(零延迟启动);
    2) 响应完成后**异步**评估(用户请求+模型回答一起给 judge): ok / medium_hard / smart;
    3) 连续 N=2 次"吃力"判定 → 升级锁存(latch): 该会话后续请求固定走高档位池;
    4) latch 闲置 TTL 2h 过期; judge 失败/超时 → fail-open(不改变状态, 不误升级);
    5) 会话键 = 首条用户消息前512字符 sha1[:16] —— 多轮请求带全量历史, 此键在会话内稳定。
  显式档位/别名/真实模型名 → 固定路由, 不参与 escalation。

用法:
  python3 agent_router.py            # 默认端口 18001 (与 launchd / .sh 一致)
  python3 agent_router.py --port 19000
"""

import os
import sys
import re
import json
import hashlib
import argparse
import asyncio
import logging
import sqlite3
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("agent_router")
# 抑制 httpx 每次请求的 INFO 日志噪音
logging.getLogger("httpx").setLevel(logging.WARNING)

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import uvicorn

# 绕过代理(直连Ollama Cloud)
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

# ============ 配置 ============
OLLAMA_BASE = "https://ollama.com/v1"

# 档位模型池(2026-09-15): 每档 = 有序候选模型列表(优先级从高到低)。
# 依据公开资料按"能力 + token 成本"选型, 各系列取最新版; 智谱无额度已整体剔除。
# 池内降级: 429(配额/限流)/5xx/请求异常 → 自动换下一个; 每模型独立熔断。
MODEL_POOLS = {
    # 简单任务(短prompt/闲聊): 轻量快模, 成本优先
    "cheap": [
        "deepseek-v4.1-flash",        # DeepSeek 最新 flash 代
        "deepseek-v4-flash:0731",     # 上一代 flash, 现役验证款
        "gpt-oss:20b",                # OpenAI 开源轻量
    ],
    # 常规任务(写代码/总结)难度≤0.8: 便宜够用
    "medium": [
        "deepseek-v4.1-flash",
        "kimi-k2.7-code",             # Kimi 代码特化
        "minimax-m2.7",
        "deepseek-v4-flash:0731",
    ],
    # 常规任务偏难(难度>0.8): 强推理但不上旗舰
    "medium_hard": [
        "glm-5.3-flash",
        "kimi-k2.7-code",
        "minimax-m3",
    ],
    # 复杂任务(重构/调试/分析/长输入): 旗舰池
    "smart": [
        "glm-5.3",                    # GLM 满血旗舰(实测延迟最低)
        "kimi-k3",                    # Kimi 最新旗舰
        "deepseek-v4-pro:0813",       # DeepSeek 旗舰
        "qwen3.5:397b",               # Qwen 旗舰
        "minimax-m3",                 # MiniMax 最新
        "glm-5.3-flash",              # 兜底: 强推理 flash
    ],
}

# 所有档位池内的真实模型名集合。用于识别"调用方显式传的是具体模型名"
# (而非档位名/别名) → 精确锁定到该模型, 不再替换为该档位池的首选。
ALL_POOL_MODELS = {m for pool in MODEL_POOLS.values() for m in pool}

# circuit breaker: 每个上游独立熔断, 双门限任一触发即冷却(键=模型名)
#   门限A(低流量快速熔断): 连续失败 >= _UPSTREAM_ALLOWED_FAILS
#   门限B(高流量按比例):   窗口内样本 >= _UPSTREAM_MIN_REQUESTS 且 失败率 > _UPSTREAM_FAILURE_PERCENT
#   门限B 的意义: 连续计数会被一次成功清零, 而"间歇性失败"(成功-失败-成功交替)的高失败率
#   上游不会被门限A抓住 —— 这正是 LiteLLM cooldown_handlers.py 用比例门限要解决的问题。
# 设计参照 LiteLLM: percent=0.5 / min_requests=5 / allowed_fails=3 / cooldown=5s。
_UPSTREAM_WINDOW = {}          # model -> deque[(ts, ok)], 失败率统计窗口
_UPSTREAM_FAILURES = {}        # model -> 连续失败次数
_UPSTREAM_COOLDOWN_UNTIL = {}  # model -> 冷却截止时间戳
_UPSTREAM_WINDOW_SECS = 60     # 失败率统计窗口(秒)
_UPSTREAM_MIN_REQUESTS = 5     # 比例门限要求的最小样本数
_UPSTREAM_FAILURE_PERCENT = 0.5  # 比例门限的失败率阈值
_UPSTREAM_ALLOWED_FAILS = 3    # 连续失败门限
_UPSTREAM_COOLDOWN_SECS = 5    # 冷却时长(原 60s 对已恢复的上游过度惩罚)

def _upstream_available(name: str) -> bool:
    """检查上游是否在冷却期。冷却期满则重置该上游的失败计数与窗口(干净重试)。"""
    until = _UPSTREAM_COOLDOWN_UNTIL.get(name, 0)
    if time.time() < until:
        return False
    if name in _UPSTREAM_COOLDOWN_UNTIL:
        _UPSTREAM_COOLDOWN_UNTIL.pop(name, None)
        _UPSTREAM_FAILURES.pop(name, None)
        _UPSTREAM_WINDOW.pop(name, None)
    return True

def _upstream_record(name: str, ok: bool):
    """记录一次上游调用结果, 按双门限判定是否熔断。"""
    now = time.time()
    window = _UPSTREAM_WINDOW.setdefault(name, deque(maxlen=64))
    window.append((now, ok))
    # 老化: 丢弃窗口外的旧样本
    cutoff = now - _UPSTREAM_WINDOW_SECS
    while window and window[0][0] < cutoff:
        window.popleft()

    if ok:
        _UPSTREAM_FAILURES[name] = 0  # 成功清零连续计数(但保留窗口样本供比例门限)
        return

    consecutive = _UPSTREAM_FAILURES.get(name, 0) + 1
    _UPSTREAM_FAILURES[name] = consecutive
    total = len(window)
    fails = sum(1 for _, o in window if not o)
    rate = (fails / total) if total else 0.0

    reason = None
    if consecutive >= _UPSTREAM_ALLOWED_FAILS:
        reason = f"连续失败 {consecutive} 次"
    elif total >= _UPSTREAM_MIN_REQUESTS and rate > _UPSTREAM_FAILURE_PERCENT:
        reason = f"{_UPSTREAM_WINDOW_SECS}s 内失败率 {rate:.0%}({fails}/{total})"

    if reason:
        _UPSTREAM_COOLDOWN_UNTIL[name] = now + _UPSTREAM_COOLDOWN_SECS
        logger.warning(f"上游 {name} {reason}, 冷却 {_UPSTREAM_COOLDOWN_SECS}s")
        _UPSTREAM_FAILURES.pop(name, None)
        _UPSTREAM_WINDOW.pop(name, None)

def _upstream_success(name: str):
    _upstream_record(name, True)

def _upstream_fail(name: str):
    _upstream_record(name, False)

def _is_retryable_status(status: int) -> bool:
    """可降级的状态码: 429(配额/限流) + 5xx(服务端错误)。4xx 其他(400/401/404)不降级。"""
    return status == 429 or 500 <= status < 600

# ============ ② 投机启动 + 事后升级(escalation/latch) ============
#
# 动机: 旧实现在**请求前**同步调 judge 做 4 分类, 每轮都付一次 judge 往返
# (延迟 + token), 而 agent 循环里绝大多数轮次是工具续写, 判定结果高度重复。
# 现改为「先跑便宜的, 事后评」——参照 Switchyard 的 escalation 思路 + 自研 latch:
#   1) 新会话投机走 cheap(零延迟, 不等任何分类器);
#   2) 响应完成后异步把 (用户请求, 模型回答) 交给 judge 评估质量;
#   3) 连续 ESCALATION_CONFIRMATIONS 次"吃力"判定 → 升级锁存该会话;
#   4) latch 闲置 ESCALATION_LATCH_TTL 秒过期; judge 失败 → fail-open 不改状态。
#
# 为什么用「请求+回答」而非单纯请求: 回答本身是质量证据——小模型答得浅薄/遗漏/
# 跑偏时 judge 能直接看到, 这比只看 prompt 猜难度准得多(Switchyard 的实测结论)。

# 连续多少次"吃力"判定才升级锁存(单次判定可能是噪声 → 需要确认)
ESCALATION_CONFIRMATIONS = 2
# latch 闲置过期时间(秒): 读即续期, 活跃会话保持锁存, 闲置 2h 自动回落投机 cheap
ESCALATION_LATCH_TTL = 7200
# 回答短于此长度不评估(空响应/工具占位轮没有评估价值)
_ESCALATION_MIN_RESPONSE_CHARS = 40
# 后台评估专用信号量: 与用户请求(_semaphore)隔离, 后台记账不挤占用户带宽
_JUDGE_SEM = asyncio.Semaphore(3)
# 持有后台任务引用, 防止 create_task 的协程被 GC 提前回收
_BG_TASKS = set()

# 档位序: 升级目标必须严格高于当前档位才构成升级信号
_TIER_RANK = {"cheap": 0, "medium": 1, "medium_hard": 2, "smart": 3}

# 会话状态(内存态, 进程重启即清空 —— latch 是会话级优化, 无持久化必要)
_LATCH = {}          # conv_key -> (tier, 最后活跃 ts)
_ESCALATION_CONF = {}  # conv_key -> (连续吃力计数, 最近一次目标档位)


def _conv_key_from_text(first_user_text: str) -> str:
    """首条用户消息 → 会话键(sha1[:16])"""
    return hashlib.sha1((first_user_text or "").strip()[:512].encode("utf-8", "ignore")).hexdigest()[:16]


def conversation_key(body: Dict[str, Any]) -> Optional[str]:
    """从请求体推导会话指纹: 首条**有文本的**用户消息前 512 字符的 sha1[:16]。

    agent 每一轮都带全量历史, 首条用户消息在会话内恒定 → 键稳定;
    不同任务(不同首条消息)不会误碰撞。无用户文本(纯工具回传) → None, 不参与 escalation。
    """
    for m in body.get("messages", []):
        if m.get("role") == "user":
            text = _extract_text(m.get("content"))
            if text and text.strip():
                return _conv_key_from_text(text)
    return None


def latch_get(conv_key: Optional[str], now: Optional[float] = None) -> Optional[str]:
    """读取锁存档位。闲置超过 TTL 自动过期; 命中则续期(以"读"当活跃信号)。"""
    if not conv_key:
        return None
    ent = _LATCH.get(conv_key)
    if not ent:
        return None
    tier, ts = ent
    now = now if now is not None else time.time()
    if now - ts > ESCALATION_LATCH_TTL:
        del _LATCH[conv_key]
        return None
    _LATCH[conv_key] = (tier, now)      # 读即续期
    return tier


def apply_escalation_verdict(conv_key: Optional[str], verdict: Optional[str],
                             current_tier: str, now: Optional[float] = None) -> str:
    """消费一次事后评估结果, 返回动作描述(日志/测试可见)。

    verdict ∈ {"ok", "medium_hard", "smart"}; None = judge 失败(fail-open)。
      - ok        → 清零连续计数(小模型胜任的正面证据)
      - 升级目标  → 计数 +1; 达 ESCALATION_CONFIRMATIONS → 写入 _LATCH 并清计数
      - None      → 状态不变(网络抖动不算"吃力", 也不清零已有计数)
    目标档位必须严格高于当前档位才计数(已经 smart 的会话不会被 medium_hard 再升)。
    """
    if not conv_key:
        return "skip(非 auto 会话)"
    if verdict is None:
        return "fail-open(状态不变)"
    if verdict == "ok":
        _ESCALATION_CONF.pop(conv_key, None)
        return "ok(计数清零)"
    target = verdict
    if _TIER_RANK.get(target, 0) <= _TIER_RANK.get(current_tier, 1):
        return f"no-op({target} 不高于 {current_tier})"
    conf, _ = _ESCALATION_CONF.get(conv_key, (0, None))
    conf += 1
    if conf >= ESCALATION_CONFIRMATIONS:
        _LATCH[conv_key] = (target, now if now is not None else time.time())
        _ESCALATION_CONF.pop(conv_key, None)
        return f"LATCH→{target}"
    _ESCALATION_CONF[conv_key] = (conf, target)
    return f"confirm {conf}/{ESCALATION_CONFIRMATIONS}→{target}"

def resolve_pool(tier: str, prompt: str = "", requested: Any = None) -> list:
    """根据档位返回候选模型池。

    - 调用方显式传入**真实模型名**(在池内) → 返回单元素池 [该模型], 精确锁定,
      不使用该档位池的首选, 也不降级到池内其他模型。
    - 其余 → MODEL_POOLS[tier] 完整降级链。tier 为权威判定结果(含 medium_hard),
      此处**不再**用规则分数二次改写 —— 判定权见 ② escalation 区块注释。
    """
    if requested:
        key = str(requested).strip()
        low = key.lower()
        for m in ALL_POOL_MODELS:
            if m.lower() == low:
                return [m]                     # 精确锁定显式模型
    if tier not in MODEL_POOLS:
        logger.warning(f"resolve_pool: 未知档位 {tier!r}, 回退 medium")
        tier = "medium"
    return MODEL_POOLS[tier]

# 显式模型名/别名 → 档位。Hermes/Claude 传入的 model 若命中这里则固定路由到该档位,
# 不再做难度自动路由。传 auto/router/difficulty 或未知名字 → 走难度路由。
MODEL_ALIASES = {
    # 档位名
    "cheap":  "cheap",
    "medium": "medium",
    "smart":  "smart",
    # 短别名
    "flash": "cheap",
    "pro":   "medium",
    "glm":   "smart",
    "glm5":  "smart",
    "glm5.2": "smart",
    "glm5.3": "smart",
    "deepseek": "medium",
    "ds":     "medium",
    # 真实模型名(透传 → 归档到主力档位)
    "deepseek-v4.1-flash":    "cheap",
    "deepseek-v4-flash:0731": "cheap",
    "gpt-oss:20b":            "cheap",
    "kimi-k2.7-code":         "medium",
    "minimax-m2.7":           "medium",
    "glm-5.3-flash":          "smart",
    "glm-5.3":                "smart",
    "kimi-k3":                "smart",
    "deepseek-v4-pro:0813":   "smart",
    "qwen3.5:397b":           "smart",
    "minimax-m3":             "smart",
    # 难度自动路由关键词
    "auto":        None,
    "router":      None,
    "route":       None,
    "difficulty":  None,
}

def resolve_tier(model_name: Any) -> Optional[str]:
    """根据传入模型名解析目标档位。

    - 命中 MODEL_ALIASES 的非 None 值 → 返回固定档位(显式指定模型)
    - 命中 auto/router/... → 返回 None(触发难度自动路由)
    - 未知模型名 → 返回 None(默认走难度自动路由)
    """
    if not model_name:
        return None
    key = str(model_name).strip().lower()
    return MODEL_ALIASES.get(key, None)  # 未知名默认 None → 难度路由

# 上游超时: 连接 15s, 读写 120s; 流式场景 read 放宽到 300s(每 chunk 间)
UPSTREAM_TIMEOUT = httpx.Timeout(120.0, connect=15.0, read=300.0)
# 并发上限: 防止并发请求打爆 Ollama Cloud
MAX_CONCURRENCY = 8

# 共享客户端与信号量(复用连接池, 避免每请求新建)
_client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# ============ Token 读取(带 mtime 缓存) ============
_TOKEN_PATH = os.path.expanduser('~/.hermes/config.yaml')
_token_cache = {"value": None, "mtime": 0}


def get_ollama_token() -> Optional[str]:
    """从 Hermes config 动态读取 token(不硬编码)。

    仅当文件 mtime 变化时重新解析, 其余情况命中缓存, 零 IO。
    """
    try:
        mtime = os.path.getmtime(_TOKEN_PATH)
    except OSError:
        return _token_cache["value"]
    if _token_cache["value"] is not None and mtime == _token_cache["mtime"]:
        return _token_cache["value"]
    value = None
    try:
        with open(_TOKEN_PATH) as f:
            for line in f:
                if line.strip().startswith('api_key:'):
                    val = line.split(':', 1)[1].strip().strip('"\'')
                    if val and len(val) > 20:
                        value = val
                    break
    except Exception:
        pass
    # 关键修复：只在读到有效 token 时更新缓存。
    # 如果读到 None（文件正在被写入/截断/格式变化），保留上一个有效值，
    # 避免竞态条件下持续返回 "no token" 导致 Hermes 重试风暴。
    if value is not None:
        _token_cache["value"] = value
        _token_cache["mtime"] = mtime
    else:
        # 保留旧 mtime，等文件写完后下次调用会重新读取
        pass
    return _token_cache["value"]



# ============ 难度路由 ============
# 判定权已全部移交「事后评估」(见上方 ② 区块): 请求前不再做任何规则打分,
# 规则分数/关键词表在 v1.5.0 移除(历史实现与坑见 docs/routing-defect-postmortem.md)。


# 事后评估 judge: 用 flash(便宜快) 看「请求 + 回答」判断是否需要升级
ESCALATION_JUDGE_MODEL = "deepseek-v4-flash:0731"
ESCALATION_JUDGE_PROMPT = (
    "你是模型路由质量评估器。下面是一个由**预算型小模型**生成的回答。\n"
    "判断该回答是否胜任, 以及该会话是否需要升级到更强的模型池:\n"
    "- 回答已充分、正确、符合请求 → 只输出: ok\n"
    "- 请求中等偏难(多步骤/需推理/重构/调试单个问题), 回答勉强可用但明显粗糙 → 只输出: medium_hard\n"
    "- 请求复杂(系统架构设计/安全审计/跨领域深度分析/长链路多步骤), 且回答明显吃力(浅薄/遗漏/混乱/有错) → 只输出: smart\n\n"
    "只输出 ok / medium_hard / smart 中的一个词, 不要解释。\n\n"
    "用户请求:\n{request}\n\n模型回答:\n{response}\n\n判定:"
)

# 推理型模型会输出 <think> 思考块, 解析前先剥掉
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)  # 剥离推理思考块


_VERDICT_RE = re.compile(r"medium_hard|medium|smart|ok")


def parse_escalation_verdict(content: str) -> Optional[str]:
    """解析 judge 输出为档位字符串; 无法识别返回 None。

    取**最右侧**的判定词: judge 常在结论前带一句简短理由(如 基本可用, 但建议升级 smart),
    真正的判定在末尾。同一位置下长档位优先(正则交替顺序已保证 medium_hard 先于 medium)。
    ok 需为独立词, 避免 book/look 之类误命中。
    """
    c = _THINK_RE.sub("", content or "").strip().lower()
    if not c:
        return None
    best = None
    for mt in _VERDICT_RE.finditer(c):
        w = mt.group(0)
        if w == "ok":                      # 独立词校验(前后不得是 ASCII 字母)
            prev = c[mt.start() - 1] if mt.start() > 0 else " "
            nxt = c[mt.end()] if mt.end() < len(c) else " "
            if (prev.isascii() and prev.isalpha()) or (nxt.isascii() and nxt.isalpha()):
                continue
        key = (mt.start(), len(w))
        if best is None or key >= best[0]:
            best = (key, w)
    return best[1] if best else None


async def judge_escalation(request_text: str, response_text: str, token: str) -> Optional[str]:
    """事后质量评估。返回 ok/medium_hard/smart; 失败/超时/无法解析 → None(fail-open)。"""
    try:
        async with _JUDGE_SEM:
            resp = await _client.post(
                f"{OLLAMA_BASE}/chat/completions",
                json={
                    "model": ESCALATION_JUDGE_MODEL,
                    "messages": [{"role": "user", "content": ESCALATION_JUDGE_PROMPT.format(
                        request=request_text[:2000], response=response_text[:4000])}],
                    "max_tokens": 600,   # 推理型输出可能较长, 过小会返回空
                    "temperature": 0,
                },
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                timeout=30,
            )
        if resp.status_code != 200:
            logger.warning(f"escalation judge 非200 status={resp.status_code}, fail-open")
            return None
        content = (resp.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "")
        verdict = parse_escalation_verdict(content)
        if verdict is None:
            logger.warning(f"escalation judge 输出无法识别({content[:60]!r}), fail-open")
        return verdict
    except Exception as e:
        logger.warning(f"escalation judge 异常 {type(e).__name__}: {str(e)[:150]}, fail-open")
        return None


async def _escalation_judge_worker(request_text: str, response_text: str,
                                   current_tier: str, conv_key: str, token: str):
    """后台任务: 评估 + 消费结果。任何异常都不外抛(fire-and-forget)。"""
    try:
        verdict = await judge_escalation(request_text, response_text, token)
        action = apply_escalation_verdict(conv_key, verdict, current_tier)
        logger.info(f"escalation: conv={conv_key[:8]} tier={current_tier} "
                    f"verdict={verdict} → {action}")
    except Exception as e:
        logger.warning(f"escalation worker 异常 {type(e).__name__}: fail-open")


def _schedule_escalation_judge(request_text: str, response_text: str,
                               current_tier: str, conv_key: Optional[str], token: str):
    """响应完成后调度异步评估(绝不阻塞用户流)。"""
    if not conv_key:
        return                                  # 显式档位/模型名请求不参与 escalation
    if latch_get(conv_key):
        return                                  # 已锁存 → 不再评估(latch = 不回落)
    if len(response_text or "") < _ESCALATION_MIN_RESPONSE_CHARS:
        return                                  # 空响应/占位轮无评估价值
    task = asyncio.create_task(_escalation_judge_worker(
        request_text[:2000], response_text[:4000], current_tier, conv_key, token))
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


def speculative_route(body: Dict[str, Any]) -> tuple:
    """auto 档路由决策: (tier, conv_key)。

    已锁存会话 → 锁存档位; 其余(含新一轮会话) → 投机 cheap(零延迟, 不等分类器)。
    事后由 _schedule_escalation_judge 评估并可能升级锁存。
    """
    conv_key = conversation_key(body)
    latched = latch_get(conv_key)
    if latched:
        logger.info(f"route: conv={(conv_key or '-')[:8]} → {latched} (latched)")
        return latched, conv_key
    logger.info(f"route: conv={(conv_key or '-')[:8]} → cheap (speculative)")
    return "cheap", conv_key


def _extract_text(content: Any) -> Optional[str]:
    """从消息 content(字符串或多模态列表)提取纯文本, 失败返回 None"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text" or "text" in p:
                    parts.append(p.get("text", ""))
        return " ".join(parts)
    return None


def extract_prompt_openai(body: Dict[str, Any]) -> str:
    """从 OpenAI 请求体提取最近最多 2 条用户消息(多轮上下文)"""
    msgs = body.get("messages", [])
    user_parts = []
    for m in reversed(msgs):
        if m.get("role") == "user":
            text = _extract_text(m.get("content", ""))
            if text:
                user_parts.append(text)
            if len(user_parts) >= 2:
                break
    return "\n".join(reversed(user_parts))[:4000]


def extract_prompt_anthropic(body: Dict[str, Any]) -> str:
    """从 Anthropic 请求体提取最近最多 2 条用户消息(多轮上下文)"""
    msgs = body.get("messages", [])
    user_parts = []
    for m in reversed(msgs):
        if m.get("role") == "user":
            text = _extract_text(m.get("content", ""))
            if text:
                user_parts.append(text)
            if len(user_parts) >= 2:
                break
    return "\n".join(reversed(user_parts))[:4000]


def _auth_headers(token: str, anthropic: bool = False) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if anthropic:
        headers["anthropic-version"] = "2023-06-01"
    return headers



# ============ cc-switch 用量统计 ============
CC_SWITCH_DB = Path.home() / ".cc-switch" / "cc-switch.db"
HERMES_APP_TYPE = "hermes"
HERMES_PROVIDER_ID = "hermes-router"


def extract_usage_from_json(data: dict) -> tuple:
    """从 OpenAI 格式响应 JSON 提取 (input_tokens, output_tokens)"""
    try:
        usage = data.get("usage") or {}
        prompt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
        return int(prompt), int(completion)
    except Exception:
        return 0, 0


def write_usage_to_cc_switch(model: str, input_tokens: int, output_tokens: int,
                             status_code: int, latency_ms: int, stream: bool,
                             requested: str = None):
    """把 Hermes 用量写入 cc-switch 统计 DB。
    model     = 实际转发的模型名(落地模型)。
    requested = 调用方原始请求的 model 字段(可能是档位名 auto/flash/smart 或真实模型名)。
                写入 request_model 列, 用于区分「请求了什么档位」——阈值校准必需。
                缺省时回退为落地模型(向后兼容)。
    """
    if not CC_SWITCH_DB.exists():
        return
    now = int(time.time())
    request_id = f"{HERMES_APP_TYPE}-{now}-{uuid.uuid4().hex[:8]}"
    req_label = requested or model
    conn = None
    try:
        conn = sqlite3.connect(str(CC_SWITCH_DB), timeout=5)
        conn.execute(
            """INSERT INTO proxy_request_logs
                (request_id, provider_id, app_type, model, request_model,
                 input_tokens, output_tokens, input_cost_usd, output_cost_usd,
                 total_cost_usd, latency_ms, status_code, is_streaming,
                 cost_multiplier, created_at, data_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (request_id, HERMES_PROVIDER_ID, HERMES_APP_TYPE, model, req_label,
             max(0, input_tokens), max(0, output_tokens),
             "0", "0", "0", max(0, latency_ms), status_code,
             1 if stream else 0, "1.0", now, HERMES_APP_TYPE)
        )
        # 更新日汇总：先查后改（避免依赖 ON CONFLICT 的唯一约束）
        ok = 1 if 200 <= status_code < 400 else 0
        row = conn.execute(
            """SELECT request_count, input_tokens, output_tokens FROM usage_daily_rollups
               WHERE date = date('now') AND app_type = ? AND provider_id = ?
                 AND model = ? AND request_model = ? AND pricing_model = ''""",
            (HERMES_APP_TYPE, HERMES_PROVIDER_ID, model, req_label)
        ).fetchone()
        if row:
            conn.execute(
                """UPDATE usage_daily_rollups SET
                     request_count = request_count + 1,
                     success_count = success_count + ?,
                     input_tokens = input_tokens + ?,
                     output_tokens = output_tokens + ?,
                     avg_latency_ms = (? + avg_latency_ms * request_count) / (request_count + 1)
                   WHERE date = date('now') AND app_type = ? AND provider_id = ?
                     AND model = ? AND request_model = ? AND pricing_model = ''""",
                (ok, max(0, input_tokens), max(0, output_tokens), max(0, latency_ms),
                 HERMES_APP_TYPE, HERMES_PROVIDER_ID, model, req_label)
            )
        else:
            conn.execute(
                """INSERT INTO usage_daily_rollups
                    (date, app_type, provider_id, model, request_model, pricing_model,
                     request_count, success_count, input_tokens, output_tokens,
                     cache_read_tokens, cache_creation_tokens,
                     total_cost_usd, avg_latency_ms)
                 VALUES (date('now'), ?, ?, ?, ?, '', 1, ?, ?, ?, 0, 0, '0', ?)""",
                (HERMES_APP_TYPE, HERMES_PROVIDER_ID, model, req_label,
                 ok, max(0, input_tokens), max(0, output_tokens), max(0, latency_ms))
            )
        conn.commit()
        logger.debug(f"usage recorded: model={model} in={input_tokens} out={output_tokens} stream={stream}")
    except Exception as e:
        logger.warning(f"cc-switch usage write failed: {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


async def stream_with_usage(resp, model: str, start_time: float, prompt_len: int,
                            requested: str = None, prompt: str = "",
                            tier: str = None, conv_key: str = None, token: str = None):
    """流式转发包装器：透传 SSE chunk，流结束后解析 usage 并写入 cc-switch。

    requested = 调用方原始请求的 model 字段, 透传给 write_usage 记入 request_model。
    同时累积回答文本, 流结束后调度事后评估(escalation) —— 全程不阻塞用户流。
    """
    chunks = []
    try:
        async for chunk in resp.aiter_bytes():
            chunks.append(chunk)
            yield chunk
    finally:
        # 流结束，解析 SSE: 提取 usage + 累积回答文本
        input_tokens = output_tokens = 0
        answer_parts = []
        usage_found = False
        try:
            full = b"".join(chunks).decode("utf-8", errors="ignore")
            for line in full.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        continue
                    data = json.loads(payload)
                    if data.get("usage") and not usage_found:
                        input_tokens, output_tokens = extract_usage_from_json(data)
                        usage_found = True
                    piece = _sse_delta_text(data)
                    if piece:
                        answer_parts.append(piece)
        except Exception:
            pass
        if not input_tokens:
            input_tokens = max(1, prompt_len // 4)  # 兜底估算
        latency_ms = int((time.time() - start_time) * 1000)
        write_usage_to_cc_switch(model, input_tokens, output_tokens,
                                 resp.status_code, latency_ms, stream=True,
                                 requested=requested)
        # 事后评估(异步 fire-and-forget, 不影响已完成的用户响应)
        if tier and token:
            _schedule_escalation_judge(prompt, "".join(answer_parts), tier, conv_key, token)


# ============ 转发逻辑 ============
def _response_text(data: dict) -> str:
    """从非流式响应 JSON 提取回答文本(兼容 OpenAI / Anthropic 两种格式)"""
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = _extract_text(msg.get("content"))
    if text:
        return text
    return _extract_text(data.get("content")) or ""


def _sse_delta_text(data: dict) -> str:
    """从单个 SSE 事件 JSON 提取增量文本(兼容两种协议)"""
    ch = (data.get("choices") or [{}])[0]
    delta = ch.get("delta") or {}
    t = delta.get("content")
    if isinstance(t, str) and t:
        return t
    d2 = data.get("delta")          # Anthropic: content_block_delta → delta.text
    if isinstance(d2, dict):
        t2 = d2.get("text")
        if isinstance(t2, str):
            return t2
    return ""


async def _resolve_and_route(body: Dict[str, Any], token: str, extract_fn) -> tuple:
    """统一决定目标档位, 返回 (tier, conv_key)。

    - 显式别名/档位/真实模型名 → 固定档位, conv_key=None(不参与 escalation)。
    - auto/未知模型名 → 投机+锁存路由(见 speculative_route)。
    """
    requested = body.get("model")
    tier = resolve_tier(requested)
    if tier is not None:
        return tier, None                # 显式指定模型
    return speculative_route(body)


async def _forward_with_failover(body: Dict[str, Any], tier: str, prompt: str,
                                 token: str, anthropic: bool = False,
                                 conv_key: str = None) -> Response:
    """按档位模型池逐个尝试, 429/5xx/请求异常自动降级到池内下一个模型。

    - 非流式: 完整降级循环。
    - 流式: 请求阶段失败可降级; 拿到 200 响应头后锁定(不能中途换模型)。
    - circuit breaker: 某模型连续失败 >= 阈值进入冷却期, 期间跳过。
    - conv_key 非空 = auto 会话, 响应完成后调度事后评估(escalation)。
    """
    pool = resolve_pool(tier, prompt, requested=body.get("model"))
    stream = body.get("stream", False)
    start_time = time.time()
    last_err = None

    # 请求标签: 显式档位/别名/真实模型名 → 原样记录;
    # 难度自动路由(auto/未知名) → 记 "auto:<实际决策档位>", 便于按档位校准阈值。
    req_raw = body.get("model")
    if req_raw is None or resolve_tier(req_raw) is None:
        req_label = f"auto:{tier}"
    else:
        req_label = req_raw

    for model in pool:
        if not _upstream_available(model):
            logger.info(f"模型 {model} 冷却中, 跳过")
            continue
        out_body = dict(body)
        out_body["model"] = model
        headers = _auth_headers(token, anthropic=anthropic)
        path = "/messages" if anthropic else "/chat/completions"
        try:
            async with _semaphore:
                if stream:
                    out_body["stream_options"] = {"include_usage": True}
                    req = _client.build_request("POST", f"{OLLAMA_BASE}{path}",
                                                json=out_body, headers=headers)
                    resp = await _client.send(req, stream=True)
                else:
                    resp = await _client.post(f"{OLLAMA_BASE}{path}",
                                              json=out_body, headers=headers)
        except Exception as e:
            _upstream_fail(model)
            last_err = e
            logger.warning(f"模型 {model} 请求异常 {type(e).__name__}: {str(e)[:150]}, 降级到下一个")
            continue

        # 非流式: 可重试状态码则降级
        if not stream:
            if _is_retryable_status(resp.status_code):
                _upstream_fail(model)
                logger.warning(f"模型 {model} status={resp.status_code}, 降级到池内下一个")
                await resp.aclose()
                continue
            _upstream_success(model)
            latency_ms = int((time.time() - start_time) * 1000)
            try:
                data = resp.json()
                input_tokens, output_tokens = extract_usage_from_json(data)
                write_usage_to_cc_switch(model, input_tokens, output_tokens,
                                         resp.status_code, latency_ms, stream=False,
                                         requested=req_label)
                answer_text = _response_text(data)
            except Exception:
                answer_text = ""
            logger.info(f"openai[{stream}] upstream={resp.status_code} model={model} "
                        f"tier={tier} prompt_len={len(prompt)}")
            # 事后评估(异步 fire-and-forget, 用户已拿到响应)
            _schedule_escalation_judge(prompt, answer_text, tier, conv_key, token)
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type="application/json")

        # 流式: 拿到响应头后锁定。仅 429/5xx 响应头可降级。
        if _is_retryable_status(resp.status_code):
            _upstream_fail(model)
            await resp.aclose()
            logger.warning(f"模型 {model} stream status={resp.status_code}, 降级到池内下一个")
            continue
        _upstream_success(model)
        logger.info(f"openai[{stream}] upstream={resp.status_code} model={model} "
                    f"tier={tier} prompt_len={len(prompt)}")
        return StreamingResponse(
            stream_with_usage(resp, model, start_time, len(prompt),
                              requested=req_label, prompt=prompt,
                              tier=tier, conv_key=conv_key, token=token),
            status_code=resp.status_code,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
            background=BackgroundTask(resp.aclose),
        )

    # 池内全部模型失败
    if last_err:
        raise last_err
    return Response(content=json.dumps({"error": "all models in pool failed",
                                        "tier": tier}),
                    status_code=502, media_type="application/json")


async def forward_openai(body: Dict[str, Any], token: str) -> Response:
    """转发 OpenAI 协议请求, 档位模型池内自动降级"""
    tier, conv_key = await _resolve_and_route(body, token, extract_prompt_openai)
    prompt = extract_prompt_openai(body)
    return await _forward_with_failover(body, tier, prompt, token, anthropic=False,
                                        conv_key=conv_key)


async def forward_anthropic(body: Dict[str, Any], token: str) -> Response:
    """转发 Anthropic 协议请求, 档位模型池内自动降级(全走 ollama)"""
    tier, conv_key = await _resolve_and_route(body, token, extract_prompt_anthropic)
    prompt = extract_prompt_anthropic(body)
    return await _forward_with_failover(body, tier, prompt, token, anthropic=True,
                                        conv_key=conv_key)


# ============ FastAPI 应用 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _client.aclose()  # 优雅关闭: 释放共享连接池


app = FastAPI(title="Agent Model Router", version="1.5.0", lifespan=lifespan)


@app.get("/health")
async def health():
    now = time.time()
    live_latch = {k: v for k, v in _LATCH.items()
                  if now - v[1] <= ESCALATION_LATCH_TTL}
    return {"status": "ok", "version": app.version,
            "tiers": {k: v[0] + f" (+{len(v)-1} fallback)" if len(v) > 1 else v[0]
                      for k, v in MODEL_POOLS.items() if k != "medium_hard"},
            "pools": MODEL_POOLS,
            "escalation": {
                "latched": len(live_latch),
                "pending": len(_ESCALATION_CONF),
                "confirmations": ESCALATION_CONFIRMATIONS,
                "ttl_secs": ESCALATION_LATCH_TTL,
            }}


@app.get("/internal/escalation")
async def escalation_state():
    """运维/测试用: 查看 latch 与待确认计数(会话键已缩短便于阅读)"""
    now = time.time()
    return {
        "latches": [{"conv": k, "tier": t, "idle_secs": int(now - ts)}
                    for k, (t, ts) in _LATCH.items()
                    if now - ts <= ESCALATION_LATCH_TTL],
        "pending": [{"conv": k, "count": c, "target": t}
                    for k, (c, t) in _ESCALATION_CONF.items()],
    }


@app.post("/internal/escalation")
async def escalation_control(request: Request):
    """运维/测试用: 注入或清除某会话的 latch(用首条用户消息原文定位会话)。

    {"text": "<首条用户消息>", "tier": "smart"}  → 注入 latch
    {"text": "...", "clear": true}              → 清除 latch 与待确认计数
    """
    body = await request.json()
    text = body.get("text") or ""
    conv = _conv_key_from_text(text)
    if body.get("clear"):
        _LATCH.pop(conv, None)
        _ESCALATION_CONF.pop(conv, None)
        return {"conv": conv, "cleared": True}
    tier = body.get("tier")
    if tier not in MODEL_POOLS:
        return Response(content=json.dumps({"error": f"tier 无效: {tier}"}),
                        status_code=400, media_type="application/json")
    _LATCH[conv] = (tier, time.time())
    return {"conv": conv, "tier": tier}


@app.get("/v1/models")
async def list_models():
    """返回直观档位名(flash/pro/smart)。Hermes 的 model picker 会把这些 id
    作为 model 传给 /chat/completions, 因此 id 必须命中 MODEL_ALIASES。
    """
    tiers = [
        ("auto",  None),
        ("flash", "cheap"),
        ("pro",   "medium"),
        ("smart", "smart"),
    ]
    data = []
    for alias, real in tiers:
        if real:
            pool = MODEL_POOLS[real]
            desc = f"{alias} 池: {' → '.join(pool)}"
        else:
            desc = ("难度自动路由(推荐): 新会话投机走 cheap(零延迟不等分类器), 响应后"
                    "异步评估(请求+回答); 连续 2 次判定吃力 → 升级锁存该会话(2h 内保持)")
        data.append({"id": alias, "object": "model", "owned_by": "agent-router",
                     "description": desc})
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    """OpenAI 协议 (Hermes 用)"""
    token = get_ollama_token()
    if not token:
        return Response(content=json.dumps({"error": "no token"}), status_code=500,
                        media_type="application/json")
    body = await request.json()
    logger.info(f"openai收到: path={request.url.path} size={len(json.dumps(body))} stream={body.get('stream')} "
                f"tools={len(body.get('tools', []))} model={body.get('model')}")
    try:
        return await forward_openai(body, token)
    except Exception as e:
        logger.error(f"openai端点异常 {type(e).__name__}: {str(e)[:300]}")
        raise


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic 协议 (Claude 用)"""
    token = get_ollama_token()
    if not token:
        return Response(content=json.dumps({"error": "no token"}), status_code=500,
                        media_type="application/json")
    body = await request.json()
    logger.info(f"anthropic收到: path={request.url.path} size={len(json.dumps(body))} stream={body.get('stream')} "
                f"tools={len(body.get('tools', []))} model={body.get('model')}")
    try:
        return await forward_anthropic(body, token)
    except Exception as e:
        logger.error(f"anthropic端点异常 {type(e).__name__}: {str(e)[:300]}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Agent Model Router")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    token = get_ollama_token()
    if not token:
        print("❌ 无法从 ~/.hermes/config.yaml 读取 Ollama token", file=sys.stderr)
        sys.exit(1)

    print(f"🚀 Agent Model Router v{app.version} 启动")
    print(f"   OpenAI 协议:  http://{args.host}:{args.port}/v1/chat/completions (Hermes用)")
    print(f"   Anthropic协议: http://{args.host}:{args.port}/v1/messages (Claude用)")
    print(f"   后端: {OLLAMA_BASE}")
    for tier, pool in MODEL_POOLS.items():
        if tier != "medium_hard":
            print(f"   {tier:6s}池: {' → '.join(pool)}")
    print(f"   并发上限: {MAX_CONCURRENCY}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
