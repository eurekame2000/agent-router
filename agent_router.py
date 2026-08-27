#!/usr/bin/env python3
"""
本地 Agent 模型路由网关 — 按任务难度路由到 Ollama Cloud 不同模型
同时暴露 OpenAI 和 Anthropic 两套协议，兼容 Hermes 和 Claude Code。

架构:
  Hermes (OpenAI协议)  ─┐
                        ├─→ 本网关 (难度路由) ─→ Ollama Cloud
  Claude (Anthropic)  ─┘

难度路由:
  cheap  (deepseek-v4-flash:0731)  — 简单任务(短prompt/闲聊)
  medium (kimi-k2.7-code)       — 常规任务(写代码/总结)
  smart  (glm-5.3-flash)          — 复杂任务(重构/调试/分析/长输入)

用法:
  python3 agent_router.py            # 默认端口 18001 (与 launchd / .sh 一致)
  python3 agent_router.py --port 19000
"""

import os
import sys
import json
import argparse
import asyncio
import logging
import sqlite3
import time
import uuid
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

# 模型档位 → Ollama Cloud 实际模型名(从 /api/tags 查证)
# 用户指定排序(从易到难): flash → Kimi K2.7-code → glm5.2
MODEL_TIERS = {
    "cheap":  "deepseek-v4-flash:0731",   # 简单任务 (存在)
    "medium": "kimi-k2.7-code",           # 常规任务 (存在, Kimi K2.7 代码版)
    "smart":  "glm-5.3-flash",            # 复杂任务 (存在)
}

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
    "kimi":  "medium",
    "kimi2.7": "medium",
    "kimi-k2.7-code": "medium",
    # 真实模型名(直接透传)
    "deepseek-v4-flash:0731": "cheap",
    "kimi-k2.7-code":         "medium",
    "glm-5.3-flash":          "smart",
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
# 难度分数区间(用户指定): 0-0.8 易(flash) / 0.8-0.95 中(Pro/judge) / 0.95-1 难(glm5.2)
# 分数越高 = 任务越难

# 简单信号(闲聊/问候/短指令) → 降低难度
EASY_KEYWORDS = [
    "hi", "hello", "hey", "你好", "嗨", "谢谢", "thanks", "ok", "好的",
    "bye", "再见", "who are you", "你是谁", "help", "帮助",
]

# 中等信号 → 中等加分
MEDIUM_KEYWORDS = [
    "write code", "implement", "review", "summarize", "create", "build",
    "explain", "fix", "test", "写代码", "实现", "总结", "创建", "修复",
    "解释", "测试", "函数", "function", "代码", "脚本", "程序", "算法",
    "python", "javascript", "solidity", "sql", "java", "golang", "rust",
    "怎么", "如何", "怎样", "为什么", "是什么", "区别", "对比",
]

# 复杂信号 → 高加分
SMART_KEYWORDS = [
    "refactor", "debug", "analyze", "explain architecture", "optimize",
    "unit test", "deploy", "security audit", "vulnerability", "reentrancy",
    "gas optimization", "architecture", "design pattern", "code review",
    "重构", "调试", "分析", "优化", "审计", "漏洞", "架构", "设计模式",
    "安全", "重入", "部署",
]


def difficulty_score(prompt: str) -> float:
    """计算 prompt 难度分数(0-1), 越高越难"""
    if not prompt:
        return 0.5
    pl = prompt.lower()
    length = len(prompt)

    # 简单信号(短闲聊) → 直接压到低分
    if any(k in pl for k in EASY_KEYWORDS) and length < 50:
        return 0.2

    # 基础分 0.5, 关键词命中往上加
    score = 0.5

    # 中等信号: 每个 +0.15, 上限 +0.4
    medium_hits = sum(1 for k in MEDIUM_KEYWORDS if k in pl)
    score += min(medium_hits, 3) * 0.15

    # 复杂信号: 每个 +0.2, 上限 +0.5
    smart_hits = sum(1 for k in SMART_KEYWORDS if k in pl)
    score += min(smart_hits, 3) * 0.2

    # 长度贡献: 超过200字符额外加分, 上限 +0.1
    if length > 200:
        score += 0.1

    return min(score, 1.0)


# 模糊地带: 规则分数落在此区间时, 用 LLM 二次判断
# 规则拿不准的边界(0.8-0.95), 避免误判
LLM_JUDGE_LOW = 0.8
LLM_JUDGE_HIGH = 0.95  # 贵模型(glm5.2)触发阈值

# LLM judge 用 flash(便宜快), 判断 prompt 难度
LLM_JUDGE_MODEL = "deepseek-v4-flash:0731"
LLM_JUDGE_PROMPT = (
    "你是任务难度分类器。判断下面用户请求的复杂度, 只输出一个词: "
    "cheap(简单闲聊/一句话/无需推理) / medium(常规任务/写代码/总结/解释) / "
    "smart(复杂任务/重构/调试/架构/安全审计/长输入多步骤)。\n\n"
    "用户请求:\n{prompt}\n\n难度:"
)


async def llm_judge(prompt: str, token: str) -> str:
    """用 flash 模型二次判断难度, 返回档位(cheap/medium/smart)"""
    try:
        async with _semaphore:
            resp = await _client.post(
                f"{OLLAMA_BASE}/chat/completions",
                json={
                    "model": LLM_JUDGE_MODEL,
                    "messages": [
                        {"role": "user", "content": LLM_JUDGE_PROMPT.format(prompt=prompt[:2000])}
                    ],
                    "max_tokens": 200,
                    "temperature": 0,
                },
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                timeout=30,  # judge 单独收紧超时, 避免拖慢主请求
            )
        if resp.status_code != 200:
            logger.warning(f"llm_judge 非200 status={resp.status_code}, 回退 medium")
            return "medium"  # 失败时保守回退到 medium
        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip().lower()
        for tier in ("smart", "medium", "cheap"):
            if tier in content:
                return tier
        return "medium"
    except Exception as e:
        logger.error(f"llm_judge 异常 {type(e).__name__}: {str(e)[:200]}, 回退 medium")
        return "medium"  # 异常时保守回退


async def route_by_difficulty(prompt: str, token: str) -> str:
    """混合路由: 简单直接走 flash, 其余(含高难度)都用 LLM 二次判断"""
    score = difficulty_score(prompt)

    # 明确简单: 直接规则路由, 零成本
    if score < LLM_JUDGE_LOW:
        return "cheap"    # 易 → flash

    # 中等和高难度都走 LLM judge 确认(避免规则误判浪费贵模型)
    return await llm_judge(prompt, token)


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
                             status_code: int, latency_ms: int, stream: bool):
    """把 Hermes 用量写入 cc-switch 统计 DB。model 必须是实际转发的模型名。"""
    if not CC_SWITCH_DB.exists():
        return
    now = int(time.time())
    request_id = f"{HERMES_APP_TYPE}-{now}-{uuid.uuid4().hex[:8]}"
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
            (request_id, HERMES_PROVIDER_ID, HERMES_APP_TYPE, model, model,
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
            (HERMES_APP_TYPE, HERMES_PROVIDER_ID, model, model)
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
                 HERMES_APP_TYPE, HERMES_PROVIDER_ID, model, model)
            )
        else:
            conn.execute(
                """INSERT INTO usage_daily_rollups
                    (date, app_type, provider_id, model, request_model, pricing_model,
                     request_count, success_count, input_tokens, output_tokens,
                     cache_read_tokens, cache_creation_tokens,
                     total_cost_usd, avg_latency_ms)
                 VALUES (date('now'), ?, ?, ?, ?, '', 1, ?, ?, ?, 0, 0, '0', ?)""",
                (HERMES_APP_TYPE, HERMES_PROVIDER_ID, model, model,
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


async def stream_with_usage(resp, model: str, start_time: float, prompt_len: int):
    """流式转发包装器：透传 SSE chunk，流结束后解析 usage 并写入 cc-switch。"""
    chunks = []
    try:
        async for chunk in resp.aiter_bytes():
            chunks.append(chunk)
            yield chunk
    finally:
        # 流结束，尝试从 SSE 中解析 usage
        input_tokens = output_tokens = 0
        try:
            full = b"".join(chunks).decode("utf-8", errors="ignore")
            for line in full.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        continue
                    data = json.loads(payload)
                    if data.get("usage"):
                        input_tokens, output_tokens = extract_usage_from_json(data)
                        break
        except Exception:
            pass
        if not input_tokens:
            input_tokens = max(1, prompt_len // 4)  # 兜底估算
        latency_ms = int((time.time() - start_time) * 1000)
        write_usage_to_cc_switch(model, input_tokens, output_tokens,
                                 resp.status_code, latency_ms, stream=True)


# ============ 转发逻辑 ============
async def _resolve_and_route(body: Dict[str, Any], token: str, extract_fn) -> str:
    """统一决定目标档位。

    优先: 传入 model 命中显式别名 → 固定档位(指定模型)。
    否则: 走难度自动路由(auto / 未知模型名)。
    """
    requested = body.get("model")
    tier = resolve_tier(requested)
    if tier is not None:
        return tier                      # 显式指定模型
    prompt = extract_fn(body)
    return await route_by_difficulty(prompt, token)   # 难度自动路由


async def forward_openai(body: Dict[str, Any], token: str) -> Response:
    """转发 OpenAI 协议请求到 Ollama Cloud"""
    tier = await _resolve_and_route(body, token, extract_prompt_openai)
    model = MODEL_TIERS[tier]

    # 替换模型名
    out_body = dict(body)
    out_body["model"] = model

    headers = _auth_headers(token)
    stream = out_body.get("stream", False)
    prompt = extract_prompt_openai(body)
    start_time = time.time()

    try:
        async with _semaphore:
            if stream:
                # 请求 Ollama Cloud 在流末尾返回带 usage 的 chunk
                out_body["stream_options"] = {"include_usage": True}
                req = _client.build_request("POST", f"{OLLAMA_BASE}/chat/completions",
                                            json=out_body, headers=headers)
                resp = await _client.send(req, stream=True)
                logger.info(f"openai[{stream}] upstream={resp.status_code} model={model} "
                            f"prompt_len={len(prompt)} tier={tier}")
                return StreamingResponse(
                    stream_with_usage(resp, model, start_time, len(prompt)),
                    status_code=resp.status_code,
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"},
                    background=BackgroundTask(resp.aclose),
                )
            else:
                resp = await _client.post(f"{OLLAMA_BASE}/chat/completions",
                                          json=out_body, headers=headers)
                logger.info(f"openai[{stream}] upstream={resp.status_code} model={model} "
                            f"prompt_len={len(prompt)} tier={tier}")
                latency_ms = int((time.time() - start_time) * 1000)
                try:
                    data = resp.json()
                    input_tokens, output_tokens = extract_usage_from_json(data)
                    write_usage_to_cc_switch(model, input_tokens, output_tokens,
                                             resp.status_code, latency_ms, stream=False)
                except Exception as e:
                    logger.warning(f"usage extract failed: {e}")
                return Response(content=resp.content, status_code=resp.status_code,
                                media_type="application/json")
    except Exception as e:
        logger.error(f"openai[{stream}] 转发异常 {type(e).__name__}: {str(e)[:200]}")
        raise


async def forward_anthropic(body: Dict[str, Any], token: str) -> Response:
    """转发 Anthropic 协议请求到 Ollama Cloud

    注意: Ollama Cloud 仅提供 OpenAI 兼容端点(/v1/chat/completions),
    Anthropic 原生 /v1/messages 仅在兼容时可用; 此处原样透传。
    """
    tier = await _resolve_and_route(body, token, extract_prompt_anthropic)
    model = MODEL_TIERS[tier]

    out_body = dict(body)
    out_body["model"] = model

    headers = _auth_headers(token, anthropic=True)
    stream = out_body.get("stream", False)
    prompt = extract_prompt_anthropic(body)

    try:
        async with _semaphore:
            if stream:
                req = _client.build_request("POST", f"{OLLAMA_BASE}/messages",
                                            json=out_body, headers=headers)
                resp = await _client.send(req, stream=True)
                logger.info(f"anthropic[{stream}] upstream={resp.status_code} model={model} "
                            f"prompt_len={len(prompt)} tier={tier}")
                return StreamingResponse(
                    resp.aiter_bytes(),
                    status_code=resp.status_code,
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"},
                    background=BackgroundTask(resp.aclose),
                )
            else:
                resp = await _client.post(f"{OLLAMA_BASE}/messages",
                                          json=out_body, headers=headers)
                logger.info(f"anthropic[{stream}] upstream={resp.status_code} model={model} "
                            f"prompt_len={len(prompt)} tier={tier}")
                return Response(content=resp.content, status_code=resp.status_code,
                                media_type="application/json")
    except Exception as e:
        logger.error(f"anthropic[{stream}] 转发异常 {type(e).__name__}: {str(e)[:200]}")
        raise


# ============ FastAPI 应用 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _client.aclose()  # 优雅关闭: 释放共享连接池


app = FastAPI(title="Agent Model Router", version="1.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "models": MODEL_TIERS}


@app.get("/v1/models")
async def list_models():
    """返回直观档位名(flash/pro/smart)。Hermes 的 model picker 会把这些 id
    作为 model 传给 /chat/completions, 因此 id 必须命中 MODEL_ALIASES。
    """
    tiers = [
        ("auto",  None, None),  # 难度自动路由
        ("flash", "cheap",  "deepseek-v4-flash:0731"),
        ("pro",   "medium", "kimi-k2.7-code"),
        ("smart", "smart",  "glm-5.3-flash"),
    ]
    return {"object": "list", "data": [
        {"id": alias, "object": "model", "owned_by": "agent-router",
         "model": MODEL_TIERS[real] if real else "auto",
         "description": "难度自动路由(推荐)" if alias == "auto"
                        else f"{alias} → {MODEL_TIERS[real]}"}
        for alias, real, _ in tiers
    ]}


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

    print(f"🚀 Agent Model Router v1.1.0 启动")
    print(f"   OpenAI 协议:  http://{args.host}:{args.port}/v1/chat/completions (Hermes用)")
    print(f"   Anthropic协议: http://{args.host}:{args.port}/v1/messages (Claude用)")
    print(f"   模型档位: {MODEL_TIERS}")
    print(f"   后端: {OLLAMA_BASE}")
    print(f"   并发上限: {MAX_CONCURRENCY}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
