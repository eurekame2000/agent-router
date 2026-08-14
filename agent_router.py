#!/usr/bin/env python3
"""
本地 Agent 模型路由网关 — 按任务难度路由到 Ollama Cloud 不同模型
同时暴露 OpenAI 和 Anthropic 两套协议，兼容 Hermes 和 Claude Code。

架构:
  Hermes (OpenAI协议)  ─┐
                        ├─→ 本网关 (难度路由) ─→ Ollama Cloud
  Claude (Anthropic)  ─┘

难度路由:
  cheap  (gemma4)              — 简单任务(短prompt/闲聊)
  medium (deepseek-v4-flash)   — 常规任务(写代码/总结)
  smart  (deepseek-v4-pro)     — 复杂任务(重构/调试/分析/长输入)

用法:
  python3 agent_router.py            # 默认端口 18000
  python3 agent_router.py --port 19000
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("agent_router")

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
# 用户指定排序(从易到难): flash → Pro → glm5.2
MODEL_TIERS = {
    "cheap":  "deepseek-v4-flash:0731",   # 简单任务 (存在)
    "medium": "deepseek-v4-pro:preview",  # 常规任务 (存在, 注意是:preview)
    "smart":  "glm-5.2",                  # 复杂任务 (存在)
}

# 难度路由规则(关键词 → 档位)
SMART_KEYWORDS = [
    "refactor", "debug", "analyze", "explain architecture", "optimize",
    "unit test", "deploy", "security audit", "vulnerability", "reentrancy",
    "gas optimization", "architecture", "design pattern", "code review",
    "重构", "调试", "分析", "优化", "审计", "漏洞", "架构", "设计模式",
    "安全", "重入", "部署",
]
MEDIUM_KEYWORDS = [
    "write code", "implement", "review", "summarize", "create", "build",
    "explain", "fix", "test", "写代码", "实现", "总结", "创建", "修复",
    "解释", "测试",
]

def get_ollama_token() -> Optional[str]:
    """从 Hermes config 动态读取 token(不硬编码)"""
    cfg_path = os.path.expanduser('~/.hermes/config.yaml')
    try:
        with open(cfg_path) as f:
            for line in f:
                if line.strip().startswith('api_key:'):
                    val = line.split(':', 1)[1].strip().strip('"\'')
                    if val and len(val) > 20:
                        return val
    except Exception:
        pass
    return None

# ============ 难度路由 ============
# 难度分数区间(用户指定): 0-0.8 易(flash) / 0.8-0.9 中(Pro) / 0.9-1 难(glm5.2)
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
# 规则拿不准的边界(0.7-0.9), 避免误判
LLM_JUDGE_LOW = 0.7
LLM_JUDGE_HIGH = 0.9

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
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
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
            )
            if resp.status_code != 200:
                return "medium"  # 失败时保守回退到 medium
            data = resp.json()
            content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip().lower()
            for tier in ("smart", "medium", "cheap"):
                if tier in content:
                    return tier
            return "medium"
    except Exception:
        return "medium"  # 异常时保守回退


async def route_by_difficulty(prompt: str, token: str) -> str:
    """混合路由: 规则粗筛, 模糊地带用 LLM 二次判断"""
    score = difficulty_score(prompt)

    # 明确区间: 直接规则路由, 零成本
    if score >= LLM_JUDGE_HIGH:
        return "smart"    # 难 → glm5.2
    if score < LLM_JUDGE_LOW:
        return "cheap"    # 易 → flash

    # 模糊地带(0.7-0.9): 用 flash 二次判断
    return await llm_judge(prompt, token)

def extract_prompt_openai(body: Dict[str, Any]) -> str:
    """从 OpenAI 请求体提取用户消息"""
    msgs = body.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # 多模态
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return " ".join(parts)
    return ""

def extract_prompt_anthropic(body: Dict[str, Any]) -> str:
    """从 Anthropic 请求体提取用户消息"""
    msgs = body.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                return " ".join(parts)
    return ""

# ============ 转发逻辑 ============
async def forward_openai(body: Dict[str, Any], token: str) -> Response:
    """转发 OpenAI 协议请求到 Ollama Cloud"""
    prompt = extract_prompt_openai(body)
    tier = await route_by_difficulty(prompt, token)
    model = MODEL_TIERS[tier]

    # 替换模型名
    out_body = dict(body)
    out_body["model"] = model

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    stream = out_body.get("stream", False)

    client = httpx.AsyncClient(timeout=120)
    try:
        if stream:
            req = client.build_request("POST", f"{OLLAMA_BASE}/chat/completions",
                                       json=out_body, headers=headers)
            resp = await client.send(req, stream=True)
            logger.info(f"openai[{stream}] upstream={resp.status_code} model={model} prompt_len={len(prompt)}")
            return StreamingResponse(
                resp.aiter_bytes(),
                status_code=resp.status_code,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
                background=BackgroundTask(client.aclose),
            )
        else:
            resp = await client.post(f"{OLLAMA_BASE}/chat/completions",
                                     json=out_body, headers=headers)
            await client.aclose()
            logger.info(f"openai[{stream}] upstream={resp.status_code} model={model} prompt_len={len(prompt)}")
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type="application/json")
    except Exception as e:
        logger.error(f"openai[{stream}] 转发异常 {type(e).__name__}: {str(e)[:200]}")
        await client.aclose()
        raise

async def forward_anthropic(body: Dict[str, Any], token: str) -> Response:
    """转发 Anthropic 协议请求到 Ollama Cloud"""
    prompt = extract_prompt_anthropic(body)
    tier = await route_by_difficulty(prompt, token)
    model = MODEL_TIERS[tier]

    out_body = dict(body)
    out_body["model"] = model

    headers = {
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    stream = out_body.get("stream", False)

    client = httpx.AsyncClient(timeout=120)
    try:
        if stream:
            req = client.build_request("POST", f"{OLLAMA_BASE}/messages",
                                       json=out_body, headers=headers)
            resp = await client.send(req, stream=True)
            return StreamingResponse(
                resp.aiter_bytes(),
                status_code=resp.status_code,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
                background=BackgroundTask(client.aclose),
            )
        else:
            resp = await client.post(f"{OLLAMA_BASE}/messages",
                                     json=out_body, headers=headers)
            await client.aclose()
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type="application/json")
    except Exception:
        await client.aclose()
        raise

# ============ FastAPI 应用 ============
app = FastAPI(title="Agent Model Router", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "ok", "models": MODEL_TIERS}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": tier, "object": "model"} for tier in MODEL_TIERS
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
    return await forward_anthropic(body, token)

def main():
    parser = argparse.ArgumentParser(description="Agent Model Router")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    token = get_ollama_token()
    if not token:
        print("❌ 无法从 ~/.hermes/config.yaml 读取 Ollama token", file=sys.stderr)
        sys.exit(1)

    print(f"🚀 Agent Model Router 启动")
    print(f"   OpenAI 协议:  http://{args.host}:{args.port}/v1/chat/completions (Hermes用)")
    print(f"   Anthropic协议: http://{args.host}:{args.port}/v1/messages (Claude用)")
    print(f"   模型档位: {MODEL_TIERS}")
    print(f"   后端: {OLLAMA_BASE}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
