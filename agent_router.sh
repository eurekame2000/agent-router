#!/bin/bash
# Agent Model Router 启动/停止脚本
# 用法: ./agent_router.sh start|stop|restart|status
# 端口: 18001 (与 launchd plist / agent_router.py 默认值一致)

PORT=18001
SCRIPT="$HOME/.hermes/scripts/agent_router.py"
VENV_PY="$HOME/.hermes/router-venv/bin/python"
PIDFILE="$HOME/.hermes/scripts/agent_router.pid"
LOGFILE="$HOME/.hermes/scripts/agent_router.log"

start() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        echo "✅ 已在运行 (PID $(cat $PIDFILE))"
        return
    fi
    echo "🚀 启动 Agent Model Router (端口 $PORT)..."
    nohup "$VENV_PY" "$SCRIPT" --port "$PORT" >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 2
    if kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        echo "✅ 启动成功 (PID $(cat $PIDFILE))"
        echo "   OpenAI:  http://127.0.0.1:$PORT/v1/chat/completions"
        echo "   Anthropic: http://127.0.0.1:$PORT/v1/messages"
    else
        echo "❌ 启动失败, 查看日志: $LOGFILE"
        rm -f "$PIDFILE"
    fi
}

stop() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        kill "$(cat $PIDFILE)"
        rm -f "$PIDFILE"
        echo "🛑 已停止"
    else
        echo "未在运行"
    fi
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        echo "✅ 运行中 (PID $(cat $PIDFILE))"
        curl -s --max-time 3 http://127.0.0.1:$PORT/health 2>/dev/null || echo "   (健康检查失败)"
    else
        echo "❌ 未运行"
    fi
}

case "$1" in
    start) start ;;
    stop) stop ;;
    restart) stop; sleep 1; start ;;
    status) status ;;
    *) echo "用法: $0 {start|stop|restart|status}"; exit 1 ;;
esac
