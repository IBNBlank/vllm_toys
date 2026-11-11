"""Bounded OpenAI function-calling loop for local detection."""

import json
import sys
from contextlib import redirect_stdout

from .yolo26 import TOOL_SCHEMA


def complete_with_tools(client, messages, detector, max_rounds=5, on_stats=None,
                        stream=False, on_text=None, on_reasoning=None, on_event=None):
    """Return a new conversation only after a final assistant response succeeds."""
    if max_rounds < 1:
        raise ValueError("max_rounds 必须大于 0")
    pending = list(messages)
    for round_number in range(1, max_rounds + 1):
        if on_event:
            on_event("round_start", {"round": round_number, "max_rounds": max_rounds})
        callbacks = {"on_reasoning": on_reasoning} if on_reasoning else {}
        message, stats = client.complete(
            pending, tools=[TOOL_SCHEMA], stream=stream, on_text=on_text, **callbacks,
        )
        if on_stats:
            on_stats(stats)
        if stats.get("finish_reason") == "length":
            raise ValueError("工具模式回复被截断，请增大 max_tokens")
        pending.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return message, pending
        for call in calls:
            if on_event:
                on_event("tool_call", call)
            try:
                function = call["function"]
                if function["name"] != "detect_objects":
                    raise ValueError(f"未知工具: {function['name']}")
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")
                with redirect_stdout(sys.stderr):
                    result = detector.detect(**arguments)
            except Exception as error:
                # Tool failures are data for the model, not a broken tool protocol.
                result = {"error": str(error)}
            if on_event:
                on_event("tool_result", {"tool_call_id": call["id"], "result": result})
            pending.append({
                "role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })
    raise ValueError(f"工具调用达到 {max_rounds} 轮上限，本轮对话未保存")
