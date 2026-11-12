"""Bounded OpenAI function-calling loop for local detection."""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .yolo26 import TOOL_SCHEMA


def complete_with_tools(client, messages, detector=None, max_rounds=5, on_stats=None,
                        stream=False, on_text=None, on_reasoning=None, on_event=None,
                        robot=None, on_message=None):
    """Return a new conversation only after a final assistant response succeeds."""
    if max_rounds < 1:
        raise ValueError("max_rounds 必须大于 0")
    schemas = ([TOOL_SCHEMA] if detector is not None else [])
    if robot is not None:
        schemas += robot.schemas
    pending = list(messages)
    for round_number in range(1, max_rounds + 1):
        if on_event:
            on_event("round_start", {"round": round_number, "max_rounds": max_rounds})
        callbacks = {"on_reasoning": on_reasoning} if on_reasoning else {}
        message, stats = client.complete(
            pending, tools=schemas, stream=stream, on_text=on_text, **callbacks,
        )
        if on_message:
            on_message(message)
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
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")
                with redirect_stdout(sys.stderr):
                    if function["name"] == "detect_objects" and detector is not None:
                        result = detector.detect(**arguments)
                    elif robot is not None and function["name"] in {
                        schema["function"]["name"] for schema in robot.schemas
                    }:
                        result = robot.execute(function["name"], arguments)
                    else:
                        raise ValueError(f"未知工具: {function['name']}")
            except Exception as error:
                # Tool failures are data for the model, not a broken tool protocol.
                result = {"error": str(error)}
            if on_event:
                on_event("tool_result", {"tool_call_id": call["id"], "result": result})
            pending.append({
                "role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })
        # Append camera observations after all tool replies to preserve tool-call protocol.
        for call in calls:
            if call["function"]["name"] in {"robot_camera_image", "robot_pixel_to_base", "robot_set_gripper"}:
                tool_reply = next(entry for entry in reversed(pending)
                                  if entry.get("role") == "tool" and entry.get("tool_call_id") == call["id"])
                observation = json.loads(tool_reply["content"])
                image_path = observation.get("marked_image_path", observation.get("preview_path", observation.get("image_path")))
                if image_path:
                    from util.cli import image_content

                    pending.append({"role": "user", "content": image_content(
                        "机器人相机观测（depth 为灰度预览，精确毫米值和本帧缩放范围见工具返回；不能当作 RGB。"
                        "pixel_to_base 附带的是标有所选像素的 RGB，洋红色圆圈和十字的青色中心即 pixel，用于核对选点。"
                        "夹爪动作附带的是手部相机 end 的当前 RGB，用于核对开合）：\n"
                        + json.dumps(observation, ensure_ascii=False),
                        Path(image_path),
                    )})
    raise ValueError(f"工具调用达到 {max_rounds} 轮上限，本轮对话未保存")
