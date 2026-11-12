"""Shared tool configuration and command-line overrides."""

import json
import math
from pathlib import Path

from util.cli import ROOT


def add_tool_arguments(parser):
    parser.add_argument("--mujoco-server", help="覆盖 MuJoCo HTTP 服务地址")
    parser.add_argument("--mujoco-timeout", type=float, help="覆盖 MuJoCo HTTP 超时秒数")
    parser.add_argument("--tool-config", type=Path, default=ROOT / "tool.json", help="工具配置文件路径")
    parser.add_argument("--yolo-model", help="覆盖工具配置中的 YOLO 权重名称或本地路径")
    parser.add_argument("--yolo-device", help="覆盖推理设备，默认 GPU 0，可设为 cpu")
    parser.add_argument("--confidence", type=float, help="覆盖检测置信度阈值")
    parser.add_argument("--max-tool-rounds", type=int, help="覆盖模型请求轮数上限，包括最终回答")


def load_tool_settings(args):
    with args.tool_config.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("工具配置必须是 JSON 对象")
    yolo = config.get("yolo", {})
    if not isinstance(yolo, dict):
        raise ValueError("工具配置的 yolo 必须是 JSON 对象")
    settings = {
        "model": args.yolo_model if args.yolo_model is not None else yolo.get("model", "yolo26s.pt"),
        "device": args.yolo_device if args.yolo_device is not None else yolo.get("device", "0"),
        "confidence": args.confidence if args.confidence is not None else yolo.get("confidence", 0.25),
        "max_tool_rounds": args.max_tool_rounds if args.max_tool_rounds is not None else config.get("max_tool_rounds", 5),
    }
    for key in ("model", "device"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"yolo.{key} 必须是非空字符串")
    confidence = settings["confidence"]
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("yolo.confidence 必须是 0 到 1 之间的数值")
    if type(settings["max_tool_rounds"]) is not int or settings["max_tool_rounds"] < 1:
        raise ValueError("max_tool_rounds 必须是正整数")
    return settings


def load_robot_settings(args):
    with args.tool_config.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("mujoco", {}), dict):
        raise ValueError("工具配置的 mujoco 必须是 JSON 对象")
    robot = config.get("mujoco", {})
    server = args.mujoco_server if args.mujoco_server is not None else robot.get("server", "http://127.0.0.1:8000")
    timeout = args.mujoco_timeout if args.mujoco_timeout is not None else robot.get("timeout_s", 20)
    if not isinstance(server, str) or not server.strip():
        raise ValueError("mujoco.server 必须是非空字符串")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("mujoco.timeout_s 必须是正数")
    return {"server": server, "timeout_s": timeout}
