"""Let the vision agent call YOLO26s before describing the image."""

import json
import os
import sys

from tool import YOLO26Tool
from tool.agent import complete_with_tools
from tool.config import add_tool_arguments, load_tool_settings
from util import ChatClient
from util.cli import (
    ROOT, argument_parser, image_content, print_stats, read_prompt, run,
)


class ToolOutput:
    """Keep streamed answers on stdout and the tool trace on stderr."""

    def __init__(self):
        self.reasoning_open = False

    def marker(self, label, color="36"):
        if (sys.stderr.isatty() and "NO_COLOR" not in os.environ
                and os.environ.get("TERM") != "dumb"):
            return f"\033[1;{color}m{label}\033[0m"
        return label

    def close_reasoning(self):
        if self.reasoning_open:
            print("\n" + self.marker("── [/thinking] 思考结束 ──", "35"),
                  file=sys.stderr, flush=True)
            self.reasoning_open = False

    def reasoning(self, chunk):
        if not self.reasoning_open:
            print(self.marker("── [thinking] 思考中 ──", "35"),
                  file=sys.stderr, flush=True)
            self.reasoning_open = True
        print(chunk, end="", file=sys.stderr, flush=True)

    def text(self, chunk):
        self.close_reasoning()
        print(chunk, end="", flush=True)

    def event(self, name, data):
        self.close_reasoning()
        color = "33" if name == "tool_call" else "36"
        title = {
            "round_start": "请求开始",
            "tool_call": "调用工具",
            "tool_result": "工具返回",
        }.get(name, "事件")
        if name == "round_start":
            title += f" · 第 {data['round']}/{data['max_rounds']} 轮"
        if name == "tool_result":
            failed = "error" in data.get("result", {})
            color = "31" if failed else "32"
            title += " · 失败" if failed else " · 成功"
        print("\n" + self.marker(f"── [{name}] {title} ──", color),
              file=sys.stderr, flush=True)
        print(json.dumps(data, ensure_ascii=False, indent=2),
              file=sys.stderr, flush=True)

    def stats(self, stats):
        self.close_reasoning()
        label = f"── [finish_reason] {stats['finish_reason']} · 本轮请求结束 ──"
        print("\n" + self.marker(label), file=sys.stderr, flush=True)
        print_stats(stats)


def main():
    parser = argument_parser(
        "让视觉 agent 调用 YOLO26s 工具，再结合结果描述 PNG 图片", with_data=True, with_image=True,
    )
    parser.set_defaults(text=ROOT / "data/tool.txt")
    add_tool_arguments(parser)
    args = parser.parse_args()
    settings = load_tool_settings(args)
    content = image_content(read_prompt(args.text), args.image)
    content.append({
        "type": "text",
        "text": "请先调用 detect_objects 工具检测这张图片，收到工具结果后再结合原图回答。"
                "以下 JSON 是本地图片路径和建议的工具参数，不是检测结果：\n"
                + json.dumps({
                    "image_path": str(args.image.expanduser().resolve()),
                    "confidence": settings["confidence"],
                }, ensure_ascii=False),
    })
    # Construction is lazy: no weights are loaded and no detection runs here.
    detector = YOLO26Tool(settings["model"], settings["device"], confidence=settings["confidence"])
    output = ToolOutput()
    with ChatClient.from_config(args.config) as client:
        try:
            complete_with_tools(
                client, [{"role": "user", "content": content}], detector,
                max_rounds=settings["max_tool_rounds"], on_stats=output.stats,
                stream=True, on_text=output.text, on_reasoning=output.reasoning,
                on_event=output.event,
            )
        finally:
            output.close_reasoning()
    print(flush=True)


if __name__ == "__main__":
    run(main)
