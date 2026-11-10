"""CLI arguments, input files, terminal output and error presentation."""

import argparse
import base64
import sys
from pathlib import Path

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError


ROOT = Path(__file__).resolve().parents[1]


def argument_parser(description, with_data=False, with_image=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json", help="配置文件路径")
    if with_data:
        parser.add_argument("--text", type=Path, default=ROOT / "data/chat.txt", help="提示词文件路径")
    if with_image:
        parser.add_argument("--image", type=Path, default=ROOT / "data/image.png", help="PNG 图片路径")
    return parser


def read_prompt(path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"提示词文件为空: {path}")
    return text


def image_content(text, path):
    image = path.read_bytes()
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"图片必须是 PNG 格式: {path}")
    encoded = base64.b64encode(image).decode("ascii")
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
    ]


def print_stats(stats):
    values = [f"总耗时: {stats['elapsed']:.3f}s"]
    if stats["first_text"] is not None:
        values.append(f"首个文本分片: {stats['first_text']:.3f}s")
    usage = stats["usage"]
    if usage:
        values.extend(
            f"{label}: {usage.get(key, '未返回')}"
            for key, label in (
                ("prompt_tokens", "输入 tokens"),
                ("completion_tokens", "输出 tokens"),
                ("total_tokens", "总 tokens"),
            )
        )
    else:
        values.append("token 用量: 服务端未返回")
    if stats["finish_reason"] == "length":
        values.append("已达到 token 上限，回复可能被截断")
    print("[" + " | ".join(values) + "]", file=sys.stderr, flush=True)


def print_message(message):
    if message.get("reasoning_content"):
        print(message["reasoning_content"])
    print(message.get("content") or "")


def error_text(error):
    if isinstance(error, APIStatusError):
        return f"HTTP {error.status_code}: {error.message}"
    if isinstance(error, APITimeoutError):
        return "请求超时，请检查服务状态或增大 timeout"
    if isinstance(error, APIConnectionError):
        return f"无法连接服务端: {error}"
    return str(error)


def run(main):
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。", file=sys.stderr)
        sys.exit(130)
    except (OSError, APIError, ValueError, KeyError, TypeError) as error:
        print(f"错误: {error_text(error)}", file=sys.stderr)
        sys.exit(1)
