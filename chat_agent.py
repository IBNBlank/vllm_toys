"""Interactive streaming chat with in-memory conversation history."""

from openai import APIError

from util import ChatClient
from util.cli import argument_parser, error_text, print_stats, run


def chat_loop(client):
    messages = []
    print(f"对话已启动（{client.model}）。/clear 清空上下文，/exit 或 /quit 退出；Ctrl+C 退出。")
    while True:
        try:
            text = input("\n你> ").strip()
        except EOFError:
            print("\n已退出。")
            return
        if text in ("/exit", "/quit"):
            return
        if text == "/clear":
            messages.clear()
            print("上下文已清空。")
            continue
        if not text:
            continue
        pending = messages + [{"role": "user", "content": text}]
        print(f"{client.model}> ", end="", flush=True)
        try:
            message, stats = client.complete(
                pending, stream=True,
                on_text=lambda chunk: print(chunk, end="", flush=True),
            )
        except (OSError, APIError, ValueError, KeyError, TypeError) as error:
            print(f"\n请求失败: {error_text(error)}")
            continue
        print(flush=True)
        messages = pending + [message]
        print_stats(stats)


def main():
    args = argument_parser("与 OpenAI 兼容服务进行连续流式对话").parse_args()
    with ChatClient.from_config(args.config) as client:
        chat_loop(client)


if __name__ == "__main__":
    run(main)
