"""Send data/chat.txt as a single chat completion request."""

from util import ChatClient
from util.cli import (
    argument_parser, print_stats, read_prompt, run,
)


def main():
    args = argument_parser("读取文本文件，进行单次会话", with_data=True).parse_args()
    messages = [{"role": "user", "content": read_prompt(args.text)}]
    with ChatClient.from_config(args.config) as client:
        _, stats = client.complete(
            messages, stream=True,
            on_text=lambda chunk: print(chunk, end="", flush=True),
        )
    print(flush=True)
    print_stats(stats)


if __name__ == "__main__":
    run(main)
