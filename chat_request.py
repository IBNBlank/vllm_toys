"""Send data/chat.txt as a single chat completion request."""

from util import ChatClient
from util.cli import (
    argument_parser, print_message, print_stats, read_prompt, run,
)


def main():
    args = argument_parser("读取文本文件，进行单次会话", with_data=True).parse_args()
    messages = [{"role": "user", "content": read_prompt(args.text)}]
    with ChatClient.from_config(args.config) as client:
        message, stats = client.complete(messages)
    print_message(message)
    print_stats(stats)


if __name__ == "__main__":
    run(main)
