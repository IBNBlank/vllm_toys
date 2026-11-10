"""Send data/chat.txt and data/image.png as one multimodal request."""

from util import ChatClient
from util.cli import (
    argument_parser, image_content, print_message,
    print_stats, read_prompt, run,
)


def main():
    args = argument_parser(
        "读取文本和 PNG 图片，进行单次会话", with_data=True, with_image=True,
    ).parse_args()
    content = image_content(read_prompt(args.text), args.image)
    with ChatClient.from_config(args.config) as client:
        message, stats = client.complete([{"role": "user", "content": content}])
    print_message(message)
    print_stats(stats)


if __name__ == "__main__":
    run(main)
