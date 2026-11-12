"""Interactive streaming chat with in-memory conversation history."""

from openai import APIError

from util import ChatClient
from util.cli import argument_parser, error_text, print_message, print_stats, run


def chat_loop(client, detector=None, max_tool_rounds=5, robot=None):
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
            if detector is None and robot is None:
                message, stats = client.complete(
                    pending, stream=True,
                    on_text=lambda chunk: print(chunk, end="", flush=True),
                )
                completed = pending + [message]
            else:
                from tool.agent import complete_with_tools

                message, completed = complete_with_tools(
                    client, pending, detector, max_tool_rounds, on_stats=print_stats,
                    robot=robot,
                )
                print_message(message)
        except (OSError, APIError, ValueError, KeyError, TypeError) as error:
            print(f"\n请求失败: {error_text(error)}")
            continue
        print(flush=True)
        messages = completed
        if detector is None and robot is None:
            print_stats(stats)


def main():
    from tool.config import add_tool_arguments, load_tool_settings, load_robot_settings

    parser = argument_parser("与 OpenAI 兼容服务进行连续对话，可启用 YOLO26s 和机器人工具")
    parser.add_argument("--enable-mujoco", action="store_true", help="启用受限的 MuJoCo 机器人工具调用")
    parser.add_argument("--enable-yolo", action="store_true", help="启用 YOLO26s 工具调用（非流式）")
    add_tool_arguments(parser)
    args = parser.parse_args()
    detector = None
    robot = None
    max_tool_rounds = 5
    if args.enable_yolo or args.enable_mujoco:
        settings = load_tool_settings(args)
        max_tool_rounds = settings["max_tool_rounds"]
    if args.enable_yolo:
        from tool import YOLO26Tool

        detector = YOLO26Tool(settings["model"], settings["device"], confidence=settings["confidence"])
    if args.enable_mujoco:
        from tool import MujocoRobotTool

        robot = MujocoRobotTool(**load_robot_settings(args))
    try:
        with ChatClient.from_config(args.config) as client:
            chat_loop(client, detector, max_tool_rounds, robot=robot)
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    run(main)
