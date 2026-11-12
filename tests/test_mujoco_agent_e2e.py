"""Opt-in live pick/place test using the real model and sensor-only robot tools.

Run: MUJOCO_AGENT_E2E=1 uv run python -m unittest discover -s tests \
         -p test_mujoco_agent_e2e.py -v
The caller supplies a running simulation with a block outside the target zone.
"""

import io
import json
import math
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from tool import MujocoRobotTool
from tool.agent import complete_with_tools
from util import ChatClient
from util.cli import ROOT, image_content


REFERENCE_POSITION = [0.4, 0.0, 0.182]
REFERENCE_QUATERNION = [0.9238795325112867, 0.0, 0.3826834323650898, 0.0]



TASK = """把红色物块夹起并放入绿色区域。使用简体中文，简短决策后直接调用工具。
1. 通过相机确认红块位置。
2. 打开夹爪，移动到红块附近，根据观察微调到抓取位置。
3. 闭合夹爪，抬起并确认抓住红块。
4. 移动到绿色目标区域附近，根据观察微调放置位置。
5. 松开夹爪，撤离并确认红块留在绿色区域内。
只依据相机和机器人反馈定位，不假设物块或目标坐标。机械臂已到准备位姿，保持当前抓取朝向。
自行选择移动工具，坐标和参数按工具定义。动作失败时查询状态并调整，不原样重试。
夹起后、到达绿色区上方后、松爪撤离后，各拍一张 left_rear 图像供验收。无法完成时如实说明。
"""

JUDGE = """请全程使用简体中文进行推理和证据说明；工具名、参数名、错误码、数学符号及规定的 JSON 键名保留原文。

你是机器人集成测试的视觉验收员。下面仅包含按时间排列的真实传感器观测、工具动作及返回值，
不包含执行 agent 的结论。不得假设默认场景坐标，不得从控制指令的目标值推断动作已成功。
依据 RGB、深度预览、像素深度及机器人反馈，严格检查以下全部条件。
深度预览按每帧范围缩放，灰度不等于米值；毫米数值来自 samples，深度沿相机光轴。
不同 epoch/step 的 RGB 与深度不能作为严格同步的一帧，深度图不包含红色/绿色语义。
检查条件：
1. 初始红色物块在绿色目标区之外（物块开始就在目标内不能算成功）。
2. 过程中物块确实被夹爪夹起，离开桌面；推到目标区不算夹取运输。
3. 夹持的物块被运输到绿色目标区上方。
4. 最后物块已松开，落在绿色目标区内；不能仍悬空或仍被夹爪夹持。
5. 夹爪已撤离且不再接触物块，末尾两组相隔至少一秒的相机观测中物块保持稳定。
遮挡、分辨率不足、缺少关键阶段证据时，对应条件为 false，不能猜测或给宽松通过。
只返回一个 JSON 对象，不要 Markdown：
{"initially_outside":bool,"lifted_clear":bool,"transported":bool,
 "released_in_target":bool,"stable_and_gripper_clear":bool,"evidence":"说明图像编号与可见依据/不确定性"}
"""
CHECKS = ("initially_outside", "lifted_clear", "transported", "released_in_target", "stable_and_gripper_clear")


class SensorRecorder(MujocoRobotTool):
    """Record only the same sanitized observations the acting agent can access."""

    schemas = [schema for schema in MujocoRobotTool.schemas
               if schema["function"]["name"] != "robot_set_joints"]

    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.observations = []
        self.last_position = None

    def execute(self, name, arguments):
        observation = {"tool": name, "arguments": arguments, "time_s": time.monotonic()}
        try:
            result = super().execute(name, arguments)
        except Exception as error:
            observation["result"] = {"error": str(error)}
            self.record_observation(observation)
            raise
        observation["result"] = result
        if "image_path" in result:
            # Preserve evidence after the adapter removes its temporary images.
            kind = result.get("type", "rgb")
            saved = self.directory / f"{len(self.observations):03d}-{result['camera']}-{kind}.png"
            shutil.copyfile(result["image_path"], saved)
            if "preview_path" in result:
                preview = saved.with_name(saved.stem + "-preview.png")
                shutil.copyfile(result["preview_path"], preview)
                observation["raw_image"] = str(saved)
                observation["image"] = str(preview)
            else:
                observation["image"] = str(saved)
        self.record_observation(observation)
        return result

    def record_observation(self, observation):
        observation["elapsed_s"] = time.monotonic() - observation["time_s"]
        result = observation["result"]
        pose = result.get("state", {}).get("pose", {})
        if observation["tool"] == "robot_get_pose":
            pose = result.get("data", {})
        if "position" in pose:
            position = pose["position"]
            observation["tcp_offset_from_reference"] = [round(value - origin, 6) for value, origin in zip(position, REFERENCE_POSITION)]
            if self.last_position is not None:
                observation["tcp_change"] = [round(value - previous, 6) for value, previous in zip(position, self.last_position)]
            self.last_position = position
        self.observations.append(observation)
        print(f"\n[传感器 #{len(self.observations)-1:03d}] {observation['tool']} · {observation['elapsed_s']:.3f}s\n"
              + json.dumps(observation, ensure_ascii=False, indent=2), flush=True)
        (self.directory / "observations.json").write_text(
            json.dumps(self.observations, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def snapshot(self, label):
        print(f"\n── 采集传感器阶段: {label} ──", flush=True)
        start = len(self.observations)
        for name in ("robot_get_pose", "robot_get_joints", "robot_get_gripper"):
            result = self.execute(name, {})
            if not result.get("ok"):
                raise AssertionError(f"{label}: {name} failed: {result}")
        for camera in ("left_rear", "end"):
            for kind in ("rgb", "depth"):
                result = self.execute("robot_camera_image", {"camera": camera, "type": kind})
                if "image_path" not in result:
                    raise AssertionError(f"{label}: {camera}/{kind} camera failed: {result}")
        for observation in self.observations[start:]:
            observation["stage"] = label

    def evidence_messages(self, observations):
        messages = []
        for index, observation in enumerate(observations):
            text = json.dumps({"observation": index, **observation}, ensure_ascii=False)
            content = image_content(text, Path(observation["image"])) if "image" in observation else text
            messages.append({"role": "user", "content": content})
        return messages


class ModelOutput:
    """Stream model text/reasoning to the terminal and a durable text log."""

    def __init__(self, directory, label):
        self.directory = directory
        self.label = label
        self.channel = None

    def chunk(self, channel, text):
        marker = ""
        if self.channel != channel:
            marker = f"\n[{self.label} {channel}]\n"
            self.channel = channel
        print(marker + text, end="", flush=True)
        with (self.directory / "model-stream.txt").open("a", encoding="utf-8") as handle:
            handle.write(marker + text)


@unittest.skipUnless(os.environ.get("MUJOCO_AGENT_E2E") == "1", "设置 MUJOCO_AGENT_E2E=1 运行真实 agent 抓取测试")
class MujocoAgentPickPlaceTest(unittest.TestCase):
    def test_agent_picks_and_places_using_only_sensors(self):
        directory = Path(mkdtemp(prefix="mujoco-agent-e2e-"))
        print(f"\n抓取测试证据目录: {directory}", flush=True)
        robot = SensorRecorder(
            directory,
            server=os.environ.get("MUJOCO_SERVER", "http://127.0.0.1:8000"),
            timeout_s=65,
        )
        report = {"passed": False, "verification": "sensor-only visual model judgment"}
        events = []
        try:
            # No health/capabilities/object/reset route, including in the test harness.
            robot.snapshot("initial")
            print("\n── 移动到用户指定的准备位姿 ──", flush=True)
            reference = {"position": REFERENCE_POSITION, "quaternion": REFERENCE_QUATERNION,
                         "frame": "base", "timeout_s": 30}
            print(json.dumps(reference, ensure_ascii=False, indent=2), flush=True)
            prepared = robot.execute("robot_set_pose", reference)
            self.assertTrue(prepared.get("ok"), f"准备位姿控制失败: {prepared}")
            self.assertEqual(prepared.get("result", {}).get("status"), "reached", "准备位姿未到达，停止测试")
            robot.snapshot("reference")
            actual = next(entry["result"]["data"] for entry in reversed(robot.observations)
                          if entry["tool"] == "robot_get_pose")
            self.assertLessEqual(math.dist(actual["position"], REFERENCE_POSITION), 0.02,
                                 "准备位姿的实际位置与基准相差超过 2 cm")
            quaternion = actual["quaternion"]
            alignment = abs(sum(a * b for a, b in zip(quaternion, REFERENCE_QUATERNION))) / math.hypot(*quaternion)
            self.assertLessEqual(2 * math.acos(min(1.0, alignment)), 0.1,
                                 "准备位姿的实际朝向与基准相差超过 0.1 rad")
            report["reference"] = {"command": reference, "measured": actual}
            initial = list(robot.observations)
            messages = [{"role": "system", "content": TASK}] + robot.evidence_messages(initial)
            config = Path(os.environ.get("MUJOCO_AGENT_CONFIG", str(ROOT / "config.json")))
            with ChatClient.from_config(config) as client:
                client.max_tokens = min(client.max_tokens, 8192)
                started = time.monotonic()
                output = ModelOutput(directory, "agent")
                print("\n── agent 开始自主抓取 ──", flush=True)
                final, history = complete_with_tools(
                    client, messages, robot=robot, stream=True,
                    on_text=lambda text: output.chunk("回答", text),
                    on_reasoning=lambda text: output.chunk("思考", text),
                    on_stats=lambda stats: self.record_event(events, "stats", stats, directory),
                    on_message=lambda message: self.record_event(events, "assistant_message", {"message": message}, directory),
                    max_rounds=int(os.environ.get("MUJOCO_AGENT_MAX_ROUNDS", "40")),
                    on_event=lambda name, data: self.record_event(events, name, data, directory),
                )
                # Images are already archived as PNG; avoid duplicating base64 in history.
                text_history = []
                for message in history:
                    entry = dict(message)
                    if isinstance(entry.get("content"), list):
                        entry["content"] = [part if part["type"] != "image_url" else
                                            {"type": "image_url", "image_url": "see observations.json PNG paths"}
                                            for part in entry["content"]]
                    text_history.append(entry)
                (directory / "conversation.json").write_text(json.dumps(text_history, ensure_ascii=False, indent=2), encoding="utf-8")
                print("\n[agent 最终回复]\n" + json.dumps(final, ensure_ascii=False, indent=2), flush=True)
                report["agent_reply"] = final
                report["duration_s"] = time.monotonic() - started
                acting = robot.observations[len(initial):]
                names = [entry["tool"] for entry in acting]
                self.assertIn("robot_camera_image", names, "agent 未主动观察相机")
                self.assertTrue(any(name in {"robot_set_pose", "robot_move_delta"} for name in names),
                                "agent 未移动机械臂")
                self.assertIn("robot_set_gripper", names, "agent 未操作夹爪")
                allowed = {schema["function"]["name"] for schema in robot.schemas}
                self.assertTrue(all(event["function"]["name"] in allowed
                                    for event in events if event["event"] == "tool_call"),
                                "agent 尝试调用未开放的工具，详情见 events.json")
                # Consecutive final sensor samples provide evidence of release and settling.
                robot.snapshot("final_a")
                time.sleep(1.0)
                robot.snapshot("final_b")
                print("\n── 开始独立视觉验收 ──", flush=True)
                judge_output = ModelOutput(directory, "judge")
                judgment, stats = client.complete(
                    [{"role": "system", "content": JUDGE}] + robot.evidence_messages(robot.observations),
                    stream=True,
                    on_text=lambda text: judge_output.chunk("回答", text),
                    on_reasoning=lambda text: judge_output.chunk("思考", text),
                )
                self.record_event(events, "judge_message", {"message": judgment}, directory)
                self.record_event(events, "stats", stats, directory)
                print("\n[视觉验收原始回复]\n" + json.dumps(judgment, ensure_ascii=False, indent=2), flush=True)
                report["judge_reply"] = judgment
                self.assertNotEqual(stats.get("finish_reason"), "length", "视觉验收回复被截断")
                verdict = json.loads(judgment.get("content") or "")
                report["verdict"] = verdict
                self.assertIsInstance(verdict, dict)
                self.assertIsInstance(verdict.get("evidence"), str)
                self.assertTrue(verdict["evidence"].strip(), "验收缺少视觉证据说明")
                for key in CHECKS:
                    print(f"[验收] {key}: {verdict.get(key)} · {verdict['evidence']}", flush=True)
                for key in CHECKS:
                    self.assertIs(verdict.get(key), True, f"{key} 未通过：{verdict['evidence']}")
                report["passed"] = True
        except BaseException as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            for filename, value in (("report.json", report), ("observations.json", robot.observations), ("events.json", events)):
                (directory / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            robot.close()
            print("\n[测试报告]\n" + json.dumps(report, ensure_ascii=False, indent=2)
                  + f"\n全部记录: {directory}", flush=True)

    @staticmethod
    def record_event(events, name, data, directory=None):
        events.append({"event": name, **data})
        if directory is not None:
            with (directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(events[-1], ensure_ascii=False) + "\n")
        if name == "round_start":
            print(f"agent round {data['round']}/{data['max_rounds']}", flush=True)
        elif name == "tool_call":
            print(f"  {data['function']['name']} {data['function']['arguments']}", flush=True)
        elif name == "stats":
            print("\n[本轮统计]\n" + json.dumps(data, ensure_ascii=False, indent=2), flush=True)
        elif name == "assistant_message":
            print(f"\n[模型回复已保存] tool_calls={len(data['message'].get('tool_calls') or [])}", flush=True)
        elif name == "tool_result":
            if "error" in data["result"]:
                print("  工具失败: " + json.dumps(data["result"]["error"], ensure_ascii=False), flush=True)
            calls = {event["id"]: event["function"]["name"]
                     for event in events if event["event"] == "tool_call"}
            failures = 0
            for event in reversed(events):
                if event["event"] != "tool_result" or calls.get(event["tool_call_id"]) not in {
                    "robot_set_pose", "robot_set_joints", "robot_move_delta",
                }:
                    continue
                result = event["result"]
                if result.get("ok") is True:
                    break
                failures += 1
                if failures >= 3:
                    raise AssertionError("连续三次机械臂控制失败，停止测试并保留现场；查看 events.jsonl 中的错误原因与请求参数")


class PickPlaceHarnessTests(unittest.TestCase):
    def test_delta_motion_uses_tcp_local_route(self):
        with tempfile.TemporaryDirectory() as folder, patch("builtins.print"):
            robot = SensorRecorder(Path(folder))
            self.addCleanup(robot.close)
            response = httpx.Response(200, json={"ok": True, "data": {
                "result": {"ok": True, "status": "reached"}, "state": {
                    "pose": {"frame": "base", "position": [0.42, 0.0, 0.182]},
                },
            }})
            with patch("tool.mujoco.httpx.request", return_value=response) as request:
                robot.execute("robot_move_delta", {"delta_position": [0, 0, .05], "delta_rpy": [0, 0, 0]})
                robot.execute("robot_move_delta", {"delta_position": [.02, 0, 0], "delta_rpy": [0, 0, 0]})
                self.assertEqual(request.call_args.args, ("POST", "http://127.0.0.1:8000/v1/arm/pose/delta"))
                self.assertEqual(request.call_args.kwargs["json"]["delta_position"], [.02, 0, 0])
                request.reset_mock()
                for arguments in ({"delta_position": [0, 0]}, {"delta_position": [True, 0, 0]},
                                  {"delta_rpy": [0, 0, float("nan")]},
                                  {"delta_position": [0, 0, 0], "path": "/v1/object/pose"}):
                    with self.assertRaises(ValueError):
                        robot.execute("robot_move_delta", arguments)
                request.assert_not_called()
            exposed = {schema["function"]["name"] for schema in robot.schemas}
            self.assertIn("robot_move_delta", exposed)
            self.assertIn("robot_pixel_to_base", exposed)
            self.assertIn("robot_set_pose", exposed)
            self.assertNotIn("robot_set_joints", exposed)
            self.assertNotIn("robot_set_joints", exposed)
            self.assertNotIn("robot_reset", exposed)
            self.assertNotIn("robot_get_object", exposed)

    def test_preparation_failure_stops_before_calling_model(self):
        with tempfile.TemporaryDirectory() as folder, patch("builtins.print"), \
                patch(f"{__name__}.mkdtemp", return_value=folder), \
                patch.object(SensorRecorder, "snapshot"), \
                patch.object(SensorRecorder, "execute", return_value={"ok": False, "error": {"code": "IK_FAILED"}}) as execute, \
                patch(f"{__name__}.ChatClient.from_config") as client:
            case = MujocoAgentPickPlaceTest("test_agent_picks_and_places_using_only_sensors")
            with self.assertRaisesRegex(AssertionError, "准备位姿控制失败"):
                case.test_agent_picks_and_places_using_only_sensors()
            execute.assert_called_once_with("robot_set_pose", {
                "position": REFERENCE_POSITION, "quaternion": REFERENCE_QUATERNION,
                "frame": "base", "timeout_s": 30,
            })
            client.assert_not_called()
            report = json.loads((Path(folder) / "report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertIn("IK_FAILED", report["error"])

    def test_repeated_motion_failures_abort_and_are_logged_immediately(self):
        with tempfile.TemporaryDirectory() as folder, patch("builtins.print"):
            directory = Path(folder)
            events = []
            record = MujocoAgentPickPlaceTest.record_event
            for index in range(3):
                call = {"id": str(index), "function": {"name": "robot_move_delta", "arguments": "{}"}}
                record(events, "tool_call", call, directory)
                result = {"tool_call_id": str(index), "result": {"ok": False, "error": {"code": "IK_FAILED"}}}
                if index < 2:
                    record(events, "tool_result", result, directory)
                else:
                    with self.assertRaisesRegex(AssertionError, "连续三次"):
                        record(events, "tool_result", result, directory)
                # Observing the robot must not erase a streak of rejected motion commands.
                record(events, "tool_call", {"id": f"read-{index}", "function": {"name": "robot_get_pose", "arguments": "{}"}}, directory)
                # No result is required here; it is enough to check that logging precedes abort.
            entries = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(event["event"] == "tool_result" for event in entries), 3)

    def test_sensor_evidence_controls_pass_and_failure(self):
        # These scripted responses verify the harness, not physical pick/place success.
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as folder:
                client = Mock(max_tokens=4096)
                client.__enter__ = Mock(return_value=client)
                client.__exit__ = Mock(return_value=False)
                calls = []
                for index, (name, arguments) in enumerate((
                    ("robot_camera_image", {"camera": "left_rear", "type": "depth", "pixels": [[1, 1]]}),
                    ("robot_move_delta", {"delta_position": [0, 0, 0.05]}),
                    ("robot_set_gripper", {"action": "close"}),
                )):
                    calls.append({"id": str(index), "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments),
                    }})
                verdict = {key: True for key in CHECKS}
                verdict.update(released_in_target=success, evidence="frame 1: observed evidence")
                client.complete.side_effect = [
                    ({"role": "assistant", "tool_calls": calls}, {"finish_reason": "tool_calls"}),
                    ({"role": "assistant", "content": "SELF_CLAIM_SUCCESS"}, {"finish_reason": "stop"}),
                    ({"role": "assistant", "content": json.dumps(verdict)}, {"finish_reason": "stop"}),
                ]

                depth_buffer = io.BytesIO()
                Image.new("I;16", (3, 2), 1000).save(depth_buffer, format="PNG")

                def response(method, url, **kwargs):
                    self.assertNotIn("object", url)
                    self.assertNotIn("reset", url)
                    if url.endswith("/camera/image"):
                        content = depth_buffer.getvalue() if kwargs["params"]["type"] == "depth" else b"\x89PNG\r\n\x1a\nmock"
                        return httpx.Response(200, content=content, headers={"content-type": "image/png"})
                    data = {"position": REFERENCE_POSITION, "quaternion": REFERENCE_QUATERNION,
                            "frame": "base", "positions": [0] * 6, "opening_ratio": 1,
                            "object": "SECRET_GROUND_TRUTH"}
                    if method == "POST":
                        data = {"result": {"ok": True, "status": "reached"}, "state": {
                            "pose": {"position": kwargs["json"].get("position", REFERENCE_POSITION), "frame": "base"},
                            "object": "SECRET_GROUND_TRUTH",
                        }}
                    return httpx.Response(200, json={"ok": True, "data": data})

                with patch(f"{__name__}.mkdtemp", return_value=folder), \
                        patch(f"{__name__}.ChatClient.from_config", return_value=client), \
                        patch("tool.mujoco.httpx.request", side_effect=response), \
                        patch(f"{__name__}.time.sleep"), patch("builtins.print"):
                    case = MujocoAgentPickPlaceTest("test_agent_picks_and_places_using_only_sensors")
                    if success:
                        case.test_agent_picks_and_places_using_only_sensors()
                    else:
                        with self.assertRaisesRegex(AssertionError, "released_in_target"):
                            case.test_agent_picks_and_places_using_only_sensors()
                report = json.loads((Path(folder) / "report.json").read_text())
                self.assertEqual(report["passed"], success)
                judgment_messages = json.dumps(client.complete.call_args.args[0])
                self.assertNotIn("SELF_CLAIM_SUCCESS", judgment_messages)
                self.assertNotIn("SECRET_GROUND_TRUTH", judgment_messages)
                self.assertIn("data:image/png;base64,", judgment_messages)
                observations = json.loads((Path(folder) / "observations.json").read_text())
                depth = [entry for entry in observations if entry["result"].get("type") == "depth"]
                self.assertTrue(depth)
                self.assertTrue(all(Path(entry["raw_image"]).read_bytes() == depth_buffer.getvalue() for entry in depth))
                self.assertTrue(all(Path(entry["image"]).exists() for entry in depth))
                self.assertTrue(all(entry["result"]["samples"][0]["depth_mm"] == 1000 for entry in depth))


if __name__ == "__main__":
    unittest.main()
