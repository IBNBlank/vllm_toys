import argparse
import base64
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from tool import MUJOCO_TOOL_SCHEMAS, MujocoRobotTool
from tool.agent import complete_with_tools
from tool.config import add_tool_arguments, load_robot_settings


POINT_ARGS = {
    "camera": "end", "pixel": [10, 20], "depth_m": .5,
    "tcp_position_base": [0, 0, 0], "tcp_quaternion_base": [1, 0, 0, 0],
    "camera_position_tcp": [0, 0, 0], "camera_rotation_tcp": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "intrinsics": {"fx": 100, "fy": 100, "cx": 10, "cy": 20, "width": 30, "height": 40},
    "epoch": 3, "step": 42,
}


def rgb_png(size=(30, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def depth_png():
    image = Image.new("I", (3, 2))
    image.putdata([0, 1000, 2000, 65535, 1000, 2000])
    buffer = io.BytesIO()
    image.convert("I;16").save(buffer, format="PNG")
    return buffer.getvalue()


def call(arguments, name, id):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def reply(calls=None):
    message = {"role": "assistant", "content": "done"}
    if calls:
        message["tool_calls"] = calls
    return message, {"finish_reason": "tool_calls" if calls else "stop"}


class MujocoToolTests(unittest.TestCase):
    def setUp(self):
        self.robot = MujocoRobotTool()
        self.addCleanup(self.robot.close)

    def test_only_explicit_robot_routes_and_arguments_reach_network(self):
        names = {item["function"]["name"] for item in MUJOCO_TOOL_SCHEMAS}
        self.assertEqual(names, {
            "robot_get_joints", "robot_set_joints", "robot_get_pose", "robot_set_pose",
            "robot_move_delta", "robot_pixel_to_base", "robot_get_gripper", "robot_set_gripper", "robot_camera_image",
        })
        self.assertNotIn("robot_reset", names)
        self.assertNotIn("robot_get_object", names)
        self.assertNotIn("robot_set_object", names)
        self.assertNotIn("robot_get_in_target", names)
        invalid = [
            ("robot_reset", {}), ("robot_get_object", {}),
            ("/v1/object/pose", {}), ("robot_get_pose", {"path": "/v1/object/pose"}),
            ("robot_set_joints", {"positions": [0] * 5}),
            ("robot_set_joints", {"positions": [True] * 6}),
            ("robot_move_delta", {"delta_position": [float("nan"), 0, 0]}),
            ("robot_move_delta", {"delta_position": [0, 0]}),
            ("robot_pixel_to_base", {"pixel": [1, 2]}),
            ("robot_pixel_to_base", {"pixel": [1.5, 2], "epoch": 1, "step": 2}),
            ("robot_pixel_to_base", {"pixel": [1, 2], "epoch": 1, "step": -1}),
            ("robot_set_pose", {"position": [0, 0, 0], "frame": "world"}),
            ("robot_set_pose", {"position": [0, 0, float("nan")]}),
            ("robot_set_gripper", {}),
            ("robot_set_gripper", {"action": "open", "opening_ratio": 1}),
            ("robot_set_gripper", {"opening_ratio": 2}),
            ("robot_get_pose", {"frame": "world"}),
            ("robot_camera_image", {"camera": "../object"}),
            ("robot_camera_image", {"type": "infrared"}),
            ("robot_camera_image", {"pixels": [[0, 0]]}),
            ("robot_camera_image", {"type": "depth", "pixels": [[True, 0]]}),
            ("robot_camera_image", {"type": "depth", "pixels": [[0.5, 0]]}),
            ("robot_camera_image", {"type": "depth", "pixels": [[-1, 0]]}),
            ("robot_camera_image", {"type": "depth", "pixels": []}),
            ("robot_camera_image", {"type": "depth", "pixels": [[0, 0]] * 65}),
            ("robot_set_pose", {"position": [0, 0, 0], "timeout_s": -1}),
        ]
        with patch("tool.mujoco.httpx.request") as request:
            for name, arguments in invalid:
                with self.subTest(name=name, arguments=arguments), self.assertRaises(ValueError):
                    self.robot.execute(name, arguments)
            request.assert_not_called()

    def test_delta_pose_and_pixel_point_use_allowlisted_routes(self):
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(200, json={
            "ok": True, "data": {"result": {"ok": True, "status": "reached"},
                                  "state": {"pose": {"frame": "base", "position": [0, 0, 0]}, "object": "SECRET"}},
        })) as request:
            result = self.robot.execute("robot_move_delta", {"delta_position": [.1, 0, 0], "delta_rpy": [0, 0, .2]})
        self.assertEqual(request.call_args.args, ("POST", "http://127.0.0.1:8000/v1/arm/pose/delta"))
        self.assertEqual(result["result"]["status"], "reached")
        self.assertNotIn("SECRET", json.dumps(result))

        body = {"ok": True, "data": {"frame": "base", "position": [.2, .3, .4],
                                       "camera": "end", "pixel": [10, 20], "depth_m": .5,
                                       "epoch": 3, "step": 42, "object": "SECRET"}}
        rgb = io.BytesIO()
        Image.new("RGB", (30, 40), (0, 0, 0)).save(rgb, format="PNG")
        with patch("tool.mujoco.httpx.request", side_effect=[
            httpx.Response(200, json=body),
            httpx.Response(200, content=rgb.getvalue(), headers={"content-type": "image/png"}),
        ]) as request:
            result = self.robot.execute("robot_pixel_to_base", dict(POINT_ARGS))
        self.assertEqual(request.call_args_list[0].args, ("POST", "http://127.0.0.1:8000/v1/camera/point"))
        self.assertEqual(request.call_args_list[0].kwargs["json"], POINT_ARGS)
        self.assertEqual(request.call_args_list[1].args, ("GET", "http://127.0.0.1:8000/v1/camera/image"))
        self.assertEqual(request.call_args_list[1].kwargs["params"], {"camera": "end", "type": "rgb"})
        self.assertEqual(result["data"]["position"], [.2, .3, .4])
        self.assertEqual(result["marked_pixel"], [10, 20])
        with Image.open(result["marked_image_path"]) as marked:
            self.assertEqual(marked.getpixel((10, 20)), (0, 255, 255))
            self.assertEqual(marked.getpixel((24, 20)), (255, 0, 255))
            self.assertEqual(marked.getpixel((10, 6)), (255, 0, 255))
        self.assertNotIn("SECRET", json.dumps(result))

    def test_pixel_point_minimal_and_explicit_pose(self):
        schema = next(item["function"]["parameters"] for item in MUJOCO_TOOL_SCHEMAS
                      if item["function"]["name"] == "robot_pixel_to_base")
        self.assertEqual(set(schema["required"]), {"camera", "pixel", "depth_m"})
        for camera in ("end", "opposite", "left"):
            minimal = {"camera": camera, "pixel": [10, 20], "depth_m": .5}
            explicit = {**minimal, "tcp_position_base": [1, 2, 3],
                        "tcp_quaternion_base": [1, 0, 0, 0]}
            for arguments in (minimal, explicit):
                with self.subTest(arguments=arguments), patch("tool.mujoco.httpx.request", side_effect=[
                    httpx.Response(200, json={"ok": True, "data": {"frame": "base", "position": [.1, .2, .3]}}),
                    httpx.Response(200, content=rgb_png(), headers={"content-type": "image/png"}),
                ]) as request:
                    result = self.robot.execute("robot_pixel_to_base", arguments)
                    self.assertTrue(result["ok"])
                    self.assertEqual(result["data"]["position"], [.1, .2, .3])
                    self.assertEqual(request.call_args_list[0].kwargs["json"], arguments)
                    self.assertEqual(request.call_args_list[0].args, ("POST", "http://127.0.0.1:8000/v1/camera/point"))

    def test_pixel_point_incomplete_pose_and_calibration_rejected(self):
        minimal = {"camera": "end", "pixel": [10, 20], "depth_m": .5}
        invalid = [
            {"pixel": [10, 20], "depth_m": .5},
            {**minimal, "tcp_position_base": [1, 2, 3]},
            {**minimal, "tcp_quaternion_base": [1, 0, 0, 0]},
            {**minimal, "camera_position_tcp": [0, 0, 0]},
            {key: value for key, value in POINT_ARGS.items()
             if key not in {"tcp_position_base", "tcp_quaternion_base"}},
        ]
        with patch("tool.mujoco.httpx.request") as request:
            for arguments in invalid:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    self.robot.execute("robot_pixel_to_base", arguments)
            request.assert_not_called()

    def test_pixel_point_errors_are_sanitized(self):
        for code, status in (("INVALID_ARGUMENT", 422),):
            with self.subTest(code=code), patch("tool.mujoco.httpx.request", return_value=httpx.Response(
                status, json={"ok": False, "error": {"code": code, "message": "SECRET"}},
            )) as request:
                result = self.robot.execute("robot_pixel_to_base", dict(POINT_ARGS))
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["code"], code)
            self.assertNotIn("marked_image_path", result)
            self.assertNotIn("SECRET", json.dumps(result))
            request.assert_called_once()

    def test_gripper_success_attaches_end_rgb_and_failure_does_not(self):
        description = next(item["function"]["description"] for item in MUJOCO_TOOL_SCHEMAS
                           if item["function"]["name"] == "robot_set_gripper")
        self.assertIn("end", description)
        success = httpx.Response(200, json={"ok": True, "data": {
            "result": {"ok": True, "status": "reached"}, "state": {"gripper": {"opening_ratio": 1, "object": "SECRET"}},
        }})
        with patch("tool.mujoco.httpx.request", side_effect=[
            success, httpx.Response(200, content=rgb_png(), headers={"content-type": "image/png"}),
        ]) as request:
            result = self.robot.execute("robot_set_gripper", {"action": "open"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["camera"], "end")
        self.assertEqual(result["type"], "rgb")
        self.assertEqual(Path(result["image_path"]).read_bytes(), rgb_png())
        self.assertEqual(request.call_args_list[1].kwargs["params"], {"camera": "end", "type": "rgb"})
        self.assertNotIn("SECRET", json.dumps(result))
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(
            422, json={"ok": False, "error": {"code": "INVALID_ARGUMENT", "message": "SECRET"}},
        )) as request:
            failed = self.robot.execute("robot_set_gripper", {"action": "close"})
        self.assertFalse(failed["ok"])
        self.assertNotIn("image_path", failed)
        request.assert_called_once()

    def test_gripper_rgb_is_sent_to_model(self):
        client = Mock()
        client.complete.side_effect = [reply([call('{"action":"close"}', "robot_set_gripper", "grip")]), reply()]
        with patch("tool.mujoco.httpx.request", side_effect=[
            httpx.Response(200, json={"ok": True, "data": {"result": {"ok": True, "status": "reached"}}}),
            httpx.Response(200, content=rgb_png(), headers={"content-type": "image/png"}),
        ]):
            _, history = complete_with_tools(client, [], robot=self.robot)
        result = json.loads(history[1]["content"])
        self.assertEqual(history[2]["role"], "user")
        image_bytes = base64.b64decode(history[2]["content"][1]["image_url"]["url"].split(",", 1)[1])
        self.assertEqual(image_bytes, Path(result["image_path"]).read_bytes())

    def test_pixel_point_marked_rgb_is_sent_to_model(self):
        client = Mock()
        client.complete.side_effect = [reply([
            call(json.dumps({"camera": "opposite", "pixel": [10, 20], "depth_m": .5}), "robot_pixel_to_base", "point"),
        ]), reply()]
        point = httpx.Response(200, json={"ok": True, "data": {
            "frame": "base", "position": [.1, .2, .3], "camera": "opposite", "pixel": [10, 20], "depth_m": .5,
        }})
        with patch("tool.mujoco.httpx.request", side_effect=[
            point, httpx.Response(200, content=rgb_png(), headers={"content-type": "image/png"}),
        ]):
            _, history = complete_with_tools(client, [], robot=self.robot)
        result = json.loads(history[1]["content"])
        self.assertEqual(result["data"]["position"], [.1, .2, .3])
        self.assertEqual(history[2]["role"], "user")
        image_bytes = base64.b64decode(history[2]["content"][1]["image_url"]["url"].split(",", 1)[1])
        self.assertEqual(image_bytes, Path(result["marked_image_path"]).read_bytes())
        description = next(item["function"]["description"] for item in MUJOCO_TOOL_SCHEMAS
                           if item["function"]["name"] == "robot_pixel_to_base")
        self.assertIn("洋红色圆圈", description)

    def test_actions_filter_ground_truth_on_success_and_failure(self):
        for success in (True, False):
            body = {"ok": success, "error": {"code": "TIMEOUT", "details": {"object": "SECRET"}}, "data": {
                "result": {"ok": success, "status": "blocked", "object": "SECRET", "error": 0.01},
                "state": {"object": {"position": "SECRET"}, "target": "SECRET", "step": 4,
                          "pose": {"position": [1, 2, 3], "quaternion": [1, 0, 0, 0], "hidden": "SECRET"},
                          "joints": {"positions": [0] * 6}, "gripper": {"opening_ratio": 0.5}},
            }}
            response = httpx.Response(200 if success else 504, json=body)
            with patch("tool.mujoco.httpx.request", return_value=response) as request:
                result = self.robot.execute("robot_set_joints", {"positions": [0] * 6, "timeout_s": 30})
            self.assertNotIn("SECRET", json.dumps(result))
            self.assertEqual(result["state"]["step"], 4)
            self.assertEqual(result["result"]["status"], "blocked")
            self.assertEqual(result["ok"], success)
            self.assertEqual(request.call_args.args, ("POST", "http://127.0.0.1:8000/v1/arm/joints"))
            self.assertEqual(request.call_args.kwargs["timeout"], 35)
            self.assertFalse(request.call_args.kwargs["follow_redirects"])

    def test_queries_filter_unknown_fields(self):
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(200, json={
            "ok": True, "data": {"position": [0, 0, 0], "frame": "base", "object": "SECRET"},
        })) as request:
            output = self.robot.execute("robot_get_pose", {"frame": "base"})
        self.assertEqual(output["data"], {"position": [0, 0, 0], "frame": "base"})
        self.assertEqual(request.call_args.kwargs["params"], {"frame": "base"})

    def test_loop_rejects_forbidden_calls_and_returns_camera_to_model(self):
        client = Mock()
        client.complete.side_effect = [reply([
            call("{}", "robot_reset", "reset"),
            call("{}", "robot_get_object", "object"),
            call("{}", "robot_camera_image", "camera"),
        ]), reply()]
        png = b"\x89PNG\r\n\x1a\nmock image"
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(
            200, content=png, headers={"content-type": "image/png"},
        )) as request:
            _, history = complete_with_tools(client, [], robot=self.robot)
        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["params"], {"type": "rgb"})
        self.assertIn("error", json.loads(history[1]["content"]))
        self.assertIn("error", json.loads(history[2]["content"]))
        image_path = Path(json.loads(history[3]["content"])["image_path"])
        self.assertEqual(image_path.read_bytes(), png)
        self.assertEqual(history[4]["role"], "user")
        self.assertEqual(history[4]["content"][1]["type"], "image_url")
        self.assertEqual(client.complete.call_args.kwargs["tools"], MUJOCO_TOOL_SCHEMAS)
        self.robot.close()
        self.assertFalse(image_path.exists())

    def test_depth_preserves_uint16_values_and_queries_exact_pixels(self):
        png = depth_png()
        frame_headers = {
            "x-tcp-position-base": base64.urlsafe_b64encode(b"[0,0,0.2]").decode(),
            "x-tcp-quaternion-base": base64.urlsafe_b64encode(b"[1,0,0,0]").decode(),
            "x-camera-position-tcp": base64.urlsafe_b64encode(b"[0.1,0,0.2]").decode(),
            "x-camera-rotation-tcp": base64.urlsafe_b64encode(b"[[1,0,0],[0,1,0],[0,0,1]]").decode(),
            "x-intrinsics": base64.urlsafe_b64encode(b"{\"fx\":100,\"fy\":100,\"cx\":1,\"cy\":1,\"width\":3,\"height\":2}").decode(),
        }
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(
            200, content=png, headers={"content-type": "image/png", "x-epoch": "2", "x-step": "123", "x-object": "SECRET", **frame_headers},
        )) as request:
            result = self.robot.execute("robot_camera_image", {
                "camera": "opposite", "type": "depth", "pixels": [[1, 0], [2, 1], [0, 0], [0, 1]],
            })
        self.assertEqual(request.call_args.kwargs["params"], {"camera": "opposite", "type": "depth"})
        self.assertEqual(Path(result["image_path"]).read_bytes(), png)
        self.assertEqual(result["encoding"], "uint16_png")
        self.assertEqual((result["width"], result["height"]), (3, 2))
        self.assertEqual(result["tcp_position_base"], [0, 0, 0.2])
        self.assertEqual(result["intrinsics"]["width"], 3)
        self.assertEqual(result["samples"], [
            {"pixel": [1, 0], "valid": True, "depth_mm": 1000, "depth_m": 1.0},
            {"pixel": [2, 1], "valid": True, "depth_mm": 2000, "depth_m": 2.0},
            {"pixel": [0, 0], "valid": False, "depth_mm": None, "depth_m": None},
            {"pixel": [0, 1], "valid": False, "depth_mm": None, "depth_m": None},
        ])
        self.assertEqual((result["preview"]["near_mm"], result["preview"]["far_mm"]), (1000, 2000))
        self.assertNotIn("SECRET", json.dumps(result))
        with Image.open(result["preview_path"]) as preview:
            self.assertEqual(preview.mode, "L")
            self.assertEqual([preview.getpixel((u, 0)) for u in range(3)], [0, 1, 255])
        paths = [Path(result[key]) for key in ("image_path", "preview_path")]
        self.robot.close()
        self.assertTrue(all(not path.exists() for path in paths))

    def test_depth_preview_and_metadata_are_sent_to_model(self):
        client = Mock()
        client.complete.side_effect = [reply([call('{"type":"depth"}', "robot_camera_image", "depth")]), reply()]
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(
            200, content=depth_png(), headers={"content-type": "image/png"},
        )):
            _, history = complete_with_tools(client, [], robot=self.robot)
        result = json.loads(history[1]["content"])
        self.assertEqual(result["samples"][0]["pixel"], [1, 1])
        self.assertEqual(result["samples"][0]["depth_mm"], 1000)
        content = history[2]["content"]
        self.assertIn('"unit": "mm"', content[0]["text"])
        image_bytes = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
        self.assertEqual(image_bytes, Path(result["preview_path"]).read_bytes())
        with Image.open(io.BytesIO(image_bytes)) as preview:
            self.assertEqual(preview.mode, "L")

    def test_depth_handles_flat_and_invalid_frames(self):
        for depth in (0, 65535, 1500):
            with self.subTest(depth=depth):
                buffer = io.BytesIO()
                Image.new("I;16", (2, 1), depth).save(buffer, format="PNG")
                with patch("tool.mujoco.httpx.request", return_value=httpx.Response(
                    200, content=buffer.getvalue(), headers={"content-type": "image/png"},
                )):
                    output = self.robot.execute("robot_camera_image", {"type": "depth"})
                valid = depth == 1500
                self.assertIs(output["samples"][0]["valid"], valid)
                self.assertEqual(output["preview"]["near_mm"], depth if valid else None)
                self.assertEqual(output["preview"]["far_mm"], depth if valid else None)
                with Image.open(output["preview_path"]) as preview:
                    self.assertEqual(preview.getpixel((1, 0)), 128 if valid else 0)

    def test_depth_rejects_wrong_encoding_and_out_of_bounds_pixels(self):
        buffer = io.BytesIO()
        Image.new("RGB", (3, 2)).save(buffer, format="PNG")
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(
            200, content=buffer.getvalue(), headers={"content-type": "image/png"},
        )):
            with self.assertRaisesRegex(ValueError, "uint16"):
                self.robot.execute("robot_camera_image", {"type": "depth"})
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(
            200, content=depth_png(), headers={"content-type": "image/png"},
        )):
            with self.assertRaisesRegex(ValueError, "超出"):
                self.robot.execute("robot_camera_image", {"type": "depth", "pixels": [[3, 0]]})

    def test_pose_rejection_has_diagnostics_and_duplicate_never_reaches_server(self):
        rejected = httpx.Response(422, json={"ok": False, "error": {
            "code": "IK_FAILED", "message": "SECRET object position", "details": {"object": "SECRET"},
        }, "data": {"state": {"object": "SECRET"}, "result": {"code": "IK_FAILED", "message": "SECRET"}}})
        success = httpx.Response(200, json={"ok": True, "data": {"result": {"ok": True, "status": "reached"}}})
        arguments = {"position": [0.4, 0, 0.5], "frame": "base"}
        with patch("tool.mujoco.httpx.request", side_effect=[rejected, success, rejected]) as request:
            first = self.robot.execute("robot_set_pose", arguments)
            self.assertEqual(first["error"]["http_status"], 422)
            self.assertIn("逆运动学", first["error"]["message"])
            self.assertNotIn("SECRET", json.dumps(first))
            duplicate = self.robot.execute("robot_set_pose", {**arguments, "frame": "base", "timeout_s": 30})
            self.assertEqual(duplicate["error"]["code"], "POSE_ALREADY_REJECTED")
            self.assertEqual(request.call_count, 1)
            self.robot.execute("robot_set_joints", {"positions": [0] * 6})
            self.robot.execute("robot_set_pose", arguments)
            self.assertEqual(request.call_count, 3)

    def test_invalid_quaternion_is_rejected_before_network(self):
        with patch("tool.mujoco.httpx.request") as request:
            for quaternion in ([0, 0, 0, 0], [1, 0, 1, 0], [1e308] * 4):
                with self.subTest(quaternion=quaternion), self.assertRaisesRegex(ValueError, "单位四元数"):
                    self.robot.execute("robot_set_pose", {"position": [0.4, 0, 0.5], "quaternion": quaternion})
            request.assert_not_called()

    def test_fastapi_validation_details_are_sanitized(self):
        with patch("tool.mujoco.httpx.request", return_value=httpx.Response(422, json={
            "detail": [{"loc": ["body", "quaternion"], "type": "value_error", "msg": "SECRET", "input": "SECRET"}],
        })):
            output = self.robot.execute("robot_set_pose", {"position": [0.4, 0, 0.5]})
        self.assertEqual(output["error"]["code"], "INVALID_ARGUMENT")
        self.assertEqual(output["error"]["fields"], ["quaternion"])
        self.assertNotIn("SECRET", json.dumps(output))

    def test_chat_loop_uses_robot_without_yolo(self):
        from chat_agent import chat_loop

        client = Mock(model="test-model")
        client.complete.side_effect = [reply([call("{}", "robot_get_joints", "joints")]), reply()]
        with patch("builtins.input", side_effect=["查询关节", "/exit"]), \
                patch("builtins.print"), patch("chat_agent.print_stats"), \
                patch("tool.mujoco.httpx.request", return_value=httpx.Response(
                    200, json={"ok": True, "data": {"positions": [0] * 6}},
                )) as request:
            chat_loop(client, robot=self.robot)
        request.assert_called_once()
        self.assertEqual(client.complete.call_args.kwargs["tools"], MUJOCO_TOOL_SCHEMAS)

    def test_config_and_overrides(self):
        parser = argparse.ArgumentParser()
        add_tool_arguments(parser)
        settings = load_robot_settings(parser.parse_args([
            "--mujoco-server", "http://localhost:9000", "--mujoco-timeout", "30",
        ]))
        self.assertEqual(settings, {"server": "http://localhost:9000", "timeout_s": 30})
        with self.assertRaises(ValueError):
            load_robot_settings(parser.parse_args(["--mujoco-timeout", "nan"]))


if __name__ == "__main__":
    unittest.main()
