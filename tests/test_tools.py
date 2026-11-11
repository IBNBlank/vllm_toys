import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tool import YOLO26Tool
from tool.agent import complete_with_tools
from tool.config import add_tool_arguments, load_tool_settings


def call(arguments, name="detect_objects", id="call-1"):
    return {"id": id, "type": "function", "function": {
        "name": name, "arguments": arguments,
    }}


def reply(calls=None):
    message = {"role": "assistant", "content": "done"}
    if calls:
        message["tool_calls"] = calls
    return message, {"finish_reason": "tool_calls" if calls else "stop"}


class ToolTests(unittest.TestCase):
    def test_prediction_and_reuse(self):
        def tensor(data):
            value = Mock()
            value.cpu.return_value.tolist.return_value = data
            return value

        result = SimpleNamespace(
            orig_shape=(100, 200), names={0: "person"},
            boxes=SimpleNamespace(xyxy=tensor([[1, 2, 30, 40]]), conf=tensor([0.9]), cls=tensor([0])),
        )
        model = Mock()
        model.predict.return_value = [result]
        factory = Mock(return_value=model)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"fake image for mocked prediction")
            with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=factory)}):
                detector = YOLO26Tool()
                output = detector.detect(str(image))
                detector.detect(str(image), 0.5)
        factory.assert_called_once_with("yolo26s.pt", task="detect")
        self.assertEqual(output["width"], 200)
        self.assertEqual(output["height"], 100)
        self.assertEqual(output["detections"], [{
            "class_id": 0, "class_name": "person", "confidence": 0.9,
            "bbox_xyxy": [1.0, 2.0, 30.0, 40.0],
        }])
        self.assertFalse(model.predict.call_args.kwargs["save"])
        self.assertEqual(model.predict.call_args.kwargs["device"], "0")
        detector.confidence = 0.65
        result.boxes = None
        with tempfile.NamedTemporaryFile(suffix=".png") as image:
            self.assertEqual(detector.detect(image.name)["detections"], [])
        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.65)

    def test_validation_before_model_load(self):
        detector = YOLO26Tool()
        for value in [-1, 2, float("nan"), float("inf"), True, "0.5", None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                detector.detect("missing.png", value)
        with self.assertRaisesRegex(ValueError, "本地图片不存在"):
            detector.detect("https://example.com/image.png")
        self.assertIsNone(detector._model)

    def test_tool_config_and_overrides(self):
        parser = argparse.ArgumentParser()
        add_tool_arguments(parser)
        defaults = load_tool_settings(parser.parse_args([]))
        self.assertEqual(defaults["device"], "0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool.json"
            path.write_text(json.dumps({
                "max_tool_rounds": 3,
                "yolo": {"model": "custom.pt", "device": "1", "confidence": 0.6},
            }))
            args = parser.parse_args(["--tool-config", str(path)])
            self.assertEqual(load_tool_settings(args), {
                "model": "custom.pt", "device": "1", "confidence": 0.6, "max_tool_rounds": 3,
            })
            args = parser.parse_args([
                "--tool-config", str(path), "--yolo-device", "cpu", "--confidence", "0",
                "--yolo-model", "override.pt", "--max-tool-rounds", "2",
            ])
            self.assertEqual(load_tool_settings(args), {
                "model": "override.pt", "device": "cpu", "confidence": 0.0, "max_tool_rounds": 2,
            })
            for invalid in [{"yolo": []}, {"yolo": {"confidence": True}},
                            {"max_tool_rounds": 0}, {"yolo": {"device": ""}}]:
                path.write_text(json.dumps(invalid))
                with self.subTest(config=invalid), self.assertRaises(ValueError):
                    load_tool_settings(parser.parse_args(["--tool-config", str(path)]))

    def test_successful_tool_loop_preserves_protocol(self):
        client, detector = Mock(), Mock()
        detector.detect.return_value = {"detections": []}
        client.complete.side_effect = [reply([call('{"image_path":"a.png"}')]), reply()]
        original = [{"role": "user", "content": "detect a.png"}]
        message, history = complete_with_tools(client, original, detector)
        self.assertEqual(message["content"], "done")
        self.assertEqual(len(original), 1)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[2]["tool_call_id"], "call-1")
        self.assertEqual(json.loads(history[2]["content"]), {"detections": []})
        detector.detect.assert_called_once_with(image_path="a.png")

    def test_invalid_calls_return_errors(self):
        client, detector = Mock(), Mock()
        client.complete.side_effect = [reply([
            call("broken", id="bad-json"), call("[]", id="bad-object"),
            call("{}", name="unknown", id="unknown"),
        ]), reply()]
        _, history = complete_with_tools(client, [], detector)
        self.assertEqual([entry["tool_call_id"] for entry in history[1:4]],
                         ["bad-json", "bad-object", "unknown"])
        for entry in history[1:4]:
            self.assertIn("error", json.loads(entry["content"]))
        detector.detect.assert_not_called()

    def test_round_limit_and_failed_request_leave_original_untouched(self):
        client, detector = Mock(), Mock()
        detector.detect.return_value = {}
        client.complete.return_value = reply([call('{"image_path":"a.png"}')])
        original = [{"role": "user", "content": "detect"}]
        with self.assertRaisesRegex(ValueError, "上限"):
            complete_with_tools(client, original, detector, max_rounds=2)
        self.assertEqual(len(original), 1)
        self.assertEqual(client.complete.call_count, 2)
        client.complete.side_effect = ValueError("request failed")
        with self.assertRaisesRegex(ValueError, "request failed"):
            complete_with_tools(client, original, detector)
        self.assertEqual(len(original), 1)

    def test_truncated_tool_reply_not_executed(self):
        client, detector = Mock(), Mock()
        message, _ = reply([call("{}")])
        client.complete.return_value = message, {"finish_reason": "length"}
        with self.assertRaisesRegex(ValueError, "截断"):
            complete_with_tools(client, [], detector)
        detector.detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
