import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletionChunk

from util import ChatClient
from util.cli import ROOT, image_content


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, payload))
        self.server.auth_headers.append(self.headers.get("Authorization"))
        if self.server.fail_next:
            self.server.fail_next = False
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "temporarily unavailable", "type": "server_error"}}')
            return
        self.send_response(200)
        if self.server.mode == "tools" and payload.get("tools") and payload["messages"][-1]["role"] != "tool":
            arguments = {"image_path": "missing.png"}
            content = payload["messages"][-1].get("content")
            if isinstance(content, list):
                arguments = json.loads(content[-1]["text"].split("\n", 1)[1])
            if payload.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                encoded = json.dumps(arguments)
                events = [
                    {"choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": 0, "id": "detect-1", "type": "function",
                        "function": {"name": "detect_objects", "arguments": encoded[:8]},
                    }]}}]},
                    {"choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": 0, "function": {"arguments": encoded[8:]},
                    }]}}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                ]
                if self.server.reasoning_field:
                    events.insert(0, {"choices": [{"index": 0, "delta": {
                        self.server.reasoning_field: "先检测图片。",
                    }}]})
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "choices": [{"message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "detect-1", "type": "function", "function": {
                        "name": "detect_objects", "arguments": json.dumps(arguments),
                    }}],
                }, "finish_reason": "tool_calls"}],
            }).encode())
            return
        if payload.get("stream"):
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                {"choices": [{"index": 0, "delta": {"content": "你"}}]},
                {"choices": [{"index": 0, "delta": {"content": "好"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
            ]
            if self.server.reasoning_field:
                events.insert(1, {"choices": [{"index": 0, "delta": {
                    self.server.reasoning_field: "根据检测结果定位。",
                }}]})
            self.wfile.write(b": heartbeat\r\n\r\n")
            if self.server.mode == "truncated":
                events = events[:3]
            elif self.server.mode == "reasoning":
                events.insert(1, {"choices": [{"index": 0, "delta": {"reasoning_content": "思考"}}]})
            elif self.server.mode == "no_usage":
                events = events[:-1]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\r\n\r\n")
        else:
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "你好"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }).encode())


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.requests = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.requests.clear()
        self.server.auth_headers = []
        self.server.fail_next = False
        self.server.mode = "normal"
        self.server.reasoning_field = None
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = {
            "base_url": f"http://127.0.0.1:{self.server.server_port}/v1",
            "model": "qwen-robot", "max_tokens": 128, "timeout": 5,
        }
        self.config_path = self.directory / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.client = ChatClient.from_config(self.config_path)
        self.addCleanup(self.client.close)

    def invoke(self, script, *args, input=None):
        return subprocess.run(
            [sys.executable, str(ROOT / script), "--config", str(self.config_path), *map(str, args)],
            input=input, text=True, capture_output=True, cwd=self.directory, timeout=10,
        )

    def test_text_default_path_and_usage(self):
        result = self.invoke("chat_request.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "你好")
        self.assertIn("总 tokens: 6", result.stderr)
        path, payload = self.server.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["model"], "qwen-robot")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertTrue(payload["stream"])
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.assertIn("首个文本分片:", result.stderr)
        self.assertEqual(payload["messages"][0]["content"], (ROOT / "data/chat.txt").read_text().strip())

    def test_interleaved_stream_tool_calls(self):
        events = [
            {"tool_calls": [
                {"index": 1, "id": "second", "type": "function",
                 "function": {"name": "detect_", "arguments": '{"image_path":'}},
                {"index": 0, "id": "first", "type": "function",
                 "function": {"name": "detect_objects", "arguments": '{"image_path":"a'}},
            ]},
            {"tool_calls": [
                {"index": 0, "function": {"arguments": '.png"}'}},
                {"index": 1, "function": {"name": "objects", "arguments": '"b.png"}'}},
            ]},
        ]
        for finish in ["tool_calls", None]:
            chunks = [ChatCompletionChunk(
                id="chunk", object="chat.completion.chunk", created=0, model="test",
                choices=[{"index": 0, "delta": delta}],
            ) for delta in events]
            if finish:
                chunks.append(ChatCompletionChunk(
                    id="chunk", object="chat.completion.chunk", created=0, model="test",
                    choices=[{"index": 0, "delta": {}, "finish_reason": finish}],
                ))
            response = MagicMock()
            response.__iter__.return_value = iter(chunks)
            with patch.object(self.client._sdk.chat.completions, "create", return_value=response):
                if finish is None:
                    with self.assertRaisesRegex(ValueError, "未收到 finish_reason"):
                        self.client.complete([], stream=True, tools=[{}])
                    continue
                message, stats = self.client.complete([], stream=True, tools=[{}])
            self.assertEqual(stats["finish_reason"], "tool_calls")
            self.assertIsNone(message["content"])
            self.assertEqual([call["id"] for call in message["tool_calls"]], ["first", "second"])
            self.assertEqual([json.loads(call["function"]["arguments"])
                              for call in message["tool_calls"]],
                             [{"image_path": "a.png"}, {"image_path": "b.png"}])
            self.assertTrue(all(call["function"]["name"] == "detect_objects"
                                for call in message["tool_calls"]))

    def test_stream_usage_and_callback(self):
        chunks = []
        message, stats = self.client.complete([{"role": "user", "content": "你好"}], True, chunks.append)
        self.assertEqual(chunks, ["你", "好"])
        self.assertEqual(message, {"role": "assistant", "content": "你好"})
        self.assertEqual(stats["usage"]["total_tokens"], 6)
        self.assertIsNotNone(stats["first_text"])
        self.assertTrue(self.server.requests[-1][1]["stream_options"]["include_usage"])

    def test_agent_history_and_clear(self):
        result = self.invoke("chat_agent.py", input="第一轮\n第二轮\n/clear\n新对话\n/exit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        requests = [item[1] for item in self.server.requests]
        self.assertEqual([len(item["messages"]) for item in requests], [1, 3, 1])
        self.assertEqual(requests[1]["messages"][1], {"role": "assistant", "content": "你好"})
        self.assertEqual(requests[2]["messages"][0]["content"], "新对话")

    def test_plain_image_payload_without_tools(self):
        image = self.directory / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = self.invoke("image_request.py", "--image", image)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.server.requests), 1)
        payload = self.server.requests[0][1]
        self.assertTrue(payload["stream"])
        self.assertEqual(result.stdout, "你好\n")
        self.assertIn("总 tokens: 6", result.stderr)
        self.assertIn("首个文本分片:", result.stderr)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        content = payload["messages"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["text"], (ROOT / "data/chat.txt").read_text().strip())
        self.assertEqual(content[1]["type"], "image_url")

    def test_tool_request_payload(self):
        image = self.directory / "image.png"
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=")
        image.write_bytes(png)
        from tool_request import main

        self.server.mode = "tools"
        detection = {"width": 1, "height": 1, "detections": []}
        with patch("sys.argv", ["tool_request.py", "--config", str(self.config_path),
                                "--image", str(image), "--confidence", "0.4",
                                "--yolo-device", "cpu", "--yolo-model", "custom.pt"]), \
                patch("tool_request.YOLO26Tool") as tool, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                patch("sys.stderr", new_callable=io.StringIO):
            def detect(**kwargs):
                self.assertEqual(len(self.server.requests), 1)
                return detection

            tool.return_value.detect.side_effect = detect
            main()
            tool.assert_called_once_with("custom.pt", "cpu", confidence=0.4)
            tool.return_value.detect.assert_called_once_with(image_path=str(image), confidence=0.4)
            self.assertEqual(stdout.getvalue().strip(), "你好")
        self.assertEqual(len(self.server.requests), 2)
        first, second = [payload for _, payload in self.server.requests]
        self.assertEqual(first["tools"][0]["function"]["name"], "detect_objects")
        self.assertEqual(len(first["messages"]), 1)
        self.assertTrue(first["stream"])
        self.assertTrue(second["stream"])
        content = first["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], (ROOT / "data/tool.txt").read_text().strip())
        self.assertEqual(json.loads(content[2]["text"].split("\n", 1)[1]),
                         {"image_path": str(image), "confidence": 0.4})
        self.assertEqual(second["messages"][-2]["tool_calls"][0]["id"], "detect-1")
        self.assertEqual(second["messages"][-1]["role"], "tool")
        self.assertEqual(second["messages"][-1]["tool_call_id"], "detect-1")
        self.assertEqual(json.loads(second["messages"][-1]["content"]), detection)
        prefix, encoded = content[1]["image_url"]["url"].split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(base64.b64decode(encoded), png)

    def test_tool_request_reasoning_and_trace(self):
        from tool_request import main

        image = self.directory / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.server.mode = "tools"
        for field in ("reasoning_content", "reasoning", "thinking"):
            self.server.requests.clear()
            self.server.reasoning_field = field
            with self.subTest(field=field), \
                    patch("sys.argv", ["tool_request.py", "--config", str(self.config_path),
                                       "--image", str(image)]), \
                    patch("tool_request.YOLO26Tool") as tool, \
                    patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                    patch("sys.stderr", new_callable=io.StringIO) as stderr:
                def detect(**kwargs):
                    self.assertIn("先检测图片。", stderr.getvalue())
                    print("YOLO diagnostic")
                    return {"width": 1, "height": 1, "detections": []}

                tool.return_value.detect.side_effect = detect
                main()
                self.assertEqual(stdout.getvalue(), "你好\n")
                trace = stderr.getvalue()
                for text in ("[round_start]", "[thinking]", "先检测图片。", "根据检测结果定位。",
                             "[tool_call]", "detect_objects", "[tool_result]", '"detections": []',
                             "YOLO diagnostic", "[finish_reason] tool_calls", "[finish_reason] stop"):
                    self.assertIn(text, trace)
                self.assertEqual(trace.count("[/thinking]"), 2)
                assistant = self.server.requests[1][1]["messages"][1]
                self.assertEqual(assistant["reasoning_content"], "先检测图片。")

    def test_tool_request_failure_is_returned_to_model(self):
        from tool_request import main

        image = self.directory / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        prompt = self.directory / "prompt.txt"
        prompt.write_text("自定义描述要求", encoding="utf-8")
        self.server.mode = "tools"
        with patch("sys.argv", ["tool_request.py", "--config", str(self.config_path),
                                "--image", str(image), "--text", str(prompt)]), \
                patch("tool_request.YOLO26Tool") as tool:
            tool.return_value.detect.side_effect = RuntimeError("missing weights")
            with patch("sys.stdout", new_callable=io.StringIO), \
                    patch("sys.stderr", new_callable=io.StringIO):
                main()
        self.assertEqual(len(self.server.requests), 2)
        content = self.server.requests[0][1]["messages"][0]["content"]
        self.assertEqual(content[0]["text"], "自定义描述要求")
        result = self.server.requests[-1][1]["messages"][-1]
        self.assertEqual(json.loads(result["content"]), {"error": "missing weights"})

    def test_tool_request_does_not_detect_without_model_tool_call(self):
        from tool_request import main

        image = self.directory / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        with patch("sys.argv", ["tool_request.py", "--config", str(self.config_path),
                                "--image", str(image)]), \
                patch("tool_request.YOLO26Tool") as tool, \
                patch("sys.stdout", new_callable=io.StringIO), \
                patch("sys.stderr", new_callable=io.StringIO):
            main()
            tool.return_value.detect.assert_not_called()
        self.assertEqual(len(self.server.requests), 1)

    def test_missing_image_fails_before_request(self):
        result = self.invoke("image_request.py", "--image", self.directory / "missing.png")
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing.png", result.stderr)
        self.assertEqual(self.server.requests, [])

    def test_invalid_image(self):
        image = self.directory / "invalid.png"
        image.write_bytes(b"not a PNG")
        with self.assertRaisesRegex(ValueError, "PNG"):
            image_content("test", image)

    def test_interrupted_stream_is_rejected(self):
        self.server.mode = "truncated"
        with self.assertRaisesRegex(ValueError, "未收到 finish_reason"):
            self.client.complete([], stream=True)

    def test_reasoning_stream(self):
        self.server.mode = "reasoning"
        chunks = []
        message, _ = self.client.complete([], stream=True, on_text=chunks.append)
        self.assertEqual(message["reasoning_content"], "思考")
        self.assertEqual(message["content"], "你好")
        self.assertEqual(chunks, ["思考", "你", "好"])

    def test_stream_without_usage(self):
        self.server.mode = "no_usage"
        message, stats = self.client.complete([], stream=True)
        self.assertEqual(message["content"], "你好")
        self.assertIsNone(stats["usage"])

    def test_http_error_and_no_retry(self):
        self.server.fail_next = True
        result = self.invoke("chat_request.py")
        self.assertEqual(result.returncode, 1)
        self.assertIn("HTTP 503", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(len(self.server.requests), 1)

    def test_custom_gateway_model_and_key(self):
        config = dict(self.config, base_url=self.config["base_url"] + "/gateway", model="other-model", api_key="test-key")
        with ChatClient(config) as client:
            client.complete([{"role": "user", "content": "test"}])
        path, payload = self.server.requests[-1]
        self.assertEqual(path, "/v1/gateway/chat/completions")
        self.assertEqual(payload["model"], "other-model")
        self.assertEqual(self.server.auth_headers[-1], "Bearer test-key")

    def test_empty_auth_settings(self):
        config = dict(self.config, api_key="", api_key_env="", stream_include_usage=True)
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unrelated-key"}):
            with ChatClient.from_config(self.config_path) as client:
                message, _ = client.complete([])
        self.assertEqual(message["content"], "你好")
        self.assertEqual(self.server.auth_headers[-1], "Bearer not-needed")
        self.assertEqual(json.loads(self.config_path.read_text(encoding="utf-8")), config)

    def test_environment_key_takes_precedence(self):
        config = dict(self.config, api_key="unused", api_key_env="TEST_CHAT_KEY")
        with patch.dict(os.environ, {"TEST_CHAT_KEY": "env-key"}):
            with ChatClient(config) as client:
                client.complete([])
        self.assertEqual(self.server.auth_headers[-1], "Bearer env-key")

    def test_missing_environment_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TEST_CHAT_KEY"):
                ChatClient(dict(self.config, api_key_env="TEST_CHAT_KEY"))
        self.assertEqual(self.server.requests, [])

    def test_stream_usage_option_can_be_disabled(self):
        with ChatClient(dict(self.config, stream_include_usage=False)) as client:
            client.complete([], stream=True)
        self.assertNotIn("stream_options", self.server.requests[-1][1])

    def test_reuse_and_close(self):
        with ChatClient(self.config) as client:
            client.complete([])
            client.complete([])
            self.assertFalse(client._sdk.is_closed())
        self.assertTrue(client._sdk.is_closed())
        self.assertEqual(len(self.server.requests), 2)

    def test_agent_registers_executes_and_returns_tool_result(self):
        self.server.mode = "tools"
        result = self.invoke("chat_agent.py", "--enable-yolo", input="检测 missing.png\n/exit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("你好", result.stdout)
        self.assertEqual(len(self.server.requests), 2)
        first, second = [payload for _, payload in self.server.requests]
        self.assertEqual(first["tools"][0]["function"]["name"], "detect_objects")
        self.assertEqual(first["tool_choice"], "auto")
        self.assertFalse(first["stream"])
        self.assertEqual(second["messages"][-2]["tool_calls"][0]["id"], "detect-1")
        tool_result = second["messages"][-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertEqual(tool_result["tool_call_id"], "detect-1")
        self.assertIn("本地图片不存在", json.loads(tool_result["content"])["error"])

    def test_failed_agent_turn_is_not_saved(self):
        self.server.fail_next = True
        result = self.invoke("chat_agent.py", input="失败的一轮\n再试一次\n/exit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HTTP 503", result.stdout)
        self.assertEqual([len(payload["messages"]) for _, payload in self.server.requests], [1, 1])


if __name__ == "__main__":
    unittest.main()
