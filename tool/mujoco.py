"""Restricted adapter for the pinned mujoco_robot_cli HTTP protocol.

Only the routes below are reachable. Never forward complete simulation snapshots.
"""

import base64
import io
import json
import math
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image


def _schema(name, description, properties=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": list(required), "additionalProperties": False},
    }}


def _vector(size, description):
    return {"type": "array", "items": {"type": "number"},
            "minItems": size, "maxItems": size, "description": description}


_FRAME = {"type": "string", "enum": ["base"], "default": "base",
          "description": "所有绝对 pose 均以机械臂 base_link 为基准，位置单位米；运动方向按 base 轴：+X前、+Y左、+Z上。"}
_TIMEOUT = {"type": "number", "exclusiveMinimum": 0, "description": "动作超时秒数；超时后先查询状态，勿盲目重试。"}
TOOL_SCHEMAS = [
    _schema("robot_get_joints", "查询机械臂六个关节的状态，角度单位为弧度。"),
    _schema("robot_set_joints", "控制机械臂六个关节；检查返回的 reached/blocked 状态。", {
        "positions": _vector(6, "六个目标关节角，单位弧度"), "timeout_s": _TIMEOUT,
    }, ["positions"]),
    _schema("robot_get_pose", "查询末端位姿。TCP 位置对标夹爪闭合时的指尖：原点在两指尖之间，+X 指向指尖前方，+Y 为开合方向。张开时指尖离开该点，位姿原点不随开合移动。位置和姿态均以机械臂 base_link 为基准，位置单位米，四元数顺序 wxyz。", {"frame": _FRAME}),
    _schema("robot_set_pose", "控制末端绝对位姿。position 是夹爪闭合时指尖中点的目标位置，不是张开指尖或腕部位置。位置以 base_link 为基准，位置单位米，四元数顺序 wxyz。frame 固定为 base。省略四元数时保持当前朝向。IK_FAILED 表示目标无解，应观察当前状态并调整目标，不要原样重试。检查 reached/blocked 状态。", {
        "position": _vector(3, "闭合指尖中点的目标 xyz，单位米，base_link 坐标系"),
        "quaternion": _vector(4, "单位四元数 wxyz，模长必须接近 1；保持朝向时省略"), "frame": _FRAME, "timeout_s": _TIMEOUT,
    }, ["position"]),
    _schema("robot_move_delta", "在指令开始时的 TCP 当前坐标系中移动。TCP 原点对标夹爪闭合时的指尖中点。delta_position 沿 TCP +X指尖前方、+Y开合方向、+Z上；delta_rpy 是局部 roll/pitch/yaw（弧度，右手规则），位置按当前朝向变换，旋转局部右乘。适合连续自身坐标系操控；每次移动后检查 reached/blocked。", {
        "delta_position": {**_vector(3, "TCP 局部平移增量，米；默认 [0,0,0]"), "default": [0, 0, 0]},
        "delta_rpy": {**_vector(3, "TCP 局部 roll,pitch,yaw 增量，弧度；默认 [0,0,0]"), "default": [0, 0, 0]}, "timeout_s": _TIMEOUT,
    }),
    _schema("robot_pixel_to_base", "将指定相机的深度像素转换为 base_link 下 XYZ。必须提供 camera、pixel 和 depth_m。默认使用服务端相机标定；省略 TCP 位姿时 end 使用当前 TCP 姿态，opposite 和 left 使用固定相机姿态。可提供 tcp_position_base 与 tcp_quaternion_base 指定 TCP 姿态；若使用特定深度帧的标定，则传入该帧 robot_camera_image(type=depth) 返回的完整采集元数据。成功时除坐标外，还会返回同一相机的 RGB，并用洋红色圆圈和十字标出所选 pixel，中心为青色。这两种颜色不用于场景中的红、橙、黄物块、绿色目标或灰白机械臂。该图会发给模型，用于核对选点是否落在目标上。", {
        "camera": {"type": "string", "enum": ["end", "opposite", "left"], "description": "深度来自的相机，必填。left 是机械臂左侧固定相机"},
        "pixel": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2, "description": "深度图像素 [u,v]，原点左上"},
        "depth_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 65.534, "description": "所选像素相机光轴深度，米"},
        "tcp_position_base": _vector(3, "可选：用于计算的闭合指尖中点在 base_link 下位置，米；与四元数成对传入"),
        "tcp_quaternion_base": _vector(4, "可选：用于计算的 TCP 在 base_link 下 wxyz 四元数；与位置成对传入"),
        "camera_position_tcp": _vector(3, "可选：同一深度帧的相机光心在 TCP 下的位置，米；精确帧标定时与其他标定字段一起传入"),
        "camera_rotation_tcp": {"type": "array", "items": _vector(3, "旋转矩阵行"), "minItems": 3, "maxItems": 3},
        "intrinsics": {"type": "object", "properties": {
            "fx": {"type": "number", "exclusiveMinimum": 0}, "fy": {"type": "number", "exclusiveMinimum": 0},
            "cx": {"type": "number"}, "cy": {"type": "number"},
            "width": {"type": "integer", "minimum": 1}, "height": {"type": "integer", "minimum": 1},
        }, "required": ["fx", "fy", "cx", "cy", "width", "height"], "additionalProperties": False},
        "epoch": {"type": "integer", "minimum": 0}, "step": {"type": "integer", "minimum": 0},
    }, ["camera", "pixel", "depth_m"]),
    _schema("robot_get_gripper", "查询夹爪开合状态。"),
    _schema("robot_set_gripper", "控制夹爪，action 与 opening_ratio 必须且只能提供一个。动作成功后额外返回手部相机 end 的当前 RGB，供核对夹爪开合和指尖是否接触物体。", {
        "action": {"type": "string", "enum": ["open", "close"]},
        "opening_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "timeout_s": _TIMEOUT,
    }),
    _schema("robot_camera_image", "获取 RGB 或毫米深度图，默认 rgb。深度返回原始 uint16 PNG、灰度预览和像素距离；预览会发送给模型。detect_objects 仅用于 RGB。", {
        "camera": {"type": "string", "enum": ["end", "opposite", "left"], "description": "end 为手部相机，opposite 为桌对面固定相机，left 为左侧固定相机"},
        "type": {"type": "string", "enum": ["rgb", "depth"], "description": "默认 rgb；depth 为相机光轴方向深度，不是 world Z"},
        "pixels": {"type": "array", "minItems": 1, "maxItems": 64,
                   "items": {"type": "array", "minItems": 2, "maxItems": 2,
                             "items": {"type": "integer", "minimum": 0}},
                   "description": "仅 depth 可用；查询最多64个 [u,v] 像素的原始毫米/米深度。原点左上，u向右，v向下；省略时返回中心像素。"},
    }),
]

_ROUTES = {
    "robot_get_joints": ("GET", "/v1/arm/joints"),
    "robot_set_joints": ("POST", "/v1/arm/joints"),
    "robot_get_pose": ("GET", "/v1/arm/pose"),
    "robot_set_pose": ("POST", "/v1/arm/pose"),
    "robot_move_delta": ("POST", "/v1/arm/pose/delta"),
    "robot_pixel_to_base": ("POST", "/v1/camera/point"),
    "robot_get_gripper": ("GET", "/v1/gripper"),
    "robot_set_gripper": ("POST", "/v1/gripper"),
    "robot_camera_image": ("GET", "/v1/camera/image"),
}
_PARAMETERS = {schema["function"]["name"]: schema["function"]["parameters"] for schema in TOOL_SCHEMAS}
# Use positive field lists so future server additions cannot expose privileged state.
_FIELDS = {
    "joints": ("names", "positions", "velocities", "limits"),
    "pose": ("frame", "position", "quaternion"),
    "gripper": ("opening_ratio", "moving", "status", "raw_position", "target", "velocity"),
}
_META = ("epoch", "step", "sim_time_s", "captured_at_ns")
# Translate known failure codes locally instead of exposing free-form server text.
_ERROR_HELP = {
    "IK_FAILED": ("目标位姿在关节限位内未找到逆运动学解。",
                  "查询当前末端位姿并重新观察；所有绝对位姿使用 base_link 坐标，局部动作使用 TCP 坐标。缩小目标位移；保持朝向时省略 quaternion；不要原样重试。"),
    "INVALID_DEPTH": ("所选像素没有有效深度。", "选择图像中有有效深度的像素后重试。"),
    "INVALID_ARGUMENT": ("服务器拒绝参数。", "检查参数类型、坐标范围、关节限位以及四元数的 wxyz 顺序和单位模长。"),
    "TIMEOUT": ("动作超时，可能已部分执行。", "先查询机器人状态，再决定下一步，不要盲目重试。"),
    "BUSY": ("机器人正在执行其他动作。", "查询状态，等待当前动作完成。"),
    "NOT_READY": ("机器人状态或图像尚未就绪。", "等待后重新获取传感器观测。"),
}


def _error_feedback(body, status):
    error = body.get("error") or {}
    code = error.get("code", "INVALID_ARGUMENT" if status == 422 else "HTTP_ERROR")
    message, hint = _ERROR_HELP.get(code, ("机器人服务请求失败。", "查看服务日志，勿重复发送相同动作。"))
    output = {"code": code, "message": message, "hint": hint, "http_status": status}
    # FastAPI's validation responses use detail instead of the service envelope.
    # Include only known parameter names, never input/ctx/msg.
    fields = []
    for detail in body.get("detail", []) if isinstance(body.get("detail"), list) else []:
        if not isinstance(detail, dict):
            continue
        loc = detail.get("loc", [])
        if isinstance(loc, (list, tuple)):
            fields += [name for name in loc if isinstance(name, str) and name in
                       {"position", "positions", "quaternion", "frame", "timeout_s",
                        "delta_position", "delta_rpy", "camera", "pixel", "depth_m", "tcp_position_base",
                        "tcp_quaternion_base", "camera_position_tcp", "camera_rotation_tcp", "intrinsics", "u", "v", "epoch", "step"}]
    if fields:
        output["fields"] = sorted(set(fields))
    return output



def _select(value, keys):
    return {key: value[key] for key in keys if key in value}


def _validate(value, schema):
    kind = schema["type"]
    if kind in {"number", "integer"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("参数必须是有限数值")
        if kind == "integer" and type(value) is not int:
            raise ValueError("像素坐标必须是整数")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError("数值超出范围")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError("timeout_s 必须大于 0")
    elif kind == "array":
        if not isinstance(value, list) or not schema["minItems"] <= len(value) <= schema["maxItems"]:
            raise ValueError(f"数组长度必须在 {schema['minItems']} 到 {schema['maxItems']} 之间")
        for item in value:
            _validate(item, schema["items"])
    elif kind == "object":
        if not isinstance(value, dict) or value.keys() - schema["properties"].keys():
            raise ValueError("对象包含未允许的参数")
        if set(schema.get("required", [])) - value.keys():
            raise ValueError("对象缺少必要参数")
        for key, item in value.items():
            _validate(item, schema["properties"][key])
    elif kind == "string" and (not isinstance(value, str) or value not in schema["enum"]):
        raise ValueError("字符串参数不在允许的取值中")


class MujocoRobotTool:
    schemas = TOOL_SCHEMAS

    def __init__(self, server="http://127.0.0.1:8000", timeout_s=20):
        url = urlsplit(server)
        if url.scheme not in {"http", "https"} or not url.hostname or url.query or url.fragment:
            raise ValueError("mujoco.server 必须是 HTTP(S) 服务地址")
        _validate(timeout_s, _TIMEOUT)
        self.server = server.rstrip("/")
        self.timeout_s = timeout_s
        self._images = None
        self._rejected_pose = None

    def execute(self, name, arguments):
        if name not in _ROUTES:
            raise ValueError(f"未知或禁止的机器人接口: {name}")
        schema = _PARAMETERS[name]
        if not isinstance(arguments, dict) or arguments.keys() - schema["properties"].keys():
            raise ValueError("包含未允许的工具参数")
        if set(schema["required"]) - arguments.keys():
            raise ValueError("缺少必要的工具参数")
        for key, value in arguments.items():
            _validate(value, schema["properties"][key])
        if name == "robot_set_gripper" and (("action" in arguments) == ("opening_ratio" in arguments)):
            raise ValueError("action 与 opening_ratio 必须且只能提供一个")
        if name == "robot_set_pose" and "quaternion" in arguments:
            norm = math.hypot(*arguments["quaternion"])
            if abs(norm - 1.0) > 1e-3:
                raise ValueError("quaternion 必须是模长接近 1 的单位四元数，顺序 wxyz；保持当前朝向时请省略 quaternion")
        pose_key = None
        if name == "robot_set_pose":
            pose_key = (tuple(arguments["position"]),
                        tuple(arguments["quaternion"]) if "quaternion" in arguments else None,
                        arguments.get("frame", "base"))
            if self._rejected_pose is not None and self._rejected_pose[0] == pose_key:
                return {"ok": False, "error": {
                    "code": "POSE_ALREADY_REJECTED", "previous_code": self._rejected_pose[1],
                    "message": "该位姿已被服务器拒绝，机器人未成功移动前不再原样提交。",
                    "hint": "重新查询位姿和相机，检查坐标系、位置与朝向并修改目标；改变 timeout_s 不能修复无解位姿。",
                }}
        if name == "robot_camera_image" and "pixels" in arguments and arguments.get("type", "rgb") != "depth":
            raise ValueError("pixels 参数仅用于 type=depth")
        if name == "robot_pixel_to_base":
            has_position = "tcp_position_base" in arguments
            has_quaternion = "tcp_quaternion_base" in arguments
            if has_position != has_quaternion:
                raise ValueError("tcp_position_base 与 tcp_quaternion_base 必须成对提供")
            calibration = {"camera_position_tcp", "camera_rotation_tcp", "intrinsics"}
            present = calibration.intersection(arguments)
            if present and present != calibration:
                raise ValueError("camera_position_tcp、camera_rotation_tcp 与 intrinsics 必须完整提供，或全部省略")
            if present and not has_position:
                raise ValueError("完整采集标定必须同时提供采集时 TCP 位姿")
        method, path = _ROUTES[name]
        payload = dict(arguments)
        if name == "robot_camera_image":
            payload.setdefault("type", "rgb")
            payload.pop("pixels", None)
        elif name == "robot_get_pose":
            payload.setdefault("frame", "base")
        elif name == "robot_set_pose":
            payload.setdefault("frame", "base")
        # No redirects or automatic retries: a timed-out motion may still have executed.
        response = httpx.request(
            method, self.server + path,
            **({"params": payload} if method == "GET" else {"json": payload}),
            timeout=max(self.timeout_s, arguments.get("timeout_s", 0) + 5),
            follow_redirects=False,
        )
        if name == "robot_pixel_to_base" and response.is_success:
            body = response.json()
            data = body.get("data") or {}
            output = {"ok": bool(body.get("ok")) and response.is_success,
                      "data": _select(data, ("frame", "position", "camera", "pixel", "depth_m", "epoch", "step"))}
            if output["ok"]:
                output.update(self._marked_rgb(arguments["camera"], arguments["pixel"]))
            return output
        if name == "robot_camera_image" and response.is_success:
            if not response.headers.get("content-type", "").startswith("image/png") or not response.content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("相机未返回 PNG 图片")
            path = self._save_png(response.content)
            output = {"image_path": str(path.resolve()), "camera": payload.get("camera", "end"),
                      "type": payload["type"]}
            # Expose only frame metadata, never arbitrary response headers.
            for key in ("epoch", "step", "captured_at_ns", "width", "height"):
                value = response.headers.get("x-" + key.replace("_", "-"))
                if value is not None:
                    output[key] = int(value)
            for header, field in (("x-tcp-position-base", "tcp_position_base"),
                                  ("x-tcp-quaternion-base", "tcp_quaternion_base"),
                                  ("x-camera-position-tcp", "camera_position_tcp"),
                                  ("x-camera-rotation-tcp", "camera_rotation_tcp"),
                                  ("x-intrinsics", "intrinsics")):
                value = response.headers.get(header)
                if value is not None:
                    output[field] = json.loads(base64.urlsafe_b64decode(value.encode()).decode())
            if payload["type"] == "depth":
                output.update(self._depth_observation(response.content, arguments.get("pixels"), path))
            return output
        # Never include raw HTTP bodies or server error details (which may contain snapshots).
        body = response.json()
        data = body.get("data") or {}
        output = {"ok": bool(body.get("ok")) and response.is_success}
        if not output["ok"]:
            output["error"] = _error_feedback(body, response.status_code)
            if pose_key is not None and response.status_code == 422:
                self._rejected_pose = (pose_key, output["error"]["code"])
        elif name in {"robot_set_pose", "robot_set_joints", "robot_move_delta"}:
            self._rejected_pose = None
        if "result" in data:
            output["result"] = _select(data["result"], ("ok", "status", "code", "error", "velocity", "operation_id"))
            state = data.get("state") or {}
            output["state"] = _select(state, _META)
            for kind, fields in _FIELDS.items():
                if kind in state:
                    output["state"][kind] = _select(state[kind], fields)
        else:
            kind = "joints" if name.endswith("joints") else "pose" if name.endswith("pose") or name == "robot_move_delta" else "gripper"
            output["data"] = _select(data, _FIELDS[kind] + _META)
        if name == "robot_set_gripper" and output["ok"]:
            output.update(self._camera_rgb("end"))
        return output

    def _save_png(self, content):
        if self._images is None:
            self._images = tempfile.TemporaryDirectory(prefix="vllm-robot-")
        with tempfile.NamedTemporaryFile(dir=self._images.name, suffix=".png", delete=False) as image:
            image.write(content)
        return Path(image.name)

    def _fetch_rgb(self, camera):
        response = httpx.request(
            "GET", self.server + "/v1/camera/image", params={"camera": camera, "type": "rgb"},
            timeout=self.timeout_s, follow_redirects=False,
        )
        if not response.is_success or not response.headers.get("content-type", "").startswith("image/png") or not response.content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("动作已成功，但未能获取用于核对的 RGB")
        return response.content

    def _camera_rgb(self, camera):
        """Attach a current RGB frame from one camera."""
        path = self._save_png(self._fetch_rgb(camera))
        return {"image_path": str(path.resolve()), "camera": camera, "type": "rgb"}

    def _marked_rgb(self, camera, pixel):
        """Attach a current RGB of the same camera with the selected pixel marked."""
        with Image.open(io.BytesIO(self._fetch_rgb(camera))) as image:
            if image.format != "PNG" or image.mode not in {"RGB", "RGBA"}:
                raise ValueError("选点核对图必须是 RGB PNG")
            rgb = image.convert("RGB")
        u, v = pixel
        width, height = rgb.size
        if u >= width or v >= height:
            raise ValueError(f"像素 [{u},{v}] 超出核对图 {width}×{height} 范围")
        # Magenta and cyan are absent from the blocks, target and arm, so the mark stays readable.
        mark = rgb.load()
        radius = max(14, min(width, height) // 7)
        thickness = max(3, min(width, height) // 48)

        def paint(x, y, color):
            if 0 <= x < width and 0 <= y < height:
                mark[x, y] = color

        for offset in range(-radius, radius + 1):
            for du, dv in ((offset, 0), (0, offset)):
                for width_offset in range(-thickness // 2, thickness // 2 + 1):
                    paint(u + du + (width_offset if dv else 0), v + dv + (width_offset if du else 0), (255, 0, 255))
        for y in range(max(0, v - radius), min(height, v + radius + 1)):
            for x in range(max(0, u - radius), min(width, u + radius + 1)):
                distance = math.hypot(x - u, y - v)
                if radius - thickness <= distance <= radius:
                    paint(x, y, (255, 0, 255))
        for offset in range(-thickness, thickness + 1):
            for du, dv in ((offset, 0), (0, offset)):
                paint(u + du, v + dv, (0, 255, 255))
        path = self._save_png(b"")
        rgb.save(path)
        return {
            "marked_image_path": str(path.resolve()), "marked_camera": camera, "marked_pixel": [u, v],
            "marked_image": {
                "width": width, "height": height,
                "legend": "洋红色圆圈和十字标出选点，青色小块是所选 pixel。这些颜色不属于场景物体。请核对标记是否落在目标物体上；偏离则重新选点。",
            },
        }

    def _depth_observation(self, content, pixels, path):
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG" or image.mode not in {"I", "I;16", "I;16L", "I;16B"}:
                raise ValueError("深度图必须是单通道 uint16 PNG，单位毫米")
            width, height = image.size
            pixels_data = image.load()
            values = [pixels_data[u, v] for v in range(height) for u in range(width)]
        requested = pixels if pixels is not None else [[width // 2, height // 2]]
        samples = []
        for u, v in requested:
            if u >= width or v >= height:
                raise ValueError(f"像素 [{u},{v}] 超出深度图 {width}×{height} 范围")
            depth = values[v * width + u]
            valid = 0 < depth < 65535
            samples.append({"pixel": [u, v], "valid": valid, "depth_mm": depth if valid else None,
                            "depth_m": depth / 1000 if valid else None})
        valid_values = [value for value in values if 0 < value < 65535]
        near = min(valid_values) if valid_values else None
        far = max(valid_values) if valid_values else None
        # Reserve black for invalid/saturated pixels. Brighter means farther away.
        levels = [0 if not 0 < value < 65535 else
                  128 if near == far else 1 + round((value - near) * 254 / (far - near))
                  for value in values]
        preview_path = path.with_name(path.stem + "-preview.png")
        preview = Image.new("L", (width, height))
        preview.putdata(levels)
        preview.save(preview_path)
        return {
            "width": width, "height": height, "encoding": "uint16_png", "unit": "mm",
            "depth_definition": "相机光轴方向深度；不是世界坐标高度或物体三维坐标。valid 仅表示数值未无效/饱和，背景可能返回远裁剪面距离，不代表命中目标物体。",
            "samples": samples, "preview_path": str(preview_path.resolve()),
            "preview": {"near_mm": near, "far_mm": far,
                        "legend": "黑色0=无效或饱和；灰度1近、255远，按本帧有效范围线性缩放；等深度时灰度128。精确距离请使用 samples。"},
            "invalid_values_mm": [0, 65535],
        }

    def close(self):
        if self._images is not None:
            self._images.cleanup()
            self._images = None
