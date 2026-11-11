"""YOLO26s image detection, usable as a Python tool or a JSON CLI."""

import argparse
import json
import math
import sys
from pathlib import Path


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_objects",
        "description": "使用 YOLO26s 检测本地图片，返回类别、置信度和像素坐标 xyxy。只接受本地图片文件路径，不接受 URL。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "本地图片文件路径，相对当前工作目录"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "置信度阈值，省略时使用本地工具配置（默认 0.25）"},
            },
            "required": ["image_path"],
            "additionalProperties": False,
        },
    },
}


_DEFAULT_CONFIDENCE = object()


class YOLO26Tool:
    """Lazily load and reuse weights; model and device are controlled by the host."""

    def __init__(self, model="yolo26s.pt", device="0", confidence=0.25):
        self.model_path = str(model)
        self.device = device
        self.confidence = confidence
        self._model = None

    def detect(self, image_path, confidence=_DEFAULT_CONFIDENCE):
        if confidence is _DEFAULT_CONFIDENCE:
            confidence = self.confidence
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("image_path 必须是非空本地图片路径")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence 必须是 0 到 1 之间的数值")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence 必须是 0 到 1 之间的数值")
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"本地图片不存在: {path}")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
            raise ValueError("仅支持本地图片文件，不支持视频、目录或 URL")
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError("请先运行 uv sync --extra yolo 安装 YOLO 依赖") from error
            self._model = YOLO(self.model_path, task="detect")
        result = self._model.predict(
            source=str(path), conf=confidence, device=self.device,
            verbose=False, save=False,
        )[0]
        detections = []
        if result.boxes is not None:
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                class_id = int(class_id)
                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(score),
                    "bbox_xyxy": [float(value) for value in box],
                })
        height, width = result.orig_shape
        return {
            "image_path": str(path), "model": self.model_path,
            "width": int(width), "height": int(height),
            "detections": detections,
        }


def main():
    parser = argparse.ArgumentParser(description="使用 YOLO26s 检测本地图片，向 stdout 输出 JSON")
    parser.add_argument("--image", required=True, help="本地图片路径")
    parser.add_argument("--model", default="yolo26s.pt", help="权重名称或本地路径")
    parser.add_argument("--device", default="0", help="推理设备，默认 GPU 0，可设为 cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    # Ultralytics may print weight download/setup messages; keep stdout JSON-only.
    from contextlib import redirect_stdout
    try:
        with redirect_stdout(sys.stderr):
            result = YOLO26Tool(args.model, args.device).detect(args.image, args.confidence)
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
