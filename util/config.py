"""Load and validate settings for OpenAI-compatible chat services."""

import json
from pathlib import Path


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("配置必须是 JSON 对象")
    for key in ("base_url", "model"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"配置缺少有效的 {key}")
    if not config["base_url"].startswith(("http://", "https://")):
        raise ValueError("base_url 必须使用 http:// 或 https://")
    max_tokens = config.get("max_tokens", 128)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens 必须是正整数")
    timeout = config.get("timeout", 120)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout 必须是正数")
    for key in ("api_key", "api_key_env"):
        if key in config and not isinstance(config[key], str):
            raise ValueError(f"{key} 必须是字符串，不使用时留空")
    if type(config.get("stream_include_usage", True)) is not bool:
        raise ValueError("stream_include_usage 必须是布尔值")
    if "reasoning_effort" in config:
        effort = config["reasoning_effort"]
        if not isinstance(effort, str) or not effort.strip():
            raise ValueError("reasoning_effort 必须是非空字符串，例如 low；可用值取决于服务端")
    if "extra_body" in config and not isinstance(config["extra_body"], dict):
        raise ValueError("extra_body 必须是 JSON 对象")
    return dict(config)


def load_config(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return validate_config(json.load(handle))
