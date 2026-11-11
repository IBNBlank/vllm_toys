"""Reusable client for services implementing OpenAI Chat Completions."""

import os
import time

from openai import OpenAI

from .config import load_config, validate_config


def reasoning_text(delta):
    """Normalize reasoning fields used by OpenAI-compatible services."""
    for field in ("reasoning_content", "reasoning", "thinking"):
        value = getattr(delta, field, None)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list):
            text = "".join(
                part.get("text") or part.get("thinking") or ""
                for part in value if isinstance(part, dict)
            )
            if text:
                return text
    return ""


class ChatClient:
    """Own one SDK connection pool; callers own conversation history and output."""

    def __init__(self, config):
        config = validate_config(config)
        self.model = config["model"]
        self.max_tokens = config.get("max_tokens", 128)
        self.stream_include_usage = config.get("stream_include_usage", True)
        api_key = config.get("api_key")
        if config.get("api_key_env"):
            api_key = os.environ.get(config["api_key_env"])
            if not api_key or not api_key.strip():
                raise ValueError(f"环境变量 {config['api_key_env']} 未设置或为空")
        self._sdk = OpenAI(
            base_url=config["base_url"],
            api_key=api_key or "not-needed",
            timeout=config.get("timeout", 120),
            max_retries=0,
        )

    @classmethod
    def from_config(cls, path):
        """Construct a client from a JSON configuration file."""
        return cls(load_config(path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Release HTTP connections. Prefer using the client as a context manager."""
        self._sdk.close()

    def complete(self, messages, stream=False, on_text=None, tools=None, on_reasoning=None):
        """Return (assistant message, stats), optionally sending chunks to a callback."""
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        if stream and self.stream_include_usage:
            params["stream_options"] = {"include_usage": True}
        first_text = usage = finish_reason = None
        start = time.perf_counter()
        response = self._sdk.chat.completions.create(**params)
        if stream:
            parts, reasoning_parts = [], []
            tool_calls = {}
            with response:
                for chunk in response:
                    if chunk.usage is not None:
                        usage = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.index != 0:
                            continue
                        text = choice.delta.content or ""
                        reasoning = reasoning_text(choice.delta)
                        if text or reasoning:
                            if first_text is None:
                                first_text = time.perf_counter() - start
                            parts.append(text)
                            reasoning_parts.append(reasoning)
                            if reasoning and on_reasoning:
                                on_reasoning(reasoning)
                            if on_text:
                                output = text if on_reasoning else reasoning + text
                                if output:
                                    on_text(output)
                        for delta in choice.delta.tool_calls or []:
                            call = tool_calls.setdefault(delta.index, {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                            if delta.id:
                                call["id"] += delta.id
                            if delta.type:
                                call["type"] = delta.type
                            if delta.function:
                                call["function"]["name"] += delta.function.name or ""
                                call["function"]["arguments"] += delta.function.arguments or ""
                        if choice.finish_reason is not None:
                            finish_reason = choice.finish_reason
            if finish_reason is None:
                raise ValueError("流式响应意外中断，未收到 finish_reason")
            message = {"role": "assistant", "content": "".join(parts)}
            if tool_calls:
                message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
                if not message["content"]:
                    message["content"] = None
                for call in message["tool_calls"]:
                    if not call["id"] or not call["function"]["name"]:
                        raise ValueError("流式工具调用缺少 id 或函数名")
            if any(reasoning_parts):
                message["reasoning_content"] = "".join(reasoning_parts)
        else:
            if not response.choices or response.choices[0].message is None:
                raise ValueError("服务端响应缺少 choices[0].message")
            message = response.choices[0].message.model_dump(exclude_none=True)
            usage = response.usage.model_dump() if response.usage else None
            finish_reason = response.choices[0].finish_reason
        return message, {
            "elapsed": time.perf_counter() - start,
            "first_text": first_text,
            "usage": usage,
            "finish_reason": finish_reason,
        }
