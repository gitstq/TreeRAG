"""
多LLM后端适配客户端模块

提供统一的LLM调用接口，支持OpenAI、Claude（Anthropic）和Ollama三种后端。
内置重试机制和错误处理，确保调用的可靠性。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """多后端LLM客户端。

    支持三种LLM后端：
    - openai: OpenAI API（也兼容任何OpenAI格式的API端点）
    - claude: Anthropic Claude API（通过httpx直接调用）
    - ollama: Ollama本地模型服务

    Usage:
        >>> client = LLMClient(backend="ollama", model="llama3")
        >>> response = client.generate("请总结以下内容：...")
        >>> print(response)
    """

    def __init__(
        self,
        backend: str = "ollama",
        api_key: str = "",
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """初始化LLM客户端。

        Args:
            backend: LLM后端类型（openai/claude/ollama）
            api_key: API密钥
            model: 模型名称
            base_url: API基础URL
            temperature: 生成温度
            max_tokens: 最大生成token数
            retry_count: 重试次数
            retry_delay: 重试间隔（秒）
        """
        self.backend = backend.lower()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # 验证后端类型
        if self.backend not in ("openai", "claude", "ollama"):
            raise ValueError(
                f"不支持的LLM后端: {self.backend}，"
                f"请选择 openai、claude 或 ollama"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """统一的文本生成接口。

        根据配置的后端类型，自动路由到对应的API调用方法。
        内置重试机制，失败时自动重试。

        Args:
            prompt: 用户提示文本
            system_prompt: 系统提示文本（可选）
            temperature: 生成温度（可选，覆盖默认值）
            max_tokens: 最大生成token数（可选，覆盖默认值）

        Returns:
            LLM生成的文本内容

        Raises:
            RuntimeError: 所有重试均失败后抛出
        """
        actual_temp = temperature if temperature is not None else self.temperature
        actual_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_count + 1):
            try:
                if self.backend == "openai":
                    return self._call_openai(
                        prompt, system_prompt, actual_temp, actual_max_tokens
                    )
                elif self.backend == "claude":
                    return self._call_claude(
                        prompt, system_prompt, actual_temp, actual_max_tokens
                    )
                elif self.backend == "ollama":
                    return self._call_ollama(
                        prompt, system_prompt, actual_temp, actual_max_tokens
                    )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM调用失败（第{attempt}/{self.retry_count}次）: {e}"
                )
                if attempt < self.retry_count:
                    # 指数退避
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM调用失败，已重试{self.retry_count}次。"
            f"最后错误: {last_error}"
        ) from last_error

    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """调用OpenAI兼容API。

        支持OpenAI官方API以及任何兼容OpenAI格式的API端点（如Ollama的OpenAI兼容端点）。

        Args:
            prompt: 用户提示文本
            system_prompt: 系统提示文本
            temperature: 生成温度
            max_tokens: 最大生成token数

        Returns:
            生成的文本内容
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "使用OpenAI后端需要安装httpx库。"
                "请运行: pip install httpx"
            )

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=120.0,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_claude(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """调用Anthropic Claude API。

        使用httpx直接调用Anthropic的Messages API。

        Args:
            prompt: 用户提示文本
            system_prompt: 系统提示文本
            temperature: 生成温度
            max_tokens: 最大生成token数

        Returns:
            生成的文本内容
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "使用Claude后端需要安装httpx库。"
                "请运行: pip install httpx"
            )

        if not self.api_key:
            raise ValueError("使用Claude后端必须提供API密钥（api_key）")

        url = f"{self.base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=120.0,
        )
        response.raise_for_status()

        data = response.json()
        # Claude API返回的content是一个列表
        content_blocks = data.get("content", [])
        text_parts = [
            block["text"] for block in content_blocks
            if block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()

    def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """调用Ollama本地API。

        通过Ollama的REST API调用本地部署的模型。

        Args:
            prompt: 用户提示文本
            system_prompt: 系统提示文本
            temperature: 生成温度
            max_tokens: 最大生成token数

        Returns:
            生成的文本内容
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "使用Ollama后端需要安装httpx库。"
                "请运行: pip install httpx"
            )

        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = httpx.post(
            url,
            json=payload,
            timeout=300.0,  # 本地模型可能需要更长时间
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """生成JSON格式的响应。

        在提示中要求LLM返回JSON格式，并解析响应。

        Args:
            prompt: 用户提示文本
            system_prompt: 系统提示文本
            temperature: 生成温度

        Returns:
            解析后的JSON字典

        Raises:
            json.JSONDecodeError: LLM返回的内容无法解析为JSON
        """
        json_system = "你必须只返回有效的JSON格式数据，不要包含任何其他文字说明。"
        if system_prompt:
            json_system = f"{system_prompt}\n{json_system}"
        else:
            json_system = json_system

        response = self.generate(prompt, system_prompt=json_system, temperature=temperature)

        # 尝试提取JSON内容（处理可能的markdown代码块包裹）
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        return json.loads(cleaned)

    def __repr__(self) -> str:
        """返回客户端的字符串表示。"""
        return (
            f"LLMClient(backend={self.backend!r}, model={self.model!r}, "
            f"base_url={self.base_url!r})"
        )
