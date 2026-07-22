"""
LLM client wrapper
Uses OpenAI-compatible API format
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from ..utils.locale import t
from .openai_chat_compat import create_chat_completion, extract_chat_completion_text


class LLMClient:
    """LLM client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError(t("api.llmApiKeyMissing"))

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 8192,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        Send a chat request

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            response_format: Response format (e.g. JSON mode)

        Returns:
            Model response text
        """
        response = create_chat_completion(
            self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        content = extract_chat_completion_text(response)
        # Some models (e.g. MiniMax M2.5) include thinking content in the content; remove it
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # Clean up markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(
            r"^```(?:json)?\s*\n?", "", cleaned_response, flags=re.IGNORECASE
        )
        cleaned_response = re.sub(r"\n?```\s*$", "", cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(t("api.llmJsonInvalid", response=cleaned_response))