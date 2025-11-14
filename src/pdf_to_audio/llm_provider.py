"""Unified LLM provider abstraction using any-llm library."""

import os
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    from any_llm import LLM
    from any_llm.models import ChatMessage, ChatResponse
except ImportError:
    raise ImportError(
        "any-llm is required for this module. "
        "Install it with: pip install any-llm"
    )


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: str  # e.g., "openai", "anthropic", "mistral", "ollama"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4000


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Send a chat completion request to the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            The assistant's response text
        """
        pass


class AnyLLMProvider(LLMProvider):
    """LLM provider using the any-llm library."""

    def __init__(self, config: LLMConfig):
        """
        Initialize the any-llm provider.

        Args:
            config: LLMConfig with provider details
        """
        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the any-llm client based on config."""
        # Build the model string for any-llm
        # Format: "provider/model"
        model_str = f"{self.config.provider}/{self.config.model}"

        # Get API key from config or environment
        api_key = self.config.api_key
        if not api_key:
            # Try to get from environment based on provider
            env_keys = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "mistral": "MISTRAL_API_KEY",
            }
            env_key = env_keys.get(self.config.provider.lower())
            if env_key:
                api_key = os.getenv(env_key)

        # Initialize the any-llm client
        client_kwargs = {
            "model": model_str,
            "temperature": self.config.temperature,
        }

        if api_key:
            client_kwargs["api_key"] = api_key

        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        try:
            self.client = LLM(**client_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize any-llm client for {self.config.provider}/{self.config.model}: {e}"
            )

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Send a chat completion request using any-llm.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            The assistant's response text
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized")

        # Convert messages to any-llm format if needed
        # any-llm accepts messages in a compatible format
        try:
            response = self.client.chat(
                messages=messages,
                temperature=temperature or self.config.temperature,
            )

            # Extract the text response
            if isinstance(response, str):
                return response
            elif hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'content' in response:
                return response['content']
            else:
                return str(response)
        except Exception as e:
            raise RuntimeError(f"Error calling LLM: {e}")


def create_llm_provider(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider: Provider name (e.g., "mistral", "openai", "anthropic")
        model: Model name
        api_key: Optional API key (will use environment variable if not provided)
        base_url: Optional base URL for the API
        temperature: Temperature setting (default: 0.2)
        max_tokens: Max tokens setting (default: 4000)

    Returns:
        An LLMProvider instance
    """
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return AnyLLMProvider(config)
