"""
Backward-compatible re-export.

New code should use `model_manager.agent.factory.create_client` directly.
`AnthropicClient` is kept as an alias for `AnthropicProvider`.
"""

from model_manager.agent.base import LLMClient
from model_manager.agent.factory import LLMProvider, create_client, create_client_from_settings
from model_manager.agent.providers.anthropic_provider import AnthropicProvider as AnthropicClient

__all__ = [
    "LLMClient",
    "LLMProvider",
    "AnthropicClient",
    "create_client",
    "create_client_from_settings",
]
