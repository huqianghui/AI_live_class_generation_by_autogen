"""
Azure OpenAI Round-Robin Client implementation.

This module provides a round-robin client for Azure OpenAI that distributes requests
across multiple endpoints to balance load and prevent rate limits.
"""

from .azureOpenAIClientRoundRobin import (
    AzureOpenAIRoundRobinClient,
    AzureOpenAIClientsRoundRobin,
    ClientConfig,
    client_manager,
    initialize_client_manager_from_env,
)

__all__ = [
    "AzureOpenAIRoundRobinClient",
    "AzureOpenAIClientsRoundRobin",
    "ClientConfig",
    "client_manager",
    "initialize_client_manager_from_env",
]