"""
Example usage of the Azure OpenAI Round-Robin client.

This example demonstrates how to:
1. Set up environment variables for multiple endpoints
2. Initialize the client manager
3. Create and use the AzureOpenAIRoundRobinClient
"""

import asyncio
import json
import os
from typing import List, Optional

from autogen_core import CancellationToken
from autogen_core.models import LLMMessage, SystemMessage, UserMessage

from roundRobin import (
    AzureOpenAIRoundRobinClient,
    initialize_client_manager_from_env,
)

# Example environment variable that would be set in your .env file or system environment
# AZURE_OPENAI_ROUND_ROBIN_CONNECTION='[
#   {
#     "AZURE_OPENAI_ENDPOINT": "https://endpoint1.openai.azure.com/",
#     "AZURE_OPENAI_API_KEY": "your-api-key-1"
#   },
#   {
#     "AZURE_OPENAI_ENDPOINT": "https://endpoint2.openai.azure.com/",
#     "AZURE_OPENAI_API_KEY": "your-api-key-2"
#   }
# ]'

async def setup_example():
    # For testing, if the environment variable is not set, you can set it programmatically
    # In production, you would set this in your environment or .env file
    if not os.environ.get("AZURE_OPENAI_ROUND_ROBIN_CONNECTION"):
        # This is just for example purposes - replace with your actual endpoints or
        # preferably set the environment variable externally
        sample_connections = [
            {
                "AZURE_OPENAI_ENDPOINT": "https://endpoint1.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "your-api-key-1"
            },
            {
                "AZURE_OPENAI_ENDPOINT": "https://endpoint2.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "your-api-key-2"
            }
        ]
        os.environ["AZURE_OPENAI_ROUND_ROBIN_CONNECTION"] = json.dumps(sample_connections)
    
    # Base configuration that will be used for all clients
    base_config = {
        "model": "gpt-4o",  # The deployment name in Azure OpenAI
        "api_version": "2023-07-01-preview",
        "temperature": 0.7,
        "max_tokens": 800,
    }
    
    # Initialize the client manager with configurations from env
    await initialize_client_manager_from_env(base_config)
    
    # Create the round-robin client
    # Note: The azure_endpoint and api_key provided here will be overridden by the round-robin manager
    # We provide placeholder values just to satisfy the constructor
    client = AzureOpenAIRoundRobinClient(
        model=base_config["model"],
        api_version=base_config["api_version"],
        temperature=base_config["temperature"],
        max_tokens=base_config["max_tokens"],
        azure_endpoint="https://placeholder.openai.azure.com/",  # Will be overridden
        api_key="placeholder-api-key"  # Will be overridden
    )
    
    return client

async def run_example():
    # Set up the client
    client = await setup_example()
    
    # Create some messages
    messages: List[LLMMessage] = [
        SystemMessage(content="You are a helpful AI assistant."),
        UserMessage(content="Tell me a short joke about programming.")
    ]
    
    # Make multiple calls to demonstrate round-robin behavior
    for i in range(3):
        print(f"\nRequest {i+1}:")
        result = await client.create(messages, cancellation_token=CancellationToken())
        print(f"Response: {result.content}")
    
    # Close the client when finished
    await client.close()

if __name__ == "__main__":
    asyncio.run(run_example())