# Azure OpenAI Round-Robin Client

This module provides a round-robin client implementation for Azure OpenAI that distributes requests across multiple endpoints to balance load and prevent rate limits.

## Features

- Distributes requests across multiple Azure OpenAI endpoints in round-robin fashion
- Thread-safe implementation for concurrent use
- Compatible with the standard `AzureOpenAIChatCompletionClient` API
- Maintains all the same methods as the original client
- Can be enabled/disabled via environment variables

## Usage

### Setting up your environment

1. Define your multiple Azure OpenAI endpoints in an environment variable:

```bash
# .env file or exported in your shell
export AZURE_OPENAI_ROUND_ROBIN_CONNECTION='[
  {
    "AZURE_OPENAI_ENDPOINT": "https://endpoint1.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "your-api-key-1"
  },
  {
    "AZURE_OPENAI_ENDPOINT": "https://endpoint2.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "your-api-key-2"
  },
  {
    "AZURE_OPENAI_ENDPOINT": "https://endpoint3.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "your-api-key-3"
  }
]'
```

2. Enable the round-robin client by setting:

```bash
export USE_AZURE_OPENAI_ROUND_ROBIN="true"
```

### Using the round-robin client directly

```python
import asyncio
from roundRobin import AzureOpenAIRoundRobinClient, initialize_client_manager_from_env

async def main():
    # Initialize the client manager first
    base_config = {
        "model": "gpt-4",
        "api_version": "2023-07-01-preview",
        "temperature": 0.7,
        "max_tokens": 800,
    }
    await initialize_client_manager_from_env(base_config)
    
    # Create the round-robin client (endpoints will be managed internally)
    client = AzureOpenAIRoundRobinClient(
        model=base_config["model"],
        api_version=base_config["api_version"],
        temperature=base_config["temperature"],
        max_tokens=base_config["max_tokens"],
        azure_endpoint="placeholder",  # Will be overridden by round-robin manager
        api_key="placeholder"          # Will be overridden by round-robin manager
    )
    
    # Use the client exactly as you would use the standard client
    messages = [...]
    result = await client.create(messages)
    
    # Close the client when done
    await client.close()

asyncio.run(main())
```

### Using the round-robin client with your application

In your application's configuration, the round-robin client will be used automatically when you set `USE_AZURE_OPENAI_ROUND_ROBIN=true` in your environment.

The configuration functions like `get_model_client()` and `get_advance_model_client()` will return the round-robin version when enabled, with no changes required to your application code.

## Benefits of Round-Robin Load Balancing

1. **Higher Throughput**: Distribute requests across multiple endpoints to increase your total throughput.
2. **Rate Limit Protection**: Avoid hitting rate limits on a single endpoint.  
3. **Improved Reliability**: If one endpoint is experiencing issues, other endpoints can still serve requests.
4. **Cost Management**: Spread usage across multiple subscriptions or regions.

## Implementation Details

The round-robin implementation consists of:

- `AzureOpenAIClientsRoundRobin`: A manager class that maintains a pool of clients and rotates through them.
- `AzureOpenAIRoundRobinClient`: A subclass of `AzureOpenAIChatCompletionClient` that delegates calls to the next client in the rotation.

## Azure Best Practices

- Use endpoints in different regions for better geographical redundancy
- Monitor usage and adjust the number of endpoints based on your needs
- Implement proper error handling for failed requests
- Consider setting up different deployment capacities for different endpoints