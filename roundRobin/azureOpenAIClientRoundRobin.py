import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union

from autogen_core import CancellationToken
from autogen_core.models import CreateResult, LLMMessage
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    AzureOpenAIClientConfigurationConfigModel,
)
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("azure_openai_round_robin")

class ClientConfig(BaseModel):
    """Configuration model for an Azure OpenAI client"""
    azure_endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="API key for the Azure OpenAI endpoint")
    additional_config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration parameters")

class AzureOpenAIClientsRoundRobin:
    """
    Manages multiple Azure OpenAI clients and provides round-robin access to them.
    
    This class maintains a pool of initialized Azure OpenAI clients and rotates through them
    for each request, helping to distribute load and avoid rate limit issues.
    """
    
    def __init__(self):
        self._clients: List[AzureOpenAIChatCompletionClient] = []
        self._current_index = 0
        self._lock = asyncio.Lock()
        self._base_config: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize(self, base_config: Dict[str, Any], connection_configs: List[ClientConfig]):
        """
        Initialize the round-robin client manager with multiple client configurations.
        
        Args:
            base_config: The base configuration shared by all clients (model, deployment, etc)
            connection_configs: List of client-specific configurations (endpoints, api keys)
        """
        async with self._lock:
            if self._initialized:
                logger.warning("AzureOpenAIClientsRoundRobin already initialized")
                return
                
            self._base_config = base_config
            
            # Create all clients
            for config in connection_configs:
                # Merge base config with client-specific config
                client_config = {**base_config, **{"azure_endpoint": config.azure_endpoint, "api_key": config.api_key}}
                client_config.update(config.additional_config)
                
                # Create and initialize the client
                client = AzureOpenAIChatCompletionClient(**client_config)
                self._clients.append(client)
                
            if not self._clients:
                raise ValueError("No client configurations provided")
                
            self._initialized = True
            logger.info(f"Initialized AzureOpenAIClientsRoundRobin with {len(self._clients)} clients")
    
    @property
    def client_count(self) -> int:
        """Return the number of clients in the pool."""
        return len(self._clients)
    
    @property
    def initialized(self) -> bool:
        """Return whether the client manager has been initialized."""
        return self._initialized
    
    async def get_next_client(self) -> AzureOpenAIChatCompletionClient:
        """
        Get the next client in the round-robin rotation.
        
        This method is thread-safe and will rotate through available clients.
        
        Returns:
            The next AzureOpenAIChatCompletionClient in the rotation
        
        Raises:
            ValueError: If no clients are available
        """
        if not self._initialized:
            raise ValueError("AzureOpenAIClientsRoundRobin not initialized")
            
        if not self._clients:
            raise ValueError("No clients available")
        
        async with self._lock:
            # Get the next client
            client = self._clients[self._current_index]
            
            # Update the index for the next call
            self._current_index = (self._current_index + 1) % len(self._clients)
            
            return client

    def get_base_config(self) -> Dict[str, Any]:
        """Return the base configuration shared by all clients."""
        return self._base_config.copy()

# Create a singleton instance of the client manager
client_manager = AzureOpenAIClientsRoundRobin()

# Helper function to initialize the client manager from environment variables
async def initialize_client_manager_from_env(
    base_config: Dict[str, Any],
    connection_env_var: str = "AZURE_OPENAI_ROUND_ROBIN_CONNECTION"
) -> AzureOpenAIClientsRoundRobin:
    """
    Initialize the client manager from environment variables.
    
    Args:
        base_config: Base configuration for all clients (model, deployment, etc.)
        connection_env_var: Environment variable containing JSON array of connection configs
        
    Returns:
        The initialized client manager
    """
    # Get connection configurations from environment variable
    connections_str = os.environ.get(connection_env_var)
    if not connections_str:
        raise ValueError(
            f"Environment variable {connection_env_var} not set. "
            f"This should contain a JSON array of connection configurations."
        )
    
    try:
        connections_data = json.loads(connections_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {connection_env_var}: {str(e)}")
    
    if not isinstance(connections_data, list):
        raise ValueError(f"{connection_env_var} must contain a JSON array")
    
    # Convert to ClientConfig objects
    connection_configs = []
    for idx, conn_data in enumerate(connections_data):
        try:
            # Extract required fields
            azure_endpoint = conn_data.get("AZURE_OPENAI_ENDPOINT")
            api_key = conn_data.get("AZURE_OPENAI_API_KEY")
            
            if not azure_endpoint or not api_key:
                raise ValueError(f"Connection config {idx} missing required fields")
            
            # Get additional configs, filtering out the required ones we've already handled
            additional_config = {
                k: v for k, v in conn_data.items() 
                if k not in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
            }
            
            # Create config object
            config = ClientConfig(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                additional_config=additional_config
            )
            connection_configs.append(config)
            
        except Exception as e:
            logger.warning(f"Error parsing connection config {idx}: {str(e)}")
    
    if not connection_configs:
        raise ValueError("No valid connection configurations found")
    
    # Initialize the client manager
    await client_manager.initialize(base_config, connection_configs)
    return client_manager

class AzureOpenAIRoundRobinClient(AzureOpenAIChatCompletionClient):
    """
    An extension of AzureOpenAIChatCompletionClient that distributes requests across multiple 
    Azure OpenAI endpoints in a round-robin fashion.
    
    This client uses the AzureOpenAIClientsRoundRobin manager to rotate between multiple 
    client configurations, balancing load and preventing rate limit issues.
    """

    def __init__(self, **kwargs: AzureOpenAIClientConfigurationConfigModel):
        """Initialize with the same parameters as AzureOpenAIChatCompletionClient.
        The actual endpoints and API keys are managed by the round-robin client manager.
        """
        # Initialize with default values that will be overridden later
        super().__init__(**kwargs)
        
        # Ensure the client manager has at least one client
        if client_manager.client_count == 0:
            raise ValueError("No Azure OpenAI clients available in the round-robin pool. "
                            "Please check your AZURE_OPENAI_ROUND_ROBIN_CONNETION environment variable.")
        
        logging.info(f"Initialized AzureOpenAIRoundRobinClient with {client_manager.client_count} endpoints")
    
    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool | Type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        """Override the create method to use round-robin client selection."""
        # Get the next client from the round-robin manager
        client = await client_manager.get_next_client()
        
        # Use the selected client to create the response
        result = await client.create(messages,
            tools=tools,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        
        return result
    
    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool | Type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
        max_consecutive_empty_chunk_tolerance: int = 0,
    ):
        """Override the create_stream method to use round-robin client selection."""
        # Get the next client from the round-robin manager
        client = await client_manager.get_next_client()
        
        # Use the selected client to create the stream
        async for chunk in client.create_stream(
            messages,
            tools=tools,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
            max_consecutive_empty_chunk_tolerance=max_consecutive_empty_chunk_tolerance,
        ):
            yield chunk
    
    async def close(self) -> None:
        """Close all clients in the round-robin pool."""
        # Get list of all clients to close
        for i in range(client_manager.client_count):
            client = await client_manager.get_next_client()
            await client.close()
    
    def actual_usage(self) -> Dict[str, int]:
        """
        Return aggregated actual usage across all clients.
        Note: This is a simplified implementation and may not reflect the exact usage
        across all clients precisely.
        """
        # We would need to collect usage from all clients here,
        # but for simplicity return usage from the current client's state
        # In a production environment, you would maintain a shared counter
        return super().actual_usage()
    
    def total_usage(self) -> Dict[str, int]:
        """
        Return aggregated total usage across all clients.
        Note: This is a simplified implementation and may not reflect the exact usage
        across all clients precisely.
        """
        # Similar to actual_usage, we would need to aggregate across all clients
        # but using a simplified implementation here
        return super().total_usage()
    
    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """Count tokens using the default model client implementation."""
        return super().count_tokens(messages, tools=tools)
    
    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        """Calculate remaining tokens using the default model client implementation."""
        return super().remaining_tokens(messages, tools=tools)