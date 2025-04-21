import asyncio
import os

from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    AzureOpenAIClientConfigurationConfigModel,
)
from dotenv import load_dotenv

# Import the round-robin client implementation
from roundRobin import (
    AzureOpenAIRoundRobinClient,
    initialize_client_manager_from_env,
)

load_dotenv()

OPEN_TOPIC_CLASS_GENERATION_AGENT = "Open Topic Agent Team"

CATCH_UP_AND_EXPLORE_BY_AI_AGENT = "Catch-up And Explore Agent Team"

CURRENT_AGENT_TEAM_NAME = "Current Agent Team Name"

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

# Check if round-robin mode is enabled
USE_ROUND_ROBIN = os.environ.get("USE_AZURE_OPENAI_ROUND_ROBIN", "false").lower() == "true"

# Initialize the round-robin client manager if enabled
_round_robin_initialized = False

async def _init_round_robin():
    """Initialize the round-robin client manager if not already initialized."""
    global _round_robin_initialized
    if not _round_robin_initialized and USE_ROUND_ROBIN:
        base_config = {
            "model": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
            "temperature": 0.0,
            "max_tokens": 2000,
            "top_p": 0.0,
        }
        await initialize_client_manager_from_env(base_config)
        _round_robin_initialized = True

# Initialize round-robin in the background if enabled
if USE_ROUND_ROBIN:
    try:
        # Create a new event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the initialization
        loop.run_until_complete(_init_round_robin())
    except Exception as e:
        print(f"Warning: Failed to initialize round-robin client manager: {str(e)}")
        print("Falling back to standard Azure OpenAI client")
        USE_ROUND_ROBIN = False


def get_model_client(**kwargs: AzureOpenAIClientConfigurationConfigModel):
    if USE_ROUND_ROBIN:
        return AzureOpenAIRoundRobinClient(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=AZURE_OPENAI_API_KEY,           # This will be overridden by the round-robin manager
            azure_endpoint=AZURE_OPENAI_ENDPOINT,   # This will be overridden by the round-robin manager
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            max_tokens=2000,
            top_p=0.0,
            **kwargs
        )
    else:
        return AzureOpenAIChatCompletionClient(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            max_tokens=2000,
            top_p=0.0,
            **kwargs
        )

def get_advance_model_client(**kwargs: AzureOpenAIClientConfigurationConfigModel):
    if USE_ROUND_ROBIN:
        return AzureOpenAIRoundRobinClient(
            model=os.environ.get("AZURE_OPENAI_ADVANCED_DEPLOYMENT_NAME"),
            api_key=AZURE_OPENAI_API_KEY,           # This will be overridden by the round-robin manager
            azure_endpoint=AZURE_OPENAI_ENDPOINT,   # This will be overridden by the round-robin manager
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            max_tokens=2000,
            top_p=0.0,
            **kwargs
        )
    else:
        return AzureOpenAIChatCompletionClient(
            model=os.environ.get("AZURE_OPENAI_ADVANCED_DEPLOYMENT_NAME"),
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            max_tokens=2000,
            top_p=0.0,
            **kwargs
        )

def get_moderate_model_client(**kwargs: AzureOpenAIClientConfigurationConfigModel):
    if USE_ROUND_ROBIN:
        return AzureOpenAIRoundRobinClient(
            model=os.environ.get("AZURE_OPENAI_MODERATED_DEPLOYMENT_NAME"),
            api_key=AZURE_OPENAI_API_KEY,           # This will be overridden by the round-robin manager
            azure_endpoint=AZURE_OPENAI_ENDPOINT,   # This will be overridden by the round-robin manager
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            max_tokens=2000,
            top_p=0.0,
            **kwargs
        )
    else:
        return AzureOpenAIChatCompletionClient(
            model=os.environ.get("AZURE_OPENAI_MODERATED_DEPLOYMENT_NAME"),
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            max_tokens=2000,
            top_p=0.0,
            **kwargs
        )

def get_low_model_client(**kwargs: AzureOpenAIClientConfigurationConfigModel):
    if USE_ROUND_ROBIN:
        return AzureOpenAIRoundRobinClient(
            model=os.environ.get("AZURE_OPENAI_LOW_DEPLOYMENT_NAME"),
            api_key=AZURE_OPENAI_API_KEY,           # This will be overridden by the round-robin manager
            azure_endpoint=AZURE_OPENAI_ENDPOINT,   # This will be overridden by the round-robin manager
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            max_tokens=2000,
            top_p=0.0,
            **kwargs
        )
    else:
        return AzureOpenAIChatCompletionClient(
            model=os.environ.get("AZURE_OPENAI_LOW_DEPLOYMENT_NAME"),
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
            max_tokens=2000,
            top_p=0.0,
            **kwargs
        )