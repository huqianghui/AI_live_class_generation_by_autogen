import os

from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    AzureOpenAIClientConfigurationConfigModel,
)
from dotenv import load_dotenv

load_dotenv()

OPEN_TOPIC_CLASS_GENERATION_AGENT = "Open Topic Class Generation Agent"

CATCH_UP_AND_EXPLORE_BY_AI_AGENT = "Catch-up And Explore By AI Agent"

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")


def get_model_client(**kwargs: AzureOpenAIClientConfigurationConfigModel):
    return AzureOpenAIChatCompletionClient(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
        top_p=0.0,
        **kwargs
    )
