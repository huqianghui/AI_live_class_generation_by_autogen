# Autonomously complete a coding task:
import asyncio

from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne

from config import get_model_client

model_client = get_model_client()

async def example_usage():
    m1 = MagenticOne(client=model_client)
    task = "Write a Python script to fetch data from an API."
    result = await Console(m1.run_stream(task=task))
    print(result)


if __name__ == "__main__":
    asyncio.run(example_usage())