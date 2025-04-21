import asyncio
import json
import os
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import chainlit as cl
import html2text
import httpx
from autogen_core.tools import FunctionTool
from bs4 import BeautifulSoup


def clean_url(url: str) -> str:
    """Clean URL by removing query parameters.
    
    Args:
        url: The URL to clean.
        
    Returns:
        str: URL without query parameters.
    """
    parsed = urlparse(url)
    clean = parsed.scheme + "://" + parsed.netloc + parsed.path
    return clean


@cl.step(type="tool", name="is_url_accessible")
def is_url_accessible_with_chainlit(url: str) -> bool:
    """Check if a URL is accessible with Chainlit context.

    Args:
        url: The URL to check.

    Returns:
        bool: True if the URL is accessible, False otherwise.
    """
    try:
        response = httpx.head(clean_url(url), timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def is_url_accessible(url: str) -> bool:
    """Check if a URL is accessible without Chainlit context.

    Args:
        url: The URL to check.

    Returns:
        bool: True if the URL is accessible, False otherwise.
    """
    try:
        response = httpx.head(clean_url(url), timeout=5)
        return response.status_code == 200
    except Exception:
        return False


url_accessible_valid_tool = FunctionTool(
    is_url_accessible,
    name="urlAccessibleValidTool",
    description="A tool that validates the url is accessible and valid.",
    global_imports=[
        "httpx",
    ],
)
