from typing import Dict, Optional
from urllib.parse import urljoin, urlparse

import chainlit as cl
import html2text
import httpx
from autogen_core.tools import FunctionTool
from bs4 import BeautifulSoup


def clean_image_url(url: str) -> str:
    """Remove query parameters from image URLs.
    
    Args:
        url: The image URL that might contain query parameters
        
    Returns:
        str: Clean URL without query parameters
    """
    parsed = urlparse(url)
    clean = parsed.scheme + "://" + parsed.netloc + parsed.path
    return clean


@cl.step(type="tool", name="fetch_webpage")
async def fetch_webpage(
    url: str,
    include_images: bool = True,
    max_length: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Fetch a webpage and convert it to markdown format.

    Args:
        url: The URL of the webpage to fetch
        include_images: Whether to include image references in the markdown
        max_length: Maximum length of the output markdown (if None, no limit)
        headers: Optional HTTP headers for the request

    Returns:
        str: Markdown version of the webpage content

    Raises:
        ValueError: If the URL is invalid or the page can't be fetched
    """
    # Use default headers if none provided
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    try:
        # Fetch the webpage
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Convert relative URLs to absolute and clean image URLs
            for tag in soup.find_all(["a", "img"]):
                if tag.get("href"):
                    tag["href"] = urljoin(url, tag["href"])
                if tag.get("src"):
                    tag["src"] = urljoin(url, tag["src"])
                    # Clean image URLs by removing query parameters
                    if tag.name == "img":
                        tag["src"] = clean_image_url(tag["src"])

            # Configure HTML to Markdown converter
            h2t = html2text.HTML2Text()
            h2t.body_width = 0  # No line wrapping
            h2t.ignore_images = not include_images
            h2t.ignore_emphasis = False
            h2t.ignore_links = False
            h2t.ignore_tables = False

            # Convert to markdown
            markdown = h2t.handle(str(soup))

            # Trim if max_length is specified
            if max_length and len(markdown) > max_length:
                markdown = markdown[:max_length] + "\n...(truncated)"

            return markdown.strip()

    except httpx.RequestError as e:
        raise ValueError(f"Failed to fetch webpage: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Error processing webpage: {str(e)}") from e

fetch_webpage_tool = FunctionTool(
    fetch_webpage,
    name="FetchWebpageTool",
    description="A tool that fetches the content of a webpage and converts it to markdown. Requires the requests and beautifulsoup4 library to function.",
    global_imports=[
        "os",
        "html2text",
        {"module": "typing", "imports": ["Optional", "Dict"]},
        "httpx",
        {"module": "bs4", "imports": ["BeautifulSoup"]},
        {"module": "html2text", "imports": ["HTML2Text"]},
        {"module": "urllib.parse", "imports": ["urljoin", "urlparse"]},
    ],
)
