from typing import Literal

from tavily import TavilyClient
import os
from langchain_core.tools import tool


@tool("internet_search")
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search via Tavily and return search results.

    Args:
        query: The query string to search for.
        max_results: Maximum number of results to return.
        topic: Search topic category.
        include_raw_content: Whether to include raw page content.

    Returns:
        Search results from Tavily API.
    """
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
