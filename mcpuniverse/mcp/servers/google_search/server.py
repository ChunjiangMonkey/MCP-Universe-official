"""
An MCP server for Google search
"""
# pylint: disable=broad-exception-caught
import os
import json
import math
from typing import List, Dict, Any

import asyncio
import click
from mcp.server.fastmcp import FastMCP
from mcpuniverse.common.logger import get_logger

SERP_API_BASE = "https://google.serper.dev/search"
API_KEY = os.environ.get("SERP_API_KEY", "")


async def _search(
        query: str,
        location: str = "",
        engine: str = "google",
        num_items: int = 20,
        timeout: float = 30
) -> List[Dict[str, Any]]:
    """
    Make a request to the Serp API using curl.

    :param query: The search query string.
    :param location: The location for the search query.
    :param engine: The search engine to use (default is "google").
    :param num_items: The maximum number of results to return.
    :param timeout: The timeout.
    """
    all_items = []

    payload = {"q": query, "num": num_items}
    if location:
        payload["location"] = location

    proc = await asyncio.create_subprocess_exec(
        "curl", "-s", "-v", "-X", "POST", SERP_API_BASE,
        "-H", f"X-API-KEY: {API_KEY}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
        "--max-time", str(int(timeout)),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"curl failed (code {proc.returncode}): stdout={stdout.decode()}, stderr={stderr.decode()}")


    results = json.loads(stdout.decode())
    # Serper.dev uses "organic" instead of "organic_results"
    for idx, item in enumerate(results.get("organic", [])):
        all_items.append({
            "position": idx + 1,
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link"),
        })
    return all_items[:num_items]


def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server.

    :param port: Port for SSE.
    :return: The MCP server.
    """
    mcp = FastMCP("google_search", port=port)

    @mcp.tool()
    async def search(query: str) -> str:
        """
        A tool to execute the Google search and return the top results.

        Args:
            query: The search query string.
        """
        try:
            items = await _search(query=query)
            return "\n".join([json.dumps(item, ensure_ascii=False, indent=2) for item in items])
        except Exception as e:
            import traceback
            return json.dumps({"error": f"Search failed: {str(e)}", "traceback": traceback.format_exc()})

    return mcp


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option("--port", default="8000", help="Port to listen on for SSE")
def main(transport: str, port: str):
    """
    Starts the initialized MCP server.

    :param port: Port for SSE.
    :param transport: The transport type, e.g., `stdio` or `sse`.
    """
    print(f"Starting the MCP server on port {port} with transport {transport}")
    assert transport.lower() in ["stdio", "sse"], \
        "Transport should be `stdio` or `sse`"
    logger = get_logger("Service:google_search")
    logger.info("Starting the MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())
