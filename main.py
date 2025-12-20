import asyncio
import sys
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from openai import OpenAI
import requests
import yaml
import argparse
from urllib.parse import urljoin, urlparse
from mcp import ClientSession
from mcp.client.sse import sse_client

def parse_args() -> argparse.Namespace:
    """Parses and returns the values of command line arguments."""
    parser = argparse.ArgumentParser(description="MCP Client for Reddit Search Agent (SSE Mode)")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config YAML file"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API endpoint for accessing complete model"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key for accessing completion and embedding models"
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        required=True,
        help="The name of the model to use"
    )
    parser.add_argument(
        "--llama-swap-np", 
        "--llama-swap-no-persist",
        dest="llama_swap_no_persist", 
        action="store_true",
        help="Models will not persist in GPU memory between calls"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="URL of the MCP server"
    )
    return parser.parse_args()

class MCPClient:
    def __init__(
            self, 
            base_url: Optional[str] = None, 
            model: str = "", 
            api_key: str = "", 
            system_prompt: str = "", 
            llama_swap_no_persist: bool = False
        ):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        # store base_url and swap flag for unloading logic
        self.base_url = base_url
        self.llama_swap_no_persist = llama_swap_no_persist

        # Initialize conversation history with the system prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.available_tools: List[Dict[str, Any]] = []

    # Argument is now a URL, not a file path
    async def connect_to_server(self, server_url: str):
        """Connect to an MCP server via SSE"""
        print(f"Attempting to connect to {server_url}...")

        # Use sse_client context manager
        # This handles the HTTP connection and headers automatically
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url=server_url)
        )
        
        self.read, self.write = sse_transport
        
        # Initialize the session using the SSE streams
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )

        await self.session.initialize()

        # List available tools and format them for OpenAI once
        response = await self.session.list_tools()
        
        self.available_tools = []
        for tool in response.tools:
            self.available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema 
                }
            })
        
        print("\nConnected to server with tools:", [tool.name for tool in response.tools])

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
    
    def unload_model(self):
        # Only attempt unload when llama swap no persist is requested
        # and the base_url is local (localhost or 127.0.0.1)
        if self.llama_swap_no_persist and self.base_url:
            parsed = urlparse(self.base_url)
            hostname = parsed.hostname or ""
            if hostname in {"localhost", "127.0.0.1"}:
                # Strip trailing /v1 or /v1/ from the path
                base_root = f"{parsed.scheme}://{parsed.netloc}"

                print(f"\n[System] Unloading model \"{self.model}\" to free VRAM for search...")
                unload_url = urljoin(base_root, f"/api/models/unload/{self.model}")
                requests.post(unload_url)

    async def process_query(self, query: str) -> str:
        """Process a query using the MCP session and OpenAI"""
        
        self.messages.append({"role": "user", "content": query})

        while True:
            response = self.openai.chat.completions.create(
                model=self.model, # Model name passed as argument
                messages=self.messages,
                tools=self.available_tools if self.available_tools else None,
            )

            response_message = response.choices[0].message

            # Handle tool calls if any
            if response_message.tool_calls:
                self.messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    # VRAM Optimization: Unload LLM before heavy search if needed
                    if tool_name == "search":
                        try:
                            self.unload_model()
                        except Exception as e:
                            print(f"Warning: Could not unload model: {e}")

                    try:
                        tool_args_dict = json.loads(tool_args)
                    except json.JSONDecodeError:
                        print(f"\nError decoding arguments for {tool_name}")
                        tool_args_dict = {}

                    print(f"\n[Calling tool: {tool_name} with args {tool_args_dict}]")
                    result = await self.session.call_tool(tool_name, tool_args_dict)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result.content)
                    })
            
            else:
                final_content = response_message.content
                self.messages.append({"role": "assistant", "content": final_content})
                return final_content

    async def chat_loop(self):
        """Interactive chat loop"""
        print("\n--- MCP Client Session Started (SSE Mode) ---")
        print("Type 'quit' or 'exit' to end the session.")
        
        while True:
            query = input("\n>>> ").strip()
            
            if query.lower() in {"quit", "exit"}:
                break
            if not query:
                continue

            try:
                response = await self.process_query(query)
                print(f"\n{response}")
            except Exception as e:
                print(f"\nError processing query: {e}")

def build_system_prompt(config: Dict[str, Any]) -> str:
    """Builds the system prompt based on the config."""
    base_system = config.get("system_prompt", "")

    system_prompt_parts = [
        base_system,
        "",
        "As a Reddit Search Agent, your task is to answer user's questions based on the content of Reddit posts/comments.",
    ]

    subreddit = config.get("subreddit")

    # Include the subreddit sentence only if the user provided subreddit info
    if subreddit:
        if subreddit.get("name"):
            system_prompt_parts.append(
                f"You have access to all posts/comments from the subreddit `{subreddit["name"]}`."
            )

        # Include the description only if provided
        if subreddit.get("description"):
            system_prompt_parts.append(
                f"Description of the subreddit: \"{subreddit["description"]}\""
            )

    system_prompt_parts.extend([
        "",
        "## Core Rule",
        "Answer questions using only the information provided in the posts/comments.",
        "Do not answer directly or make up an answer.",
        "Always cite specific parts of the post/comment used in your answer.",
        "If you are unsure of something, make a query using the tools provided.",
        "",
        "## Tools",
        "### search",
        "Use this tool to make a query to the database. It retrieves a list of post IDs.",
        "",
        "### get_post",
        "Use this tool to get the full content of the post by specifying the post ID.",
        "",
        "### get_replies",
        "Use this tool to get the replies of the message (post/comment) by specifying its ID.",
    ])

    return "\n".join([p for p in system_prompt_parts if p is not None])

async def main():
    args = parse_args()

    # Load config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Contruct the system prompt
    system_prompt = build_system_prompt(config)

    print("\n--- System Prompt ---")
    print(system_prompt)

    client = MCPClient(
        base_url=args.base_url, 
        model=args.model, 
        api_key=args.api_key, 
        system_prompt=system_prompt, 
        llama_swap_no_persist=args.llama_swap_no_persist
    )
    try:
        # Pass the URL from command line args
        await client.connect_to_server(args.server_url)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This catches Ctrl+C if it happens during input() or anywhere else
        print("\n\nUser interrupted. Exiting...")
        sys.exit(0)