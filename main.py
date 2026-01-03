import asyncio
import sys
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from openai import AsyncOpenAI
import requests
import yaml
import argparse
from urllib.parse import urljoin, urlparse
from mcp import ClientSession
from mcp.client.sse import sse_client
import readline

# ANSI colors for better visualization
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[90m"
RESET = "\033[0m"

PROMPT_YELLOW = f"\001{YELLOW}\002"
PROMPT_RESET = f"\001{RESET}\002"

def parse_args() -> argparse.Namespace:
    """Parses and returns the values of command line arguments."""
    parser = argparse.ArgumentParser(description="MCP Client for Reddit Search Agent (SSE Mode)")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config YAML file"
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
        
        # Use AsyncOpenAI for proper non-blocking streaming
        self.openai = AsyncOpenAI(
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
        
        print(f"\nConnected to server with tools: {[tool.name for tool in response.tools]}")

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
                try:
                    requests.post(unload_url)
                except Exception as e:
                    print(f"Failed to unload model: {e}")

    async def process_query(self, query: str) -> str:
        """Process a query using the MCP session and OpenAI with Streaming"""
        
        self.messages.append({"role": "user", "content": query})

        tool_choice = "auto"
        while True:            
            # Variables to accumulate the stream
            full_content = ""
            full_reasoning = ""
            tool_calls_accumulator = {} # Keyed by index
            is_thinking = False
            
            # Create the stream
            stream = await self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.available_tools if self.available_tools else None,
                tool_choice=tool_choice,
                stream=True # Enable streaming
            )
            # Reset tool choice for potential re-try
            tool_choice = "auto" 

            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta

                # --- Handle Reasoning/Thinking Tokens (DeepSeek R1 style) ---
                # Some OpenAI-compatible servers (llama.cpp/vllm) send this in 'reasoning_content'
                reasoning_chunk = getattr(delta, 'reasoning_content', None)
                if reasoning_chunk:
                    if not is_thinking:
                        print(f"\n{DIM}--- Start Thinking ---{RESET}")
                        is_thinking = True
                    print(f"{DIM}{reasoning_chunk}{RESET}", end="", flush=True)
                    full_reasoning += reasoning_chunk

                # --- Handle Standard Content ---
                if delta.content:
                    if is_thinking:
                        print(f"\n{DIM}--- End Thinking ---{RESET}\n", end="", flush=True)
                        is_thinking = False

                    if full_content == "" and not delta.tool_calls:
                        print()
                    
                    print(delta.content, end="", flush=True)
                    full_content += delta.content

                # --- Handle Tool Calls (Aggregating Fragments) ---
                if delta.tool_calls:
                    if is_thinking:
                        print(f"\n{DIM}--- End Thinking ---{RESET}\n", end="", flush=True)
                        is_thinking = False
                    
                    for tool_part in delta.tool_calls:
                        idx = tool_part.index
                        
                        if idx not in tool_calls_accumulator:
                            tool_calls_accumulator[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        if tool_part.id:
                            tool_calls_accumulator[idx]["id"] += tool_part.id
                        if tool_part.function.name:
                            tool_calls_accumulator[idx]["function"]["name"] += tool_part.function.name
                        if tool_part.function.arguments:
                            tool_calls_accumulator[idx]["function"]["arguments"] += tool_part.function.arguments

            # CHANGED: Only print a newline if actual text content was streamed.
            # This prevents blank lines appearing when the model only calls tools.
            if full_content:
                print()

            # Reconstruct the message object from the stream
            response_message = {
                "role": "assistant",
                "content": full_content if full_content else None
            }
            
            # Convert accumulated tool calls to list
            if tool_calls_accumulator:
                tool_calls_list = []
                for idx in sorted(tool_calls_accumulator.keys()):
                    tool_calls_list.append(tool_calls_accumulator[idx])
                
                # IMPORTANT: Convert dicts back to objects expected by OpenAI client or maintain as dict
                # For appending to history, dicts work fine with most clients, but let's be safe.
                # Append the dict representation to self.messages
                response_message["tool_calls"] = tool_calls_list
                self.messages.append(response_message)
                
                # Execute Tools
                for tool_call in tool_calls_list:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    tool_call_id = tool_call["id"]

                    
                    if tool_name == "search":
                        # Require model to call a tool after search to get post content
                        # tool_choice = "required"
                        
                        # VRAM Optimization
                        try:
                            self.unload_model()
                        except Exception as e:
                            print(f"Warning: Could not unload model: {e}")

                    try:
                        tool_args_dict = json.loads(tool_args)
                    except json.JSONDecodeError:
                        print(f"\nError decoding arguments for {tool_name}")
                        tool_args_dict = {}

                    print(f"\n{CYAN}[Calling tool: {tool_name} with args {tool_args_dict}]{RESET}")
                    
                    # Call the MCP Tool
                    result = await self.session.call_tool(tool_name, tool_args_dict)
                    
                    # Convert result to string handling TextContent or ImageContent
                    # MCP tool result content is a list of content objects
                    tool_output = ""
                    if hasattr(result, 'content'):
                        for content_item in result.content:
                            if hasattr(content_item, 'text'):
                                tool_output += content_item.text
                            else:
                                tool_output += str(content_item)
                    else:
                        tool_output = str(result)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_output
                    })
            else:
                # No tool calls, just a standard reply
                self.messages.append(response_message)
                return full_content

    async def chat_loop(self):
        """Interactive chat loop"""
        print("\n--- MCP Client Session Started (SSE Mode) ---")
        print("Type 'quit', 'exit', or enter Ctrl+C twice to end the session.")
        
        while True:
            # Check for EOFError to handle Ctrl+D gracefully
            try:
                # print(f"\n{YELLOW}>>>{RESET} ", end="", flush=True)
                print()
                query = input(f"{PROMPT_YELLOW}>>>{PROMPT_RESET} ").strip()
            except EOFError:
                break
            
            if query.lower() in {"quit", "exit"}:
                break
            if not query:
                continue

            try:
                # Don't print the response here anymore because it's streamed inside process_query
                await self.process_query(query)
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

    subreddits = config.get("subreddits", [])

    # Include the subreddit sentence only if the user provided subreddit info
    if subreddits:
        system_prompt_parts.append("You have access to the following subreddits:")
        for subreddit in subreddits:
            subreddit_name = subreddit.get("name")
            subreddit_desc = subreddit.get("description")
            if subreddit_name:
                system_prompt_parts.append(f"- `r/{subreddit_name}`" + (f': {subreddit_desc}' if subreddit_desc else ''))

    system_prompt_parts.extend([
        "",
        "## Instructions",
        "Carefully read and analyze each post/comment before moving on to the next.",
        "Use the tools provided to search for relevant posts/comments based on the user's query.",
        "## Core Rules and Requirements",
        "Answer questions using only the information provided in the posts/comments.",
        "Answers must be derived from **at least 5** posts/comments, so you must call `get_post` and `get_replies` at least 5 times each.",
        "Do not answer directly or make up an answer. Do not attempt to answer solely based on the post titles or snippets.",
        "Always cite specific parts of the posts/comments used in your answer.",
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
        "Use this tool to get the comments/replies of the message (post/comment) by specifying its ID.",
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
        base_url=config.get("base_url"), 
        model=config.get("model"), 
        api_key=config.get("api_key", ""), 
        system_prompt=system_prompt, 
        llama_swap_no_persist=config.get("llama_swap_no_persist", False)
    )
    try:
        if config.get("server") and config["server"].get("host") and config["server"].get("port"):
            server_url = f"http://{config['server']['host']}:{config['server']['port']}/sse"
        await client.connect_to_server(server_url)
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