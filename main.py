import asyncio
import sys
import json
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from openai import OpenAI
import requests

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SYSTEM_PROMPT = """\
Enable deep thinking subroutine.\
"""

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(
            api_key='sk-no-api-key',
            base_url="http://localhost:8080/v1",
        )
        # Initialize conversation history with the system prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.available_tools: List[Dict[str, Any]] = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        server_params = StdioServerParameters(
            command='uv',
            args=['run', server_script_path],
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

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

    async def process_query(self, query: str) -> str:
        """Process a query using the MCP session and OpenAI"""
        
        # Append the new user message to the persistent history
        self.messages.append({"role": "user", "content": query})

        # Start the conversation loop
        while True:
            response = self.openai.chat.completions.create(
                model=sys.argv[2],
                messages=self.messages,
                tools=self.available_tools if self.available_tools else None,
            )

            response_message = response.choices[0].message

            # If the model wants to call tools
            if response_message.tool_calls:
                # Append the assistant's intent to call tools to history
                self.messages.append(response_message)


                # Process every tool call requested
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    if tool_name == 'search':
                        response = requests.post(f"http://127.0.0.1:8080/api/models/unload/{sys.argv[2]}")
                    
                    # Parse arguments safely
                    try:
                        tool_args_dict = json.loads(tool_args)
                    except json.JSONDecodeError:
                        print(f"\nError decoding arguments for {tool_name}")
                        tool_args_dict = {}

                    # Execute the MCP tool
                    print(f"\n[Calling tool: {tool_name} with args {tool_args_dict}]")
                    result = await self.session.call_tool(tool_name, tool_args_dict)

                    # Append the result back to messages with the correct ID
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result.content)
                    })
                
                # The loop continues to send the tool outputs back to the model
            
            # If no tools are called, we have our final answer
            else:
                final_content = response_message.content
                # IMPORTANT: Append the final answer to history so the model remembers it
                self.messages.append({"role": "assistant", "content": final_content})
                return final_content

    async def chat_loop(self):
        """Interactive chat loop"""
        print("\n--- MCP Client Session Started ---")
        print("Type 'quit' or 'exit' to end the session.")
        
        while True:
            try:
                query = input("\n>>> ").strip()
                if query.lower() in ['quit', 'exit']:
                    break
                if not query:
                    continue

                response = await self.process_query(query)
                print(f"\n{response}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")

async def main():
    if len(sys.argv) < 3:
        print("Usage: python client.py <path_to_server_script> <model_name>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())