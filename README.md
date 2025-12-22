# Reddit Search Agent

Reddit Search Agent is an agentic system that provides responses based on posts and comments from a specific subreddit. This AI agent is designed to mimic how a human searches for information online by scrolling through relevant Reddit discussions. The agentic loop is represented as follows:

1. The user asks a question to the agent.
2. The agent takes the question and forms a query to search for related posts.
3. The agent selects relevant posts from the search results and views the content of the posts.
4. Furthermore, the agent can choose to view replies from the posts and replies from relevant comments.
5. At any point, the agent can repeat steps (2), (3), and (4) if they would like to see more information.
6. When the agent thinks there is adequate information from the posts/comments to answer the user’s question, they can generate a final response based on what they found.

Thanks to the tool-calling capabilities in many LLMs, agentic workflows are made easier to facilitate. Thus, in this project, we have an MCP server (`reddit-search-mcp.py`) that exposes the search functions of the retriever to the LLM, and an MCP client with custom system prompts to ground the LLM's actions and responses. Due to the portability of MCP servers, `reddit-search-mcp.py` can also be connected to other interfaces that support MCP server integration, such as LM Studio and Claude Desktop.

The entire agent can be run locally, with both the MCP server and the LLM hosted on a local machine.

## Data
The subreddit submissions and comments can be downloaded from this [torrent](https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13) containing the the zstandard files of the submissions and comments from the top 40,000 subreddits from June 2005 to December 2024 (more information can be found [here](https://www.reddit.com/r/pushshift/comments/1itme1k/separate_dump_files_for_the_top_40k_subreddits/)). Scripts are provided in this repo to extract the `.zst` files into CSV files, filter out submissions/comments, and embed submissions and insert all messages into a SQLite database.

## Retriever
The retriever holds the vector embeddings of the posts and the SQLite database containing the content and information of every post and comment. It uses the embedding model [`jinaai/jina-embeddings-v4`](https://huggingface.co/jinaai/jina-embeddings-v4) and the reranker model [`jinaai/jina-reranker-v3`](https://huggingface.co/jinaai/jina-reranker-v3), both of which are perfect for the information retrieval task of this system. When a query is made, the retriever performs a hybrid search (leveraging both similarity search on the vector embeddings and BM25 search on the database) over all the posts, followed by a reranking pass. The combination of these retrievals is great at delivering accurate and relevant results to the query.

## Tools and Functions
There are three tools in `reddit-retriever.py` that are made available to the LLM:

- `search`: This tool takes in a query and a limit $k$ to get the top $k$ relevant posts. It returns a list of the ID and title of each post.
- `get_post`: This tool retrieves the full content of a post along with its metadata.
- `get_replies`: This tool retrieves comments made on a Reddit post or comments, sorted by highest score.

## Usage and Configuration
This project uses the `uv` package manager. To install packages, run
```
uv sync
```

### Preprocessing
First, convert the `.zst` files to CSV files:
```bash
uv run data/to_csv.py <submission zst file> <submission csv file>
uv run data/to_csv.py <comment zst file> <comment csv file>
```

Next, filter out the posts and comments by score and depth. Example:
```bash
uv run data/filter_csv.py -d <directory of csv files> --subreddit <subreddit name> --min-score <score> --max-depth <depth>
```
The maximum depth is the maximum level of nesting of a Reddit thread.

Now, embed the Reddit posts and build the SQLite database:
```bash
uv run data/build.py -d <directory of filtered csv files> --subreddit <subreddit name> --chunk-size <size> --chunk-overlap <chunk overlap size> --batch-size <batch size>
```

### Configuration
The MCP client (`main.py`) and MCP server (`reddit-search-mcp.py`) use the `config.yaml` file for configuration. An example of `config.yaml` is shown below:
```yaml
# MCP Server
db_file_path: ./database/subreddit.db

embeddings:
  file_path: ./embeddings/subreddit_embeddings.jsonl
  persist_in_gpu: true

subreddit:
  name: subreddit
  description: Subreddit description

server:
  host: localhost
  port: 8000

# MCP Client
system_prompt: You are a helpful assistant.

base_url: http://localhost:8080/v1/
api_key: your_api_key_here
model: llama-3.1-8B
server_url: http://localhost:8000/sse
llama_swap_no_persist: false
```

The following fields are used by the MCP server:
- `db_file_path`: SQLite database containing all the posts and comments.
- `embeddings`
    - `file_path`: JSONL file containing the vector embeddings of the posts.
    - `persist_in_gpu` (optional; default: `true`): Whether the embedding and reranker model should be retained in the GPU across calls (see more information about [serving models](#serving-models-and-limitations)).
- `server`: 
    - `host` (optional; default: `localhost`): Host of the MCP server
    - `port` (optional; default: `8000`): Port number of the MCP server

These fields are used by the MCP client:
- `system_prompt`: Custom system prompt to provide additional information to the LLM. A boilerplate system prompt is already included in the MCP client script to give instructions about tool-calling.
- `base_url`: API endpoint for accessing the LLM.
- `api_key` (optional): API key for accessing the LLM, if using a remote API endpoint.
- `model`: Name/ID of the model.
- `server_url`: URL of the MCP server.
- `llama_swap_no_persist` (optional; default: `false`): Whether to unload LLM from GPU between tool calls if using `llama-swap` (see more information about [serving models](#serving-models-and-limitations)).


### Serving Models and Limitations
For this project, the LLM is served locally using [llama.cpp](https://github.com/ggml-org/llama.cpp).

Even with a setup of a single RTX 4090, it is difficult to load the embedding and reranker models alongside a decent LLM with plenty of context length onto the GPU at the same time, due to limited VRAM. The workaround is to hot-swap the models during tool calls so that the LLM is unloaded when the embedding and reranker models are loaded for retrieval and vice versa. 

The `persist_in_gpu` field in the `config.yaml` file allows the embedding and reranker model to be loaded only when a query is made. By using [llama-swap](https://github.com/mostlygeek/llama-swap) on top of llama.cpp, the `llama_swap_no_persist` field allows the MCP client to send a request to llama-swap to unload the LLM as soon as a tool call is made for retrieval. This hot-swapping approach incurs some overhead in runtime, albeit fairly minimal on a Linux machine.

Alternatively, you can set the environmental variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` when serving the LLM to prevent out of memory error on llama.cpp (more information [here](https://github.com/ggml-org/llama.cpp/blob/ddcb75dd8ac42dc23eb84f13bb17670fe9f2d49b/docs/build.md#unified-memory)).

### Running the MCP Client and Server
With the `config.yaml` set up, simply run the MCP server as follows:
```
uv run reddit-search-mcp.py
```

Similarly, run the MCP client as follows:
```
uv run main.py
```

With the LLM and MCP server running, the agent can now be interacted with via CLI.

Alternatively, if you are connecting the MCP server to LM Studio or Claude Desktop, simply add the following to the `mcp.json` file. Replace the host and port in the URL to the ones used to run the server.
```json
{
  "mcpServers": {
    "reddit-retriever": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

## Backlog
- Searching posts on multiple subreddits
- Filter search results by flair
- Construct more specific prompts and doc strings for the tool calls
- More to come!
