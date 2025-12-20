from mcp.server.fastmcp import FastMCP
from retriever import RedditRetriever
from typing import List, Dict, Any, Tuple, Optional
import yaml

# Create an MCP server
mcp = FastMCP('reddit-retriever')

# Initialize the Reddit retriever
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

retriever = RedditRetriever(config['embeddings']['file_path'], config['database']['file_path'], config['embeddings'].get('persist_in_gpu', True))

@mcp.tool()
def search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search relevant Reddit posts based on the query.

    Parameters:
        query (str): The user's question.
        limit (int): The maximum number of results to return (default: 10).
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the search results with its corresponding post id.
    """
    results = retriever.search(query, limit)
    return [{'id': r['id']} for r in results]
    # return results

@mcp.tool()
def get_post(post_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the full content of a Reddit post by its ID.

    Parameters:
        post_id (str): The ID of the Reddit post.
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the post details or None if not found.
    """
    return retriever.get_post(post_id)

@mcp.tool()
def get_replies(id: str, offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve comments for a Reddit post or comment, sorted by highest score.

    Parameters:
        id (str): The ID of the Reddit post or comment.
        offset (int): The starting index for comments (default: 0).
        limit (int): The maximum number of comments to return (default: 10).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the comments.
    """
    return retriever.get_replies(id, offset, limit)

if __name__ == '__main__':
    mcp.run()