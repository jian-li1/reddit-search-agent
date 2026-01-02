from mcp.server.fastmcp import FastMCP
from retriever import RedditRetriever
from typing import List, Dict, Any, Tuple, Optional
import yaml
import logging
from functools import lru_cache

logging.basicConfig(format='%(message)s', level=logging.INFO)

# Create an MCP server
if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mcp = FastMCP('reddit-retriever', host=config.get('server', {}).get('host', 'localhost'), port=config.get('server', {}).get('port', 8000))
else:
    mcp = FastMCP('reddit-retriever')

@mcp.tool()
@lru_cache(maxsize=50)
def search(query: str, limit: int = 10, subreddit: str = None) -> List[Dict[str, Any]]:
    """
    Search relevant Reddit posts based on the query.

    Parameters:
        query (str): A descriptive search query.
        limit (int): The maximum number of results to return (default: 10).
        subreddit (str): The subreddit to search within (default: None, meaning all subreddits).
    Returns:
        Search results containing the post ID, title, snippet, score, subreddit, and number of replies for each post.
        Note: Search results do not include the full content of the posts.
        To view the content of a post, use the get_post tool with the post ID.
    """
    if retriever_instance is None:
        raise RuntimeError("Server is starting up, retriever not ready.")
        
    results = retriever_instance.search(query, limit, subreddit)
    # return [{'id': r['id'], 'title': r['title']} for r in results]
    return results

@mcp.tool()
@lru_cache(maxsize=50)
def get_post(post_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the full content of a Reddit post by its ID. To view comments, use the get_replies tool with the post ID.

    Parameters:
        post_id (str): The ID of the Reddit post.
    
    Returns:
        Full content of the Reddit post along with its metadata, including title, body, score, and number of replies.
    """
    if retriever_instance is None:
        raise RuntimeError("Server is starting up, retriever not ready.")

    return retriever_instance.get_post(post_id)

@mcp.tool()
@lru_cache(maxsize=50)
def get_replies(id: str, offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve replies/comments for a Reddit post or comment, sorted by highest score.

    Parameters:
        id (str): The ID of the Reddit post or comment.
        offset (int): The starting index for comments (default: 0).
        limit (int): The maximum number of comments to return (default: 10).

    Returns:
        A list of replies/comments with their metadata, including the author and score.
    """
    if retriever_instance is None:
        raise RuntimeError("Server is starting up, retriever not ready.")

    return retriever_instance.get_replies(id, offset, limit)

if __name__ == '__main__':
    # Initialize the Reddit retriever
    retriever_instance = RedditRetriever(
        config.get('embeddings', {}).get('file_path', ''), 
        config.get('db_file_path', ''),
        config.get('embeddings', {}).get('persist_in_gpu', True)
    )

    try:
        mcp.run(transport='sse')
    except KeyboardInterrupt:
        del retriever_instance
        print("Server stopped by user.")