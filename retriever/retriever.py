import sqlite3
import json
import torch
import gc
import faiss
import numpy as np
import nltk
from transformers import AutoModel
from typing import List, Dict, Any, Tuple, Optional
import logging
import re

logging.basicConfig(format='%(message)s', level=logging.INFO)

# Ensure nltk punkt is downloaded for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class RedditRetriever:
    def __init__(self, embedding_file: str, db_file: str, persist_in_gpu: bool = True):
        """
        Initialize the retriever by connecting to the database, loading models, 
        and building the FAISS index.

        Parameters:
            embedding_file (str): Path to the JSONL file containing embeddings.
            db_file (str): Path to the SQLite database file.
            persist_in_gpu (bool): Whether to keep models in GPU memory persistently.
        """
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self.persist_in_gpu = persist_in_gpu
        # 1. Initialize Embedding Model (Jina v4)
        logging.info('Initializing Embedding Model...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.embed_model = AutoModel.from_pretrained(
            'jinaai/jina-embeddings-v4',
            trust_remote_code=True,
            torch_dtype=(torch.float16 if self.device == 'cuda' else torch.float32),
        )
        if not self.persist_in_gpu and self.device == 'cuda':
            self.embed_model.to('cpu')
        else:
            self.embed_model.to(self.device)
        self.embed_model.eval()

        # 2. Initialize Reranker Model (Jina v3)
        logging.info('Initializing Reranker Model...')
        self.reranker_model = AutoModel.from_pretrained(
            'jinaai/jina-reranker-v3',
            dtype='auto',
            trust_remote_code=True,
        )
        if not self.persist_in_gpu and self.device == 'cuda':
            self.reranker_model.to('cpu')
        else:
            self.reranker_model.to(self.device)
        self.reranker_model.eval()

        # 3. Build FAISS Index
        logging.info('Building FAISS Index from embeddings...')
        self.index = None
        self.metadata = []  # Maps FAISS index id -> (post_id, chunk_id)

        # Read JSONL file
        embeddings_list = []
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.metadata.append({
                    'post_id': data['id'], 
                    'chunk_id': data['chunk_id']
                })
                embeddings_list.append(data['embeddings'])

        if embeddings_list:
            # Convert to numpy array
            embeddings_matrix = np.array(embeddings_list, dtype='float32')
            dimension = embeddings_matrix.shape[1]
            
            # Initialize IndexFlatL2
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_matrix)
            logging.info(f'Indexed {len(embeddings_list)} chunks.')
        else:
            logging.info('Warning: No embeddings found in file.')

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search (BM25 + Vector) with RRF, deduplication, 
        and reranking to return the most relevant post snippets.

        Parameters:
            query (str): The user's search query.
            limit (int): The number of results to return.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing id, title, and snippet.
        """
        if not self.index:
            return []
        
        # Load model to device
        if not self.persist_in_gpu and self.device == 'cuda':
            self.embed_model.to(self.device)
            self.reranker_model.to(self.device)

        # --- Step 1: Vector Search (Dense) ---
        # Encode query
        query_embeddings = self.embed_model.encode_text(
            texts=[query],
            task='retrieval',
            prompt_name='query'
        )
        
        # Search FAISS (fetch more than limit to allow for RRF fusion and deduplication)
        # We fetch 100 or limit * 10 candidates to ensure overlap with BM25
        k_neighbors = max(100, limit * 10)
        distances, indices = self.index.search(np.array(query_embeddings[0].cpu().numpy()).reshape(1, -1), k_neighbors)
        
        # Store Dense Ranks: { (post_id, chunk_id): rank }
        dense_ranks = {}
        # indices[0] because we only have 1 query
        for rank, idx in enumerate(indices[0]):
            if idx == -1: continue # invalid index
            meta = self.metadata[idx]
            key = (meta['post_id'], meta['chunk_id'])
            dense_ranks[key] = rank

        # --- Step 2: BM25 Search (Sparse) ---
        cursor = self.conn.cursor()
        # Fetch top candidates from FTS5
        # Note: 'rank' in FTS5 is a calculated score, smaller is usually better/more relevant 
        # depending on calculation, but 'ORDER BY rank' puts best matches first.
        cursor.execute("""
            SELECT post_id, chunk_id 
            FROM submission_chunks 
            WHERE submission_chunks MATCH ? 
            ORDER BY rank 
            LIMIT ?
        """, (query, k_neighbors))
        
        bm25_results = cursor.fetchall()
        
        sparse_ranks = {}
        for rank, row in enumerate(bm25_results):
            key = (row['post_id'], row['chunk_id'])
            sparse_ranks[key] = rank

        # --- Step 3: Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        rrf_k = 60
        
        all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        for key in all_keys:
            score = 0.0
            if key in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[key])
            if key in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[key])
            rrf_scores[key] = score

        # --- Step 4: Deduplication (Max Score per Post) ---
        post_best_chunk = {} # post_id -> (chunk_id, score)
        
        for (post_id, chunk_id), score in rrf_scores.items():
            if post_id not in post_best_chunk:
                post_best_chunk[post_id] = (chunk_id, score)
            else:
                # If we found a chunk for this post with a higher score, update it
                if score > post_best_chunk[post_id][1]:
                    post_best_chunk[post_id] = (chunk_id, score)

        # Sort posts by score descending
        sorted_posts = sorted(post_best_chunk.items(), key=lambda x: x[1][1], reverse=True)
        
        # Take top limit * 2 candidates for reranking
        candidates = sorted_posts[:limit * 2]
        
        # --- Step 5: Retrieve Content & Rerank ---
        candidate_data = []
        candidate_texts = []
        
        for post_id, (chunk_id, _) in candidates:
            # Fetch chunk body and post title
            cursor.execute("""
                SELECT sc.body as chunk_body, s.title as post_title
                FROM submission_chunks sc
                JOIN submissions s ON sc.post_id = s.post_id
                WHERE sc.post_id = ? AND sc.chunk_id = ?
            """, (post_id, chunk_id))
            row = cursor.fetchone()
            
            if row:
                candidate_data.append({
                    'id': post_id,
                    'chunk_id': chunk_id,
                    'title': row['post_title'],
                    'body': row['chunk_body']
                })
                candidate_texts.append(row['chunk_body'])

        if not candidate_texts:
            return []

        # Rerank Documents
        rerank_results = self.reranker_model.rerank(query, candidate_texts)
        
        # Sort candidate_data based on reranker results
        # rerank_results is a list of dicts with 'index', 'relevance_score', 'document'
        # We need to map back to candidate_data
        reranked_indices = [res['index'] for res in rerank_results]
        top_indices = reranked_indices[:limit]
        
        final_results = []
        
        # --- Step 6: Generate Summary Snippets ---
        for idx in top_indices:
            item = candidate_data[idx]
            full_text = item['body']

            # Remove title from body if first chunk of the post
            if item['chunk_id'] == 1 and full_text.startswith(item['title']):
                full_text = full_text[len(item['title']):].strip()
            
            # Split sentences
            sentences = nltk.sent_tokenize(full_text)
            
            passages = []
            
            if len(sentences) <= 3:
                # If chunk is short, combine all sentences as one passage
                passages = [' '.join(sentences)]
            else:
                # Sliding window of 3 sentences
                for i in range(len(sentences) - 2):
                    window = ' '.join(sentences[i : i + 3])
                    passages.append(window)
            
            # If no passages (empty string), use raw body
            if not passages:
                passages = [full_text]

            # Rerank passages to find the best snippet
            # The reranker expects a query and list of docs
            snippet_results = self.reranker_model.rerank(query, passages)
            
            # Get top snippet
            if snippet_results:
                best_snippet = snippet_results[0]['document']
            else:
                best_snippet = full_text[:200] # Fallback
                
            final_results.append({
                'id': item['id'],
                'title': item['title'],
                'snippet': re.sub(r'\n+', ' ', best_snippet).strip() + ' ...'
            })
        
        # Offload model and clean up
        if not self.persist_in_gpu and self.device == 'cuda':
            self.embed_model.to('cpu')
            self.reranker_model.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()
            
        return final_results

    def get_post(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve details of a specific post including reply count.

        Parameters:
            id (str): The post ID.

        Returns:
            Optional[Dict[str, Any]]: Dictionary of post details or None if not found.
        """
        cursor = self.conn.cursor()
        
        # Get Post Details
        cursor.execute("""
            SELECT post_id as id, created, score, author, title, body, flair 
            FROM submissions 
            WHERE post_id = ?
        """, (id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        result = dict(row)
        
        # Get Reply Count (Direct comments on the post)
        cursor.execute("SELECT COUNT(*) FROM comments WHERE parent_id = ?", (id,))
        count_row = cursor.fetchone()
        result['reply_count'] = count_row[0] if count_row else 0
        
        return result

    def get_replies(self, id: str, offset: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve comments replying to a specific parent (post or comment), 
        sorted by score.

        Parameters:
            id (str): The parent ID (post_id or comment_id).
            offset (int): Pagination offset.
            limit (int): Number of comments to return.

        Returns:
            List[Dict[str, Any]]: List of comment dictionaries.
        """
        cursor = self.conn.cursor()
        
        # Fetch Comments
        cursor.execute("""
            SELECT comment_id as id, parent_id, created, score, author, body 
            FROM comments 
            WHERE parent_id = ? 
            ORDER BY score DESC 
            LIMIT ? OFFSET ?
        """, (id, limit, offset))
        
        rows = cursor.fetchall()
        results = []
        
        for row in rows:
            comment_data = dict(row)
            
            # Get reply count for this specific comment
            cursor.execute("SELECT COUNT(*) FROM comments WHERE parent_id = ?", (comment_data['id'],))
            count_row = cursor.fetchone()
            comment_data['reply_count'] = count_row[0] if count_row else 0
            
            results.append(comment_data)
            
        return results