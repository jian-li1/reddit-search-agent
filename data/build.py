import argparse
import os
import pandas as pd
import sqlite3
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoModel
import transformers
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from typing import List, Dict, Any, Generator

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing directory and subreddit.
    """
    parser = argparse.ArgumentParser(description='Build retrieval database and embeddings for Reddit data.')
    parser.add_argument(
        '-d', '--directory',
        dest='directory',
        type=str,
        required=True,
        help='Directory path containing the submission and comment CSV files'
    )
    parser.add_argument(
        '--subreddit',
        type=str,
        required=True,
        help='Subreddit name'
    )
    parser.add_argument(
        '--db', '--database',
        dest='db_path',
        type=str,
        required=True,
        help='Output path for the SQLite database'
    )
    parser.add_argument(
        '--em', '--embeddings',
        dest='embed_path',
        type=str,
        required=True,
        help='Output path for the embeddings JSONL file'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1024,
        help='Chunk size for text splitting (default: 1024)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=256,
        help='Chunk overlap for text splitting (default: 256)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for embedding generation (default: 4)'
    )

    return parser.parse_args()

def load_data(directory: str, subreddit: str) -> Dict[str, pd.DataFrame]:
    """
    Load submission and comment CSV files into pandas DataFrames with specified dtypes.

    Parameters:
        directory (str): Path to the directory containing CSV files.
        subreddit (str): Name of the subreddit to construct filenames.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing 'submissions' and 'comments' DataFrames.
    """
    # Define file paths
    path = os.path.join(directory, subreddit)
    submission_path = f'{path}_filtered_submissions.csv'
    comment_path = f'{path}_filtered_comments.csv'

    # Load submissions and comments into dataframes
    print(f'Loading submissions from {submission_path}...')
    submission_df = pd.read_csv(submission_path, dtype={
        'author': 'string',
        'title': 'string',
        'score': 'int',
        'created': 'string',
        'body': 'string',
        'id': 'string',
        'link_flair_text': 'string'
    })
    submission_df['created'] = pd.to_datetime(submission_df['created'], format='%Y-%m-%d %H:%M:%S')
    
    print(f'Loading comments from {comment_path}...')
    comment_df = pd.read_csv(comment_path, dtype={
        'author': 'string',
        'score': 'int',
        'created': 'string',
        'body': 'string',
        'id': 'string',
        'parent_id': 'string'
    })
    comment_df['created'] = pd.to_datetime(comment_df['created'], format='%Y-%m-%d %H:%M:%S')

    return {'submissions': submission_df, 'comments': comment_df}

def generate_chunks(
    submissions_df: pd.DataFrame,
    chunk_size: int = 1024,
    chunk_overlap: int = 256
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield chunks from submission bodies, prepending titles to the first chunk.

    Parameters:
        submissions_df (pd.DataFrame): The dataframe containing submission data.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Yields:
        Dict[str, Any]: A dictionary representing a chunk (post_id, chunk_id, text).
    """
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=['\n\n', '\n', ' ', '']
    )

    for _, row in submissions_df.iterrows():
        # Concatenate title and body for chunking
        body_text = str(row['body']) if not pd.isna(row['body']) else ''
        title_text = str(row['title']) if not pd.isna(row['title']) else ''
        submission_text = title_text + '\n\n' + body_text
        post_id = row['id']
        subreddit = row['subreddit']

        chunks = text_splitter.split_text(submission_text)

        # If body is empty, create a dummy chunk with just title or empty string
        # if not chunks:
        #     chunks = ['']

        for i, chunk_text in enumerate(chunks):
            final_text = chunk_text
            
            # Prepend title to the first chunk
            # if i == 0:
            #     final_text = f"{title_text}\n\n{chunk_text}"

            yield {
                'post_id': post_id,
                'chunk_id': i + 1, # 1-based index per prompt
                'text': final_text,
                'subreddit': subreddit
            }

def process_embeddings_and_save(
    chunks: List[Dict[str, Any]], 
    output_path: str,
    batch_size: int = 4
) -> None:
    """
    Generate embeddings for chunks and save them to a JSONL file.

    Parameters:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries containing text.
        output_path (str): Path to the output JSONL file.
        batch_size (int): Number of texts to embed at once.
    """
    print('Initializing Embedding Model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v4',
        trust_remote_code=True,
        torch_dtype=(torch.float16 if device == 'cuda' else torch.float32),
    )
    model.to(device)
    model.eval()

    transformers.logging.set_verbosity_error()
    print(f'Generating embeddings and writing to {output_path}...')
    
    # with open(output_path, 'w', encoding='utf-8') as f:
    # Process in batches to manage memory
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Progress", unit="batch"):
        batch = chunks[i : i + batch_size]
        texts = [c['text'] for c in batch]

        # Jina V4 API specific call
        embeddings = model.encode_text(
            texts=texts,
            task='retrieval',
            prompt_name='passage',
        )

        # Write to file
        for j, embedding_vector in enumerate(embeddings):
            chunk_data = batch[j]
            record = {
                'id': chunk_data['post_id'],
                'chunk_id': chunk_data['chunk_id'],
                'subreddit': chunk_data['subreddit'],
                'embeddings': embedding_vector.cpu().tolist()
            }
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        
        torch.cuda.empty_cache()
        gc.collect()

def setup_database(db_path: str) -> None:
    """
    Create SQLite tables including FTS virtual table for chunks.

    Parameters:
        db_path (str): Path to the SQLite database file.
    """
    print(f'Creating database at {db_path}...')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute('PRAGMA foreign_keys = ON;')

    # 1. Create Submissions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY,
            post_id TEXT UNIQUE,
            created TEXT,
            score INTEGER,
            author TEXT,
            title TEXT,
            body TEXT,
            flair TEXT,
            subreddit TEXT
        );
    """)

    # 2. Create Comments Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY,
            parent_id TEXT,
            comment_id TEXT UNIQUE,
            created TEXT,
            score INTEGER,
            author TEXT,
            body TEXT,
            subreddit TEXT
        );
    """)

    # 3. Create submission_chunks as the FTS5 Virtual Table
    # We use UNINDEXED for metadata columns so they are stored but not text-searchable
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS submission_chunks USING fts5(
            post_id UNINDEXED,
            chunk_id UNINDEXED,
            body,
            subreddit UNINDEXED
        );
    """)
    
    # Triggers to keep FTS index in sync with standard table (Optional but good practice)
    # However, for this script, we are just inserting once, so we can insert into both manually.
    
    conn.commit()
    conn.close()

def insert_data_to_db(
    db_path: str, 
    submissions: pd.DataFrame, 
    comments: pd.DataFrame, 
    chunks: List[Dict[str, Any]]
) -> None:
    """
    Insert data into SQLite tables.

    Parameters:
        db_path (str): Path to the database.
        submissions (pd.DataFrame): Processed submissions dataframe.
        comments (pd.DataFrame): Processed comments dataframe.
        chunks (List[Dict[str, Any]]): List of chunk dictionaries.
    """
    print('Inserting data into database...')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Helper function to execute inserts in batches
    def execute_batch_commit(query: str, data: List[tuple], batch_size: int = 10000) -> None:
        total = len(data)
        for i in range(0, total, batch_size):
            batch = data[i : i + batch_size]
            cursor.executemany(query, batch)
            conn.commit()
            print(f'Committed {len(batch)} records ({min(i + batch_size, total)}/{total})...')

    # Insert Submissions
    submission_data = []
    for _, row in submissions.iterrows():
        submission_data.append((
            str(row['id']),
            str(row['created']),
            int(row['score']),
            str(row['author']),
            str(row['title']),
            str(row['body']),
            str(row['link_flair_text']),
            str(row['subreddit'])
        ))
    
    execute_batch_commit("""
        INSERT OR IGNORE INTO submissions (post_id, created, score, author, title, body, flair, subreddit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, submission_data)

    # Insert Comments
    comment_data = []
    for i, row in comments.iterrows():
        comment_data.append((
            str(row['parent_id']),
            str(row['id']),
            str(row['created']),
            int(row['score']),
            str(row['author']),
            str(row['body']),
            str(row['subreddit'])
        ))
    
    execute_batch_commit("""
        INSERT OR IGNORE INTO comments (parent_id, comment_id, created, score, author, body, subreddit)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, comment_data)

    # Insert Chunks directly into the FTS table
    chunk_data = []
    for chunk in chunks:
        chunk_data.append((
            str(chunk['post_id']),
            int(chunk['chunk_id']),
            str(chunk['text']),
            str(chunk['subreddit'])
        ))

    # Note: We insert directly into the virtual table name
    execute_batch_commit("""
        INSERT INTO submission_chunks (post_id, chunk_id, body, subreddit)
        VALUES (?, ?, ?, ?)
    """, chunk_data)

    conn.commit()
    conn.close()
    print('Database population complete.')

def main():
    args = parse_args()
    directory = args.directory
    subreddit = args.subreddit

    # 1. Load Data
    data = load_data(directory, subreddit)
    
    # 2. Chunk Data
    print('Chunking submission text...')
    chunks_generator = generate_chunks(data['submissions'], args.chunk_size, args.chunk_overlap)
    all_chunks = list(chunks_generator) # Manifest list to use for both embedding and DB
    
    # 3. Create Database
    db_path = args.db_path
    setup_database(args.db_path)

    # 4. Insert Records
    insert_data_to_db(db_path, data['submissions'], data['comments'], all_chunks)

    # 5. Generate Embeddings and Save JSONL
    jsonl_path = args.embed_path
    process_embeddings_and_save(all_chunks, jsonl_path, args.batch_size)

    print('Build process finished successfully.')

if __name__ == '__main__':
    main()