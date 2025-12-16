import pandas as pd
import argparse
import os

def check_depth(idx: int) -> None:
    """
    Get depth level of each comment. Top-level comment begins at 1

    Parameters:
        idx (int): Index of the comment in the comment dataframe
    """
    # Get parent id and index
    parent_id = comment_df.loc[idx, 'parent_id']
    parent_idx = comment_df.loc[idx, 'parent_idx']

    # Parent is a submission
    if parent_id.startswith('t3'):
        comment_df.loc[idx, 'depth'] = 1
    else:
        parent_depth = comment_df.loc[parent_idx, 'depth']
        comment_df.loc[idx, 'depth'] = parent_depth + 1

def check_comment(idx: int) -> None:
    """
    Determines if a comment is detached from a conversation tree.
    Assumes parent comment is already checked.

    Parameters:
        idx (int): Index of the comment in the comment dataframe
    """
    # Get parent id and index
    parent_id = comment_df.loc[idx, 'parent_id']
    parent_idx = comment_df.loc[idx, 'parent_idx']

    has_parent = pd.notna(parent_idx)

    # Parent is a submission or confirmed to be an attached comment
    # print(idx, parent_idx)
    if parent_id in submission_ids or has_parent and comment_df.loc[parent_idx, 'drop'] == False:
        # Mark all comment ids in the stack to not drop
        comment_df.at[idx, 'drop'] = False
    # No parent submission or parent is confirmed to be a detached comment
    else:
        # Mark all comment ids in the stack to drop
        comment_df.at[idx, 'drop'] = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory path containing the submission and comment CSV files')
    parser.add_argument('--subreddit', type=str, required=True, help='Subreddit name')
    parser.add_argument('--min-score', type=int, default=1, help='Minimum score of submission/comment (default: 1)')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum depth of each discussion thread (default: 3)')
    # parser.add_argument('--include-mod', type=bool, default=False, help='Include submissions/comments from moderators')
    
    args = parser.parse_args()
    subreddit = args.subreddit
    directory = args.directory
    path = os.path.join(directory, subreddit)
    min_score = args.min_score
    max_depth = args.max_depth

    # Load submissions and comments into dataframes
    submission_df = pd.read_csv(f'{path}_submissions.csv', dtype={
        'author': 'string',
        'title': 'string',
        'text': 'string',
        'id': 'string',
        'link_flair_text': 'string',
        'distinguished': 'string',
        'subreddit': 'string',
        'link': 'string'
    }).rename(columns={'text': 'body'}).drop(columns=['subreddit', 'link'])
    submission_df['created'] = pd.to_datetime(submission_df['created'], format='%Y-%m-%d %H:%M:%S')

    comment_df = pd.read_csv(f'{path}_comments.csv', dtype={
        'author': 'string',
        'body': 'string',
        'name': 'string',
        'parent_id': 'string',
        'distinguished': 'string',
        'subreddit': 'string',
        'link': 'string'
    }).rename(columns={'name': 'id'}).drop(columns=['subreddit', 'link'])
    comment_df['created'] = pd.to_datetime(comment_df['created'], format='%Y-%m-%d %H:%M:%S')

    # Filter submissions and comments (score >= min_score, not moderator, not removed/deleted, and not empty)
    submission_df = submission_df[
        (submission_df['score'] >= min_score) & 
        (submission_df['distinguished'].ne('moderator').fillna(True)) &
        (~(submission_df['body'].isin(['[removed]', '[deleted]']))) &
        (submission_df['body'].notna())
    ]
    comment_df = comment_df[
        (comment_df['score'] >= min_score) & 
        (comment_df['distinguished'].ne('moderator').fillna(True)) &
        (~(comment_df['body']).isin(['[removed]', '[deleted]'])) &
        (comment_df['body'].notna())
    ]

    # Sort the comments by chronological order
    # Usually the dataset is already sorted
    comment_df = comment_df.sort_values(by='created', kind='stable')

    # Reset index of dataframes after dropping rows
    submission_df = submission_df.reset_index(drop=True)
    comment_df = comment_df.reset_index(drop=True)

    # Map each parent id to its index (will be efficient for parent lookup when pruning)
    id_to_idx = pd.Series(comment_df.index.values, index=comment_df['id'])
    comment_df['parent_idx'] = comment_df['parent_id'].map(id_to_idx).astype('Int64')

    # Get all submission ids
    submission_ids = set(submission_df['id'])

    # Create label to drop detached comments
    comment_df['drop'] = pd.NA
    comment_df['drop'] = comment_df['drop'].astype('boolean')

    # Iterate through each comment and drop all detached comments
    # Since the comments in the dataset are usually in chronological order,
    # this iteration implies that the parent comment of each comment is already processed upfront
    for idx, row in comment_df.iterrows():
        check_comment(idx)

    # Drop all detached comments
    comment_df = comment_df[comment_df['drop'] == False].drop(columns='drop')

    comment_df['depth'] = pd.NA
    comment_df['depth'] = comment_df['depth'].astype('Int64')

    # Iterate each comment and get its depth level
    for idx, row in comment_df.iterrows():
        check_depth(idx)
    
    # Only keep messages with a depth level of at most `max_depth`
    comment_df = comment_df[comment_df['depth'] <= max_depth]
    comment_df = comment_df.reset_index(drop=True)

    # Write to CSV
    submission_df.drop(columns=['distinguished']).to_csv(f'{path}_filtered_submissions.csv', index=False)
    comment_df.drop(columns=['parent_idx', 'distinguished', 'depth']).to_csv(f'{path}_filtered_comments.csv', index=False)