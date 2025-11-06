#!/usr/bin/env python3
"""
Basic usage examples for the PHEME dataset parser.

This script demonstrates how to:
1. Load a single tweet thread
2. Load multiple threads from the dataset
3. Access tweet, user, and annotation data
4. Extract graph structure (edges) for PyG
"""

from pathlib import Path
from src import (
    load_tweet_thread,
    load_dataset,
    find_all_threads,
    verify_thread,
)


def example_1_load_single_thread():
    """Example 1: Load and explore a single thread."""
    print("=" * 80)
    print("Example 1: Loading a single thread")
    print("=" * 80)
    print()

    # Path to a specific thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")

    # Load the thread
    thread = load_tweet_thread(thread_path)

    print(f"Thread ID: {thread.thread_id}")
    print(f"Rumour type: {thread.annotation.is_rumour}")
    print(f"Category: {thread.annotation.category}")
    print()

    # Access source tweet
    print(f"Source tweet text: {thread.source_tweet.text[:100]}...")
    print(f"Posted by: @{thread.source_tweet.user.screen_name}")
    print(f"Retweets: {thread.source_tweet.retweet_count}")
    print()

    # Access all tweets
    all_tweets = thread.get_all_tweets()
    print(f"Total tweets in thread: {len(all_tweets)}")
    print(f"Reaction tweets: {len(thread.reaction_tweets)}")
    print()

    # Access all users
    all_users = thread.get_all_users()
    print(f"Unique users: {len(all_users)}")
    print()

    # Get graph structure (edges for PyG)
    edges = thread.structure.get_reply_edges()
    print(f"Reply edges (for PyG): {len(edges)}")
    if edges:
        print(f"Sample edges:")
        for parent, child in edges[:3]:
            print(f"  {parent} -> {child}")
    print()


def example_2_load_multiple_threads():
    """Example 2: Load multiple threads from a story."""
    print("=" * 80)
    print("Example 2: Loading multiple threads from a story")
    print("=" * 80)
    print()

    # Load first 5 threads from germanwings crash story
    story_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads")

    threads = load_dataset(
        story_path,
        load_reactions=True,
        max_threads=5,
        skip_errors=True
    )

    print(f"Loaded {len(threads)} threads")
    print()

    # Aggregate statistics
    total_tweets = sum(len(t.get_all_tweets()) for t in threads)
    total_users = len(set(
        user_id
        for thread in threads
        for user_id in thread.get_all_users().keys()
    ))
    total_edges = sum(len(t.structure.get_reply_edges()) for t in threads)

    print(f"Aggregate statistics:")
    print(f"  Total tweets: {total_tweets}")
    print(f"  Unique users: {total_users}")
    print(f"  Total edges: {total_edges}")
    print()


def example_3_iterate_all_threads():
    """Example 3: Iterate through all threads in the dataset."""
    print("=" * 80)
    print("Example 3: Finding all threads in the dataset")
    print("=" * 80)
    print()

    # Find all thread directories
    dataset_root = Path("data/all-rnr-annotated-threads")
    thread_dirs = find_all_threads(dataset_root)

    print(f"Found {len(thread_dirs)} threads in the dataset")
    print()

    # Load first few threads one by one
    print("Loading first 3 threads:")
    for i, thread_dir in enumerate(thread_dirs[:3], 1):
        try:
            thread = load_tweet_thread(thread_dir)
            print(f"{i}. {thread.thread_id}: {len(thread.reaction_tweets)} reactions")
        except Exception as e:
            print(f"{i}. Error loading {thread_dir.name}: {e}")
    print()


def example_4_prepare_for_pyg():
    """Example 4: Prepare data for PyTorch Geometric."""
    print("=" * 80)
    print("Example 4: Preparing data for PyTorch Geometric")
    print("=" * 80)
    print()

    # Load a thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    thread = load_tweet_thread(thread_path)

    print(f"Thread: {thread.thread_id}")
    print()

    # Get all tweets and users
    all_tweets = thread.get_all_tweets()
    all_users = thread.get_all_users()

    # Create mappings from tweet_id/user_id to indices
    tweet_id_to_idx = {tweet_id: idx for idx, tweet_id in enumerate(all_tweets.keys())}
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(all_users.keys())}

    print(f"Created mappings:")
    print(f"  Tweet ID -> Index: {len(tweet_id_to_idx)} mappings")
    print(f"  User ID -> Index: {len(user_id_to_idx)} mappings")
    print()

    # Convert reply edges to indices
    edges = thread.structure.get_reply_edges()
    edge_index_replies = [
        (tweet_id_to_idx.get(parent), tweet_id_to_idx.get(child))
        for parent, child in edges
        if parent in tweet_id_to_idx and child in tweet_id_to_idx
    ]

    print(f"Reply edges (tweet -> tweet): {len(edge_index_replies)}")
    if edge_index_replies:
        print(f"  Sample: {edge_index_replies[:3]}")
    print()

    # Create user -> tweet edges (authorship)
    edge_index_authorship = [
        (user_id_to_idx[tweet.user.id], tweet_id_to_idx[tweet_id])
        for tweet_id, tweet in all_tweets.items()
        if tweet.user.id in user_id_to_idx
    ]

    print(f"Authorship edges (user -> tweet): {len(edge_index_authorship)}")
    if edge_index_authorship:
        print(f"  Sample: {edge_index_authorship[:3]}")
    print()

    # Feature extraction example
    print("Feature extraction examples:")
    print()

    # Tweet features
    sample_tweet_id = list(all_tweets.keys())[0]
    sample_tweet = all_tweets[sample_tweet_id]

    print(f"Tweet features for {sample_tweet_id}:")
    print(f"  - Text: {sample_tweet.text[:50]}...")
    print(f"  - Text length: {len(sample_tweet.text)}")
    print(f"  - Retweet count: {sample_tweet.retweet_count}")
    print(f"  - Favorite count: {sample_tweet.favorite_count}")
    print(f"  - Number of hashtags: {len(sample_tweet.entities.hashtags)}")
    print(f"  - Number of URLs: {len(sample_tweet.entities.urls)}")
    print(f"  - Number of mentions: {len(sample_tweet.entities.user_mentions)}")
    print(f"  - Created at: {sample_tweet.created_at}")
    print()

    # User features
    sample_user_id = list(all_users.keys())[0]
    sample_user = all_users[sample_user_id]

    print(f"User features for {sample_user.screen_name}:")
    print(f"  - Followers: {sample_user.followers_count}")
    print(f"  - Friends: {sample_user.friends_count}")
    print(f"  - Statuses: {sample_user.statuses_count}")
    print(f"  - Verified: {sample_user.verified}")
    print(f"  - Created at: {sample_user.created_at}")
    print()

    print("Next steps for PyG:")
    print("  1. Convert text to embeddings (e.g., using sentence-transformers)")
    print("  2. Create feature tensors from numerical features")
    print("  3. Build HeteroData object with all node/edge types")
    print("  4. Create temporal positional encodings from timestamps")
    print()


def example_5_verification():
    """Example 5: Verify thread integrity."""
    print("=" * 80)
    print("Example 5: Verifying thread integrity")
    print("=" * 80)
    print()

    # Load a thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    thread = load_tweet_thread(thread_path)

    # Verify it
    results = verify_thread(thread)

    print(f"Thread: {thread.thread_id}")
    print(f"Valid: {results['valid']}")
    print()

    print("Statistics:")
    for key, value in results['stats'].items():
        print(f"  {key}: {value}")
    print()

    if results['errors']:
        print("Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    else:
        print("No errors found!")
    print()

    if results['warnings']:
        print("Warnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    else:
        print("No warnings!")
    print()


def main():
    """Run all examples."""
    example_1_load_single_thread()
    print("\n")

    example_2_load_multiple_threads()
    print("\n")

    example_3_iterate_all_threads()
    print("\n")

    example_4_prepare_for_pyg()
    print("\n")

    example_5_verification()


if __name__ == "__main__":
    main()
