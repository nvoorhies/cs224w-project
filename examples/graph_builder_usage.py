#!/usr/bin/env python3
"""
Graph Builder and PyG Dataset Usage Examples.

This script demonstrates how to:
1. Build graphs from individual threads
2. Use the PyG Dataset class
3. Create train/val/test splits
4. Use DataLoaders for batching
5. Inspect graph structure and features
"""

from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

from src import (
    load_tweet_thread,
    thread_to_graph,
    print_graph_statistics,
    PHEMEDataset,
)


def example_1_single_thread_to_graph():
    """Example 1: Convert a single thread to a graph."""
    print("=" * 80)
    print("Example 1: Single Thread to Graph")
    print("=" * 80)
    print()

    # Load a thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    print(f"Loading thread from: {thread_path.name}")
    thread = load_tweet_thread(thread_path)

    print(f"Thread has {len(thread.get_all_tweets())} tweets and {len(thread.get_all_users())} users")
    print()

    # Convert to graph with different user edge strategies
    print("Building graph with user->user edges based on replies...")
    graph = thread_to_graph(thread, user_edge_type="replies")

    print_graph_statistics(graph)

    # Access features
    print("Feature shapes:")
    print(f"  Tweet features: {graph['tweet'].x.shape}")
    print(f"  User features: {graph['user'].x.shape}")
    print()

    # Access edges
    print("Sample reply edges (first 5):")
    reply_edges = graph['tweet', 'replies_to', 'tweet'].edge_index
    for i in range(min(5, reply_edges.shape[1])):
        src, dst = reply_edges[:, i]
        print(f"  Tweet {src.item()} -> Tweet {dst.item()}")
    print()


def example_2_different_edge_types():
    """Example 2: Compare different user edge strategies."""
    print("=" * 80)
    print("Example 2: Different User Edge Strategies")
    print("=" * 80)
    print()

    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    thread = load_tweet_thread(thread_path)

    strategies = ["none", "mentions", "replies", "both"]

    for strategy in strategies:
        graph = thread_to_graph(thread, user_edge_type=strategy)

        # Count user->user edges
        if ('user', 'interacts_with', 'user') in graph.edge_types:
            num_user_edges = graph['user', 'interacts_with', 'user'].edge_index.shape[1]
        else:
            num_user_edges = 0

        print(f"Strategy '{strategy:10s}': {num_user_edges:3d} user->user edges")

    print()


def example_3_pyg_dataset():
    """Example 3: Using PHEMEDataset."""
    print("=" * 80)
    print("Example 3: PHEMEDataset")
    print("=" * 80)
    print()

    # Create dataset
    print("Creating dataset...")
    dataset = PHEMEDataset(
        root='data/all-rnr-annotated-threads',
        stories=['germanwings-crash-all-rnr-threads'],
        max_threads=10,
        user_edge_type='replies',
        normalize_features=False
    )

    print(f"Dataset: {len(dataset)} threads")
    print()

    # Access graphs
    print("First 3 graphs:")
    for i in range(min(3, len(dataset))):
        graph = dataset[i]
        num_tweets = graph['tweet'].x.shape[0]
        num_users = graph['user'].x.shape[0]
        print(f"  {i}. Thread {graph.thread_id}: {num_tweets} tweets, {num_users} users")
        print(f"      Rumour: {graph.y_rumour}, Misinfo: {graph.y_misinformation}, True: {graph.y_true}")
    print()


def example_4_train_val_test_splits():
    """Example 4: Create train/val/test splits."""
    print("=" * 80)
    print("Example 4: Train/Val/Test Splits")
    print("=" * 80)
    print()

    # Create full dataset
    dataset = PHEMEDataset(
        root='data/all-rnr-annotated-threads',
        stories=['germanwings-crash-all-rnr-threads'],
        max_threads=30,
        user_edge_type='replies'
    )

    print(f"Full dataset: {len(dataset)} threads")
    print()

    # Create splits
    train_dataset, val_dataset, test_dataset = dataset.get_split_datasets()

    print("Split sizes:")
    print(f"  Train: {len(train_dataset):3d} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"  Val:   {len(val_dataset):3d} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"  Test:  {len(test_dataset):3d} ({len(test_dataset)/len(dataset)*100:.1f}%)")
    print()

    # Load one from each
    print("Sample from each split:")
    print(f"  Train: Thread {train_dataset[0].thread_id}")
    print(f"  Val:   Thread {val_dataset[0].thread_id}")
    print(f"  Test:  Thread {test_dataset[0].thread_id}")
    print()


def example_5_dataloader():
    """Example 5: Using PyG DataLoader for batching."""
    print("=" * 80)
    print("Example 5: DataLoader for Batching")
    print("=" * 80)
    print()

    # Create dataset
    dataset = PHEMEDataset(
        root='data/all-rnr-annotated-threads',
        stories=['germanwings-crash-all-rnr-threads'],
        max_threads=20,
        user_edge_type='replies'
    )

    # Create DataLoader
    print("Creating DataLoader with batch_size=4...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Number of batches: {len(loader)}")
    print()

    # Iterate through first batch
    print("First batch:")
    batch = next(iter(loader))

    print(f"  Batch type: {type(batch)}")
    print(f"  Tweet nodes: {batch['tweet'].x.shape}")
    print(f"  User nodes: {batch['user'].x.shape}")
    print(f"  Tweet batch assignment: {batch['tweet'].batch.shape}")

    # Check how many graphs are in the batch
    num_graphs = batch['tweet'].batch.max().item() + 1
    print(f"  Number of graphs in batch: {num_graphs}")
    print()


def example_6_feature_inspection():
    """Example 6: Inspect features in detail."""
    print("=" * 80)
    print("Example 6: Feature Inspection")
    print("=" * 80)
    print()

    # Load a graph
    dataset = PHEMEDataset(
        root='data/all-rnr-annotated-threads',
        stories=['germanwings-crash-all-rnr-threads'],
        max_threads=1,
        include_temporal_encoding=True,
        temporal_encoding_dim=16
    )

    graph = dataset[0]

    print(f"Thread: {graph.thread_id}")
    print()

    # Tweet features
    tweet_features = graph['tweet'].x
    print(f"Tweet features shape: {tweet_features.shape}")
    print(f"  Number of tweets: {tweet_features.shape[0]}")
    print(f"  Feature dimension: {tweet_features.shape[1]}")
    print(f"    - Base features: 15")
    print(f"    - Temporal features: 4")
    print(f"    - Positional encoding: 16")
    print()

    # User features
    user_features = graph['user'].x
    print(f"User features shape: {user_features.shape}")
    print(f"  Number of users: {user_features.shape[0]}")
    print(f"  Feature dimension: {user_features.shape[1]}")
    print()

    # Sample values
    print("Sample tweet feature vector (first tweet):")
    print(f"  Mean: {tweet_features[0].mean():.4f}")
    print(f"  Std:  {tweet_features[0].std():.4f}")
    print(f"  Min:  {tweet_features[0].min():.4f}")
    print(f"  Max:  {tweet_features[0].max():.4f}")
    print()


def example_7_edge_types():
    """Example 7: Explore all edge types."""
    print("=" * 80)
    print("Example 7: Edge Types")
    print("=" * 80)
    print()

    # Load a graph with all edge types
    dataset = PHEMEDataset(
        root='data/all-rnr-annotated-threads',
        stories=['germanwings-crash-all-rnr-threads'],
        max_threads=1,
        user_edge_type='both'  # Include both mention and reply edges
    )

    graph = dataset[0]

    print(f"Thread: {graph.thread_id}")
    print()

    print("All edge types:")
    for edge_type in graph.edge_types:
        src, relation, dst = edge_type
        num_edges = graph[edge_type].edge_index.shape[1]
        print(f"  ({src:10s}, {relation:15s}, {dst:10s}): {num_edges:4d} edges")
    print()

    # Analyze connectivity
    reply_edges = graph['tweet', 'replies_to', 'tweet'].edge_index
    print(f"Tweet reply graph:")
    print(f"  Edges: {reply_edges.shape[1]}")
    if reply_edges.shape[1] > 0:
        print(f"  Source tweets (unique): {reply_edges[0].unique().shape[0]}")
        print(f"  Target tweets (unique): {reply_edges[1].unique().shape[0]}")
    print()


def main():
    """Run all examples."""
    example_1_single_thread_to_graph()
    print("\n")

    example_2_different_edge_types()
    print("\n")

    example_3_pyg_dataset()
    print("\n")

    example_4_train_val_test_splits()
    print("\n")

    example_5_dataloader()
    print("\n")

    example_6_feature_inspection()
    print("\n")

    example_7_edge_types()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
