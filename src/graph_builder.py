"""
Graph builder for converting PHEME threads to PyTorch Geometric HeteroData.

This module provides functions to construct heterogeneous graphs from
tweet threads, including node features, edge indices, and temporal encodings.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from torch_geometric.data import HeteroData

from .models import TweetThread, Tweet, TwitterUser
from .features import (
    extract_all_tweet_features,
    extract_all_user_features,
    parse_twitter_timestamp,
)


# ============================================================================
# Edge Construction
# ============================================================================

def build_reply_edges(
    tweets: Dict[str, Tweet],
    structure_edges: List[tuple[str, str]],
    tweet_id_to_idx: Dict[str, int]
) -> torch.Tensor:
    """
    Build tweet->tweet reply edges.

    Args:
        tweets: Dictionary of tweet_id -> Tweet
        structure_edges: List of (parent_id, child_id) from structure
        tweet_id_to_idx: Mapping from tweet_id to node index

    Returns:
        Edge index tensor [2, num_edges]
    """
    edges = []

    for parent_id, child_id in structure_edges:
        if parent_id in tweet_id_to_idx and child_id in tweet_id_to_idx:
            parent_idx = tweet_id_to_idx[parent_id]
            child_idx = tweet_id_to_idx[child_id]
            edges.append([parent_idx, child_idx])

    if not edges:
        # Return empty tensor with correct shape
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_authorship_edges(
    tweets: Dict[str, Tweet],
    tweet_id_to_idx: Dict[str, int],
    user_id_to_idx: Dict[int, int]
) -> torch.Tensor:
    """
    Build user->tweet authorship edges.

    Args:
        tweets: Dictionary of tweet_id -> Tweet
        tweet_id_to_idx: Mapping from tweet_id to node index
        user_id_to_idx: Mapping from user_id to node index

    Returns:
        Edge index tensor [2, num_edges] (user_idx, tweet_idx)
    """
    edges = []

    for tweet_id, tweet in tweets.items():
        if tweet_id in tweet_id_to_idx and tweet.user.id in user_id_to_idx:
            user_idx = user_id_to_idx[tweet.user.id]
            tweet_idx = tweet_id_to_idx[tweet_id]
            edges.append([user_idx, tweet_idx])

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_tweet_to_author_edges(
    tweets: Dict[str, Tweet],
    tweet_id_to_idx: Dict[str, int],
    user_id_to_idx: Dict[int, int]
) -> torch.Tensor:
    """
    Build tweet->user edges (reverse of authorship).

    Args:
        tweets: Dictionary of tweet_id -> Tweet
        tweet_id_to_idx: Mapping from tweet_id to node index
        user_id_to_idx: Mapping from user_id to node index

    Returns:
        Edge index tensor [2, num_edges] (tweet_idx, user_idx)
    """
    edges = []

    for tweet_id, tweet in tweets.items():
        if tweet_id in tweet_id_to_idx and tweet.user.id in user_id_to_idx:
            tweet_idx = tweet_id_to_idx[tweet_id]
            user_idx = user_id_to_idx[tweet.user.id]
            edges.append([tweet_idx, user_idx])

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_mention_edges(
    tweets: Dict[str, Tweet],
    users: Dict[int, TwitterUser],
    user_id_to_idx: Dict[int, int]
) -> torch.Tensor:
    """
    Build user->user edges based on mentions.

    If user A posts a tweet mentioning user B, create edge A->B.

    Args:
        tweets: Dictionary of tweet_id -> Tweet
        users: Dictionary of user_id -> TwitterUser
        user_id_to_idx: Mapping from user_id to node index

    Returns:
        Edge index tensor [2, num_edges] (from_user_idx, to_user_idx)
    """
    edges = []

    for tweet in tweets.values():
        author_id = tweet.user.id

        if author_id not in user_id_to_idx:
            continue

        author_idx = user_id_to_idx[author_id]

        # Add edge for each mentioned user
        for mention in tweet.entities.user_mentions:
            mentioned_id = mention.id

            if mentioned_id in user_id_to_idx:
                mentioned_idx = user_id_to_idx[mentioned_id]
                edges.append([author_idx, mentioned_idx])

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    # Remove duplicates
    edges = list(set(tuple(e) for e in edges))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_reply_based_user_edges(
    tweets: Dict[str, Tweet],
    structure_edges: List[tuple[str, str]],
    user_id_to_idx: Dict[int, int]
) -> torch.Tensor:
    """
    Build user->user edges based on replies.

    If user A replies to user B's tweet, create edge A->B.

    Args:
        tweets: Dictionary of tweet_id -> Tweet
        structure_edges: List of (parent_tweet_id, child_tweet_id)
        user_id_to_idx: Mapping from user_id to node index

    Returns:
        Edge index tensor [2, num_edges] (from_user_idx, to_user_idx)
    """
    edges = []

    for parent_tweet_id, child_tweet_id in structure_edges:
        if parent_tweet_id not in tweets or child_tweet_id not in tweets:
            continue

        parent_tweet = tweets[parent_tweet_id]
        child_tweet = tweets[child_tweet_id]

        parent_user_id = parent_tweet.user.id
        child_user_id = child_tweet.user.id

        # Don't create self-loops
        if parent_user_id == child_user_id:
            continue

        if parent_user_id in user_id_to_idx and child_user_id in user_id_to_idx:
            parent_user_idx = user_id_to_idx[parent_user_id]
            child_user_idx = user_id_to_idx[child_user_id]
            # Edge from replier (child) to replied-to (parent)
            edges.append([child_user_idx, parent_user_idx])

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    # Remove duplicates
    edges = list(set(tuple(e) for e in edges))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ============================================================================
# Graph Construction
# ============================================================================

class PHEMEGraphBuilder:
    """
    Builder for constructing PyG HeteroData graphs from PHEME threads.
    """

    def __init__(
        self,
        include_temporal_encoding: bool = True,
        temporal_encoding_dim: int = 16,
        normalize_features: bool = False,
        user_edge_type: Literal["mentions", "replies", "both", "none"] = "replies"
    ):
        """
        Initialize the graph builder.

        Args:
            include_temporal_encoding: Whether to include temporal positional encoding
            temporal_encoding_dim: Dimension of temporal positional encoding
            normalize_features: Whether to normalize features
            user_edge_type: How to construct user->user edges:
                - "mentions": Based on @mentions
                - "replies": Based on reply relationships
                - "both": Combine mentions and replies
                - "none": No user->user edges
        """
        self.include_temporal_encoding = include_temporal_encoding
        self.temporal_encoding_dim = temporal_encoding_dim
        self.normalize_features = normalize_features
        self.user_edge_type = user_edge_type

    def build_graph(
        self,
        thread: TweetThread,
        add_reverse_edges: bool = True
    ) -> HeteroData:
        """
        Build a heterogeneous graph from a tweet thread.

        Args:
            thread: TweetThread object
            add_reverse_edges: Whether to add reverse edges for message passing

        Returns:
            HeteroData object with nodes, edges, and features
        """
        data = HeteroData()

        # Get all tweets and users
        all_tweets = thread.get_all_tweets()
        all_users = thread.get_all_users()
        structure_edges = thread.structure.get_reply_edges()

        # Create ID mappings
        tweet_ids = list(all_tweets.keys())
        user_ids = list(all_users.keys())

        tweet_id_to_idx = {tid: i for i, tid in enumerate(tweet_ids)}
        user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

        # Extract features
        reference_time = parse_twitter_timestamp(thread.source_tweet.created_at)

        tweet_features_dict, tweet_dim = extract_all_tweet_features(
            all_tweets,
            reference_time=reference_time,
            include_temporal_encoding=self.include_temporal_encoding,
            temporal_encoding_dim=self.temporal_encoding_dim,
            edges=structure_edges
        )

        user_features_dict, user_dim = extract_all_user_features(all_users)

        # Convert to tensors (maintaining order)
        tweet_features = torch.tensor(
            np.stack([tweet_features_dict[tid] for tid in tweet_ids]),
            dtype=torch.float
        )
        user_features = torch.tensor(
            np.stack([user_features_dict[uid] for uid in user_ids]),
            dtype=torch.float
        )

        # Normalize if requested
        if self.normalize_features:
            tweet_features = self._normalize(tweet_features)
            user_features = self._normalize(user_features)

        # Add node features
        data['tweet'].x = tweet_features
        data['user'].x = user_features

        # Store ID mappings as metadata
        data['tweet'].tweet_ids = tweet_ids
        data['user'].user_ids = user_ids

        # Build edges
        # 1. Tweet -> Tweet (replies)
        reply_edge_index = build_reply_edges(
            all_tweets, structure_edges, tweet_id_to_idx
        )
        data['tweet', 'replies_to', 'tweet'].edge_index = reply_edge_index

        if add_reverse_edges:
            data['tweet', 'replied_by', 'tweet'].edge_index = reply_edge_index.flip(0)

        # 2. User -> Tweet (authorship)
        authorship_edge_index = build_authorship_edges(
            all_tweets, tweet_id_to_idx, user_id_to_idx
        )
        data['user', 'posts', 'tweet'].edge_index = authorship_edge_index

        # 3. Tweet -> User (reverse authorship)
        tweet_to_author_edge_index = build_tweet_to_author_edges(
            all_tweets, tweet_id_to_idx, user_id_to_idx
        )
        data['tweet', 'posted_by', 'user'].edge_index = tweet_to_author_edge_index

        # 4. User -> User (based on selected strategy)
        if self.user_edge_type != "none":
            user_edges = []

            if self.user_edge_type in ["mentions", "both"]:
                mention_edges = build_mention_edges(
                    all_tweets, all_users, user_id_to_idx
                )
                if mention_edges.shape[1] > 0:
                    user_edges.append(mention_edges)

            if self.user_edge_type in ["replies", "both"]:
                reply_edges = build_reply_based_user_edges(
                    all_tweets, structure_edges, user_id_to_idx
                )
                if reply_edges.shape[1] > 0:
                    user_edges.append(reply_edges)

            if user_edges:
                # Combine and deduplicate
                combined_edges = torch.cat(user_edges, dim=1)
                unique_edges = torch.unique(combined_edges, dim=1)
                data['user', 'interacts_with', 'user'].edge_index = unique_edges

                if add_reverse_edges:
                    data['user', 'interacted_by', 'user'].edge_index = unique_edges.flip(0)

        # Add thread metadata
        data.thread_id = thread.thread_id
        data.num_nodes_dict = {
            'tweet': len(tweet_ids),
            'user': len(user_ids)
        }

        return data

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize features (zero mean, unit variance)."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        std[std == 0] = 1.0  # Avoid division by zero
        return (features - mean) / std


# ============================================================================
# Convenience Functions
# ============================================================================

def thread_to_graph(
    thread: TweetThread,
    include_temporal_encoding: bool = True,
    temporal_encoding_dim: int = 16,
    normalize_features: bool = False,
    user_edge_type: Literal["mentions", "replies", "both", "none"] = "replies",
    add_reverse_edges: bool = True
) -> HeteroData:
    """
    Convert a TweetThread to a PyG HeteroData graph.

    This is a convenience function that creates a builder and builds the graph.

    Args:
        thread: TweetThread object
        include_temporal_encoding: Whether to include temporal positional encoding
        temporal_encoding_dim: Dimension of temporal positional encoding
        normalize_features: Whether to normalize features
        user_edge_type: How to construct user->user edges
        add_reverse_edges: Whether to add reverse edges

    Returns:
        HeteroData object

    Example:
        >>> from src import load_tweet_thread
        >>> thread = load_tweet_thread(thread_path)
        >>> graph = thread_to_graph(thread)
        >>> print(graph)
    """
    builder = PHEMEGraphBuilder(
        include_temporal_encoding=include_temporal_encoding,
        temporal_encoding_dim=temporal_encoding_dim,
        normalize_features=normalize_features,
        user_edge_type=user_edge_type
    )

    return builder.build_graph(thread, add_reverse_edges=add_reverse_edges)


def print_graph_statistics(data: HeteroData) -> None:
    """
    Print statistics about a HeteroData graph.

    Args:
        data: HeteroData object
    """
    print("=" * 80)
    print("Graph Statistics")
    print("=" * 80)
    print()

    # Node statistics
    print("Nodes:")
    for node_type in data.node_types:
        num_nodes = data[node_type].x.shape[0]
        num_features = data[node_type].x.shape[1]
        print(f"  {node_type:10s}: {num_nodes:4d} nodes, {num_features:3d} features")
    print()

    # Edge statistics
    print("Edges:")
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.shape[1]
        src, rel, dst = edge_type
        print(f"  {src:10s} --[{rel:15s}]--> {dst:10s}: {num_edges:4d} edges")
    print()

    # Metadata
    if hasattr(data, 'thread_id'):
        print(f"Thread ID: {data.thread_id}")
    print()
