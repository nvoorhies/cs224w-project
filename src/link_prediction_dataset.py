"""
Link Prediction Dataset for PHEME.

This module provides a dataset class for link prediction tasks,
specifically for predicting "follow request" links between users.
"""

import torch
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from torch_geometric.data import HeteroData, Data
import numpy as np

from .pyg_dataset import PHEMEDataset


# All 9 news stories in the PHEME dataset
ALL_STORIES = [
    'charliehebdo-all-rnr-threads',
    'ebola-essien-all-rnr-threads',
    'ferguson-all-rnr-threads',
    'germanwings-crash-all-rnr-threads',
    'gurlitt-all-rnr-threads',
    'ottawashooting-all-rnr-threads',
    'prince-toronto-all-rnr-threads',
    'putinmissing-all-rnr-threads',
    'sydneysiege-all-rnr-threads',
]


def split_stories(
    stories: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split stories into train/val/test sets.
    
    Args:
        stories: List of story names
        train_ratio: Ratio of stories for training
        val_ratio: Ratio of stories for validation
        test_ratio: Ratio of stories for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_stories, val_stories, test_stories)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    random.seed(seed)
    shuffled_stories = stories.copy()
    random.shuffle(shuffled_stories)
    
    n = len(shuffled_stories)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_stories = shuffled_stories[:train_end]
    val_stories = shuffled_stories[train_end:val_end]
    test_stories = shuffled_stories[val_end:]
    
    return train_stories, val_stories, test_stories


class LinkPredictionDataset(PHEMEDataset):
    """
    Dataset for link prediction on PHEME graphs.
    
    Extends PHEMEDataset to generate positive and negative samples
    for link prediction tasks (e.g., predicting follow requests).
    """
    
    def __init__(
        self,
        root: str | Path,
        stories: Optional[List[str]] = None,
        rumour_types: Optional[List[str]] = None,
        max_threads: Optional[int] = None,
        split: Optional[str] = None,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        include_temporal_encoding: bool = True,
        temporal_encoding_dim: int = 16,
        normalize_features: bool = False,
        user_edge_type: str = "replies",
        add_reverse_edges: bool = True,
        skip_errors: bool = True,
        seed: int = 42,
        # Link prediction specific parameters
        num_negative_samples: int = 1,
        use_existing_edges_as_positive: bool = True,
    ):
        """
        Initialize LinkPredictionDataset.
        
        Args:
            root: Root directory of the dataset
            stories: List of story names to include (None = all stories)
            rumour_types: Filter by rumour type
            max_threads: Maximum number of threads to load
            split: Which split to use ("train", "val", "test", or None)
            split_ratios: Train/val/test split ratios
            include_temporal_encoding: Include temporal positional encoding
            temporal_encoding_dim: Dimension of temporal encoding
            normalize_features: Whether to normalize features
            user_edge_type: How to construct user->user edges
            add_reverse_edges: Whether to add reverse edges
            skip_errors: Whether to skip threads that fail to load
            seed: Random seed for splits
            num_negative_samples: Number of negative samples per positive sample
            use_existing_edges_as_positive: Whether to use existing user-user edges as positive samples
        """
        super().__init__(
            root=root,
            stories=stories,
            rumour_types=rumour_types,
            max_threads=max_threads,
            split=split,
            split_ratios=split_ratios,
            include_temporal_encoding=include_temporal_encoding,
            temporal_encoding_dim=temporal_encoding_dim,
            normalize_features=normalize_features,
            user_edge_type=user_edge_type,
            add_reverse_edges=add_reverse_edges,
            skip_errors=skip_errors,
            seed=seed,
        )
        
        self.num_negative_samples = num_negative_samples
        self.use_existing_edges_as_positive = use_existing_edges_as_positive
        self.seed = seed
    
    def get(self, idx: int) -> HeteroData:
        """
        Get a graph with link prediction labels.
        
        Args:
            idx: Index of the graph
            
        Returns:
            HeteroData object with edge_label_index and edge_label for link prediction
        """
        # Get the base graph
        graph = super().get(idx)
        
        # Generate positive and negative samples for link prediction
        edge_label_index, edge_label = self._generate_link_samples(graph)
        
        # Add to graph
        graph.edge_label_index = edge_label_index
        graph.edge_label = edge_label
        
        return graph
    
    def _generate_link_samples(
        self,
        graph: HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate positive and negative samples for link prediction.
        
        For "follow request" prediction, we use existing user-user interactions
        as positive samples and randomly sample negative pairs.
        
        Args:
            graph: HeteroData graph
            
        Returns:
            Tuple of (edge_label_index, edge_label)
            - edge_label_index: [2, num_edges] tensor of edge indices
            - edge_label: [num_edges] tensor of labels (1 for positive, 0 for negative)
        """
        num_users = graph['user'].x.shape[0]
        
        # Get positive samples (existing user-user edges)
        positive_edges = []
        if ('user', 'interacts_with', 'user') in graph.edge_types:
            pos_edge_index = graph[('user', 'interacts_with', 'user')].edge_index
            if pos_edge_index.shape[1] > 0:
                positive_edges.append(pos_edge_index)
        
        if ('user', 'interacted_by', 'user') in graph.edge_types:
            pos_edge_index = graph[('user', 'interacted_by', 'user')].edge_index
            if pos_edge_index.shape[1] > 0:
                # Reverse the edges to get (src, dst) format
                positive_edges.append(pos_edge_index.flip(0))
        
        # Combine all positive edges
        if positive_edges:
            all_positive = torch.cat(positive_edges, dim=1)
            # Remove duplicates
            all_positive = torch.unique(all_positive, dim=1)
        else:
            # If no existing edges, create positive samples based on tweet interactions
            # Users who interact in the same thread are likely to have follow requests
            all_positive = self._generate_positive_from_tweets(graph)
        
        num_positive = all_positive.shape[1]
        
        # Generate negative samples
        negative_edges = self._generate_negative_samples(
            graph, num_positive * self.num_negative_samples
        )
        
        # Combine positive and negative samples
        edge_label_index = torch.cat([all_positive, negative_edges], dim=1)
        edge_label = torch.cat([
            torch.ones(num_positive, dtype=torch.long),
            torch.zeros(negative_edges.shape[1], dtype=torch.long)
        ])
        
        # Shuffle
        perm = torch.randperm(edge_label_index.shape[1])
        edge_label_index = edge_label_index[:, perm]
        edge_label = edge_label[perm]
        
        return edge_label_index, edge_label
    
    def _generate_positive_from_tweets(self, graph: HeteroData) -> torch.Tensor:
        """
        Generate positive samples from tweet interactions.
        
        If two users have tweets in the same thread and one replies to the other,
        we consider this as a potential follow request.
        
        Args:
            graph: HeteroData graph
            
        Returns:
            Edge index tensor [2, num_positive_edges]
        """
        # Get user-tweet authorship edges
        if ('user', 'posts', 'tweet') not in graph.edge_types:
            return torch.zeros((2, 0), dtype=torch.long)
        
        user_tweet_edges = graph[('user', 'posts', 'tweet')].edge_index
        num_users = graph['user'].x.shape[0]
        num_tweets = graph['tweet'].x.shape[0]
        
        # Build user -> tweets mapping
        user_tweets = {}
        for i in range(user_tweet_edges.shape[1]):
            user_idx = user_tweet_edges[0, i].item()
            tweet_idx = user_tweet_edges[1, i].item()
            if user_idx not in user_tweets:
                user_tweets[user_idx] = []
            user_tweets[user_idx].append(tweet_idx)
        
        # Get reply edges
        positive_pairs = set()
        if ('tweet', 'replies_to', 'tweet') in graph.edge_types:
            reply_edges = graph[('tweet', 'replies_to', 'tweet')].edge_index
            
            # Build tweet -> author mapping
            tweet_author = {}
            for i in range(user_tweet_edges.shape[1]):
                user_idx = user_tweet_edges[0, i].item()
                tweet_idx = user_tweet_edges[1, i].item()
                tweet_author[tweet_idx] = user_idx
            
            # For each reply edge, create a user-user edge
            for i in range(reply_edges.shape[1]):
                parent_tweet = reply_edges[0, i].item()
                child_tweet = reply_edges[1, i].item()
                
                if parent_tweet in tweet_author and child_tweet in tweet_author:
                    parent_user = tweet_author[parent_tweet]
                    child_user = tweet_author[child_tweet]
                    
                    if parent_user != child_user:
                        # Child user replies to parent user -> potential follow request
                        positive_pairs.add((child_user, parent_user))
        
        if not positive_pairs:
            # Fallback: create edges between users who have tweets in the thread
            users_with_tweets = list(user_tweets.keys())
            if len(users_with_tweets) >= 2:
                # Create edges between all pairs (sparse)
                for i, u1 in enumerate(users_with_tweets):
                    for u2 in users_with_tweets[i+1:min(i+3, len(users_with_tweets))]:
                        positive_pairs.add((u1, u2))
        
        if positive_pairs:
            edges = list(positive_pairs)
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            return torch.zeros((2, 0), dtype=torch.long)
    
    def _generate_negative_samples(
        self,
        graph: HeteroData,
        num_negative: int
    ) -> torch.Tensor:
        """
        Generate negative samples (non-edges) for link prediction.
        
        Args:
            graph: HeteroData graph
            num_negative: Number of negative samples to generate
            
        Returns:
            Edge index tensor [2, num_negative]
        """
        num_users = graph['user'].x.shape[0]
        
        if num_users < 2:
            return torch.zeros((2, 0), dtype=torch.long)
        
        # Get all existing edges to avoid sampling them
        existing_edges = set()
        if ('user', 'interacts_with', 'user') in graph.edge_types:
            edges = graph[('user', 'interacts_with', 'user')].edge_index
            for i in range(edges.shape[1]):
                existing_edges.add((edges[0, i].item(), edges[1, i].item()))
        
        if ('user', 'interacted_by', 'user') in graph.edge_types:
            edges = graph[('user', 'interacted_by', 'user')].edge_index
            for i in range(edges.shape[1]):
                existing_edges.add((edges[1, i].item(), edges[0, i].item()))
        
        # Sample negative edges
        negative_edges = []
        max_attempts = num_negative * 10
        attempts = 0
        
        random.seed(self.seed + hash(graph.thread_id) if hasattr(graph, 'thread_id') else self.seed)
        
        while len(negative_edges) < num_negative and attempts < max_attempts:
            u1 = random.randint(0, num_users - 1)
            u2 = random.randint(0, num_users - 1)
            
            if u1 != u2 and (u1, u2) not in existing_edges:
                negative_edges.append([u1, u2])
            
            attempts += 1
        
        if negative_edges:
            return torch.tensor(negative_edges, dtype=torch.long).t().contiguous()
        else:
            return torch.zeros((2, 0), dtype=torch.long)


def create_story_split_datasets(
    root: str | Path,
    train_stories: List[str],
    val_stories: List[str],
    test_stories: List[str],
    **kwargs
) -> Tuple[LinkPredictionDataset, LinkPredictionDataset, LinkPredictionDataset]:
    """
    Create train/val/test datasets based on story splits.
    
    Args:
        root: Root directory of the dataset
        train_stories: List of story names for training
        val_stories: List of story names for validation
        test_stories: List of story names for testing
        **kwargs: Additional arguments passed to LinkPredictionDataset
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = LinkPredictionDataset(
        root=root,
        stories=train_stories,
        split=None,  # Don't apply additional split
        **kwargs
    )
    
    val_dataset = LinkPredictionDataset(
        root=root,
        stories=val_stories,
        split=None,
        **kwargs
    )
    
    test_dataset = LinkPredictionDataset(
        root=root,
        stories=test_stories,
        split=None,
        **kwargs
    )
    
    return train_dataset, val_dataset, test_dataset

