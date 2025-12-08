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
        
        # Generate positive and negative samples for tweet->user link prediction
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
        
        For "user reply" prediction, we use user->tweet edges that represent
        a reply-driven engagement (user replying to another user's tweet).
        We construct the positive samples from interactions using `('user', 'posts', 'tweet')` 
        and `('tweet', 'replies', 'tweet')` relation constructed by the graph builder. 
        We then randomly sample negative user->tweet pairs that are not observed.
        
        Args:
            graph: HeteroData graph
            
        Returns:
            Tuple of (edge_label_index, edge_label)
            - edge_label_index: [2, num_edges] tensor of edge indices
            - edge_label: [num_edges] tensor of labels (1 for positive, 0 for negative)
        """
        num_tweets = graph['tweet'].x.shape[0]
        num_users = graph['user'].x.shape[0]
        
        # Get positive samples (user->tweet reply edges)
        all_positive = self._generate_positive_from_tweets(graph)
        
        num_positive = all_positive.shape[1]
        
        # Generate negative samples
        negative_edges = self._generate_negative_samples(
            graph, num_positive * self.num_negative_samples, all_positive
        )
        if negative_edges.shape[1] == 0 and num_positive > 0:
            # Ensure at least some negatives exist
            negative_edges = self._generate_negative_samples(
                graph, max(num_positive, 10), all_positive
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
        
        If a tweet replies to another tweet, we link the replying tweet's author to the
        parent tweet.
        
        Args:
            graph: HeteroData graph
            
        Returns:
            Edge index tensor [2, num_positive_edges] (user_idx, tweet_idx)
        """
        if (
            ('tweet', 'replies_to', 'tweet') not in graph.edge_types
            or ('user', 'posts', 'tweet') not in graph.edge_types
        ):
            return torch.zeros((2, 0), dtype=torch.long)
        
        reply_edges = graph[('tweet', 'replies_to', 'tweet')].edge_index
        authorship = graph[('user', 'posts', 'tweet')].edge_index
        
        tweet_to_author = {}
        for i in range(authorship.shape[1]):
            user_idx = authorship[0, i].item()
            tweet_idx = authorship[1, i].item()
            tweet_to_author[tweet_idx] = user_idx
        
        positive_pairs = set()
        for i in range(reply_edges.shape[1]):
            parent_tweet = reply_edges[0, i].item()
            child_tweet = reply_edges[1, i].item()
            
            if (
                parent_tweet in tweet_to_author
                and child_tweet in tweet_to_author
            ):
                child_author = tweet_to_author[child_tweet]
                parent_tweet_idx = parent_tweet
                
                positive_pairs.add((child_author, parent_tweet_idx))
        
        if not positive_pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        
        edges = list(positive_pairs)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _generate_negative_samples(
        self,
        graph: HeteroData,
        num_negative: int,
        positive_edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate negative samples (non-edges) for link prediction.
        
        Args:
            graph: HeteroData graph
            num_negative: Number of negative samples to generate
            positive_edges: positive samples index tensor [2, num_positive_edges] (user_idx, tweet_idx)
            
        Returns:
            Edge index tensor [2, num_negative]
        """
        # TODO: negative samples should exlude any reply edges that exist in the original graph, despite the edge being the first reply after t or not.
        # take the median time of each thread  as the cutoff, so 50% interactions in that thread  for base graph
        num_tweets = graph['tweet'].x.shape[0]
        num_users = graph['user'].x.shape[0]
        
        if num_tweets == 0 or num_users == 0 or num_negative == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        
        # Get all existing user-tweet edges to avoid sampling them
        existing_edges = set()
        edges = graph[('user', 'posts', 'tweet')].edge_index
        for i in range(edges.shape[1]):
            existing_edges.add((edges[0, i].item(), edges[1, i].item()))
        for i in range(positive_edges.shape[1]):
            existing_edges.add((positive_edges[0, i].item(), positive_edges[1, i].item()))
        
        # Sample negative edges
        negative_edges = []
        max_attempts = num_negative * 10
        attempts = 0
        
        random.seed(self.seed + hash(graph.thread_id) if hasattr(graph, 'thread_id') else self.seed)
        
        while len(negative_edges) < num_negative and attempts < max_attempts:
            tweet_idx = random.randint(0, num_tweets - 1)
            user_idx = random.randint(0, num_users - 1)
            
            if (user_idx, tweet_idx) not in existing_edges:
                negative_edges.append([user_idx, tweet_idx])
                existing_edges.add((user_idx, tweet_idx))
            
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

