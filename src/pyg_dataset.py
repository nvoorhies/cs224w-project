"""
PyTorch Geometric Dataset for PHEME threads.

This module provides a Dataset class for loading PHEME threads as
PyG HeteroData objects, with support for train/val/test splits.
"""

import torch
from pathlib import Path
from typing import List, Optional, Literal, Callable
from torch_geometric.data import Dataset, HeteroData
import logging

from .parsers import find_all_threads, load_tweet_thread, ParseError
from .graph_builder import PHEMEGraphBuilder

logger = logging.getLogger(__name__)


class PHEMEDataset(Dataset):
    """
    PyTorch Geometric Dataset for PHEME tweet threads.

    This dataset loads PHEME threads and converts them to heterogeneous graphs
    on-the-fly. Supports filtering by story, rumour type, and train/val/test splits.

    Example:
        >>> dataset = PHEMEDataset(
        ...     root='data/all-rnr-annotated-threads',
        ...     stories=['germanwings-crash-all-rnr-threads'],
        ...     max_threads=100
        ... )
        >>> print(len(dataset))
        100
        >>> graph = dataset[0]
        >>> print(graph)
    """

    def __init__(
        self,
        root: str | Path,
        stories: Optional[List[str]] = None,
        rumour_types: Optional[List[str]] = None,
        max_threads: Optional[int] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        include_temporal_encoding: bool = True,
        temporal_encoding_dim: int = 16,
        normalize_features: bool = False,
        user_edge_type: Literal["mentions", "replies", "both", "none"] = "replies",
        add_reverse_edges: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        skip_errors: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the PHEME Dataset.

        Args:
            root: Root directory of the dataset
            stories: List of story names to include (None = all stories)
            rumour_types: Filter by rumour type (["rumour"], ["non-rumour"], or both)
            max_threads: Maximum number of threads to load (None = all)
            split: Which split to use ("train", "val", "test", or None for all)
            split_ratios: Train/val/test split ratios (must sum to 1.0)
            include_temporal_encoding: Include temporal positional encoding
            temporal_encoding_dim: Dimension of temporal encoding
            normalize_features: Whether to normalize features
            user_edge_type: How to construct user->user edges
            add_reverse_edges: Whether to add reverse edges
            transform: Optional transform to apply to graphs
            pre_transform: Optional pre-transform
            skip_errors: Whether to skip threads that fail to load
            seed: Random seed for splits
        """
        self.root_path = Path(root)
        self.stories = stories
        self.rumour_types = rumour_types
        self.max_threads = max_threads
        self.split = split
        self.split_ratios = split_ratios
        self.skip_errors = skip_errors
        self.seed = seed

        # Graph builder configuration
        self.graph_builder = PHEMEGraphBuilder(
            include_temporal_encoding=include_temporal_encoding,
            temporal_encoding_dim=temporal_encoding_dim,
            normalize_features=normalize_features,
            user_edge_type=user_edge_type
        )
        self.add_reverse_edges = add_reverse_edges

        # Load thread paths (before super().__init__)
        self.thread_paths = self._load_thread_paths()

        # Initialize dataset
        super().__init__(str(root), transform, pre_transform)

        # Apply split if requested
        if split is not None:
            self.thread_paths = self._apply_split(self.thread_paths, split)

        logger.info(f"Initialized PHEME dataset with {len(self.thread_paths)} threads")

    def _load_thread_paths(self) -> List[Path]:
        """Load and filter thread paths based on configuration."""
        # Find all threads
        if self.stories:
            thread_paths = []
            for story in self.stories:
                story_path = self.root_path / story
                if story_path.exists():
                    thread_paths.extend(find_all_threads(story_path))
                else:
                    logger.warning(f"Story not found: {story}")
        else:
            thread_paths = find_all_threads(self.root_path)

        # Filter by rumour type if specified
        if self.rumour_types:
            filtered_paths = []
            for path in thread_paths:
                # Check if path contains rumours or non-rumours
                if any(rt in str(path) for rt in self.rumour_types):
                    filtered_paths.append(path)
            thread_paths = filtered_paths

        # Limit number of threads
        if self.max_threads:
            thread_paths = thread_paths[:self.max_threads]

        return thread_paths

    def _apply_split(
        self,
        thread_paths: List[Path],
        split: str
    ) -> List[Path]:
        """Apply train/val/test split to thread paths."""
        import random

        # Set seed for reproducibility
        random.seed(self.seed)
        shuffled_paths = thread_paths.copy()
        random.shuffle(shuffled_paths)

        # Calculate split indices
        train_ratio, val_ratio, test_ratio = self.split_ratios
        assert abs(sum(self.split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

        n = len(shuffled_paths)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Select split
        if split == "train":
            return shuffled_paths[:train_end]
        elif split == "val":
            return shuffled_paths[train_end:val_end]
        elif split == "test":
            return shuffled_paths[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.thread_paths)

    def get(self, idx: int) -> HeteroData:
        """
        Get a graph by index.

        Args:
            idx: Index of the graph

        Returns:
            HeteroData object

        Raises:
            IndexError: If index is out of bounds
            RuntimeError: If thread cannot be loaded and skip_errors is False
        """
        if idx < 0 or idx >= len(self.thread_paths):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")

        thread_path = self.thread_paths[idx]

        try:
            # Load thread
            thread = load_tweet_thread(thread_path, load_reactions=True)

            # Build graph
            graph = self.graph_builder.build_graph(
                thread,
                add_reverse_edges=self.add_reverse_edges
            )

            # Add labels from annotation
            graph.y_rumour = self._encode_rumour_type(thread.annotation.is_rumour)
            graph.y_misinformation = thread.annotation.misinformation
            graph.y_true = thread.annotation.true
            graph.y_turnaround = thread.annotation.is_turnaround

            return graph

        except (ParseError, Exception) as e:
            if self.skip_errors:
                logger.warning(f"Skipping thread {thread_path.name}: {e}")
                # Return next valid graph
                if idx + 1 < len(self):
                    return self.get(idx + 1)
                else:
                    raise RuntimeError(f"No valid graphs found after index {idx}")
            else:
                raise RuntimeError(f"Failed to load thread {thread_path.name}: {e}")

    def _encode_rumour_type(self, rumour_type: str) -> int:
        """Encode rumour type as integer label."""
        encoding = {
            "rumour": 0,
            "non-rumour": 1,
            "unverified": 2
        }
        return encoding.get(rumour_type, 2)

    def get_split_datasets(
        self
    ) -> tuple["PHEMEDataset", "PHEMEDataset", "PHEMEDataset"]:
        """
        Create train/val/test split datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)

        Example:
            >>> full_dataset = PHEMEDataset(root='data/...')
            >>> train, val, test = full_dataset.get_split_datasets()
            >>> print(len(train), len(val), len(test))
        """
        # Create datasets for each split
        train_dataset = PHEMEDataset(
            root=self.root_path,
            stories=self.stories,
            rumour_types=self.rumour_types,
            max_threads=self.max_threads,
            split="train",
            split_ratios=self.split_ratios,
            include_temporal_encoding=self.graph_builder.include_temporal_encoding,
            temporal_encoding_dim=self.graph_builder.temporal_encoding_dim,
            normalize_features=self.graph_builder.normalize_features,
            user_edge_type=self.graph_builder.user_edge_type,
            add_reverse_edges=self.add_reverse_edges,
            transform=self.transform,
            pre_transform=self.pre_transform,
            skip_errors=self.skip_errors,
            seed=self.seed,
        )

        val_dataset = PHEMEDataset(
            root=self.root_path,
            stories=self.stories,
            rumour_types=self.rumour_types,
            max_threads=self.max_threads,
            split="val",
            split_ratios=self.split_ratios,
            include_temporal_encoding=self.graph_builder.include_temporal_encoding,
            temporal_encoding_dim=self.graph_builder.temporal_encoding_dim,
            normalize_features=self.graph_builder.normalize_features,
            user_edge_type=self.graph_builder.user_edge_type,
            add_reverse_edges=self.add_reverse_edges,
            transform=self.transform,
            pre_transform=self.pre_transform,
            skip_errors=self.skip_errors,
            seed=self.seed,
        )

        test_dataset = PHEMEDataset(
            root=self.root_path,
            stories=self.stories,
            rumour_types=self.rumour_types,
            max_threads=self.max_threads,
            split="test",
            split_ratios=self.split_ratios,
            include_temporal_encoding=self.graph_builder.include_temporal_encoding,
            temporal_encoding_dim=self.graph_builder.temporal_encoding_dim,
            normalize_features=self.graph_builder.normalize_features,
            user_edge_type=self.graph_builder.user_edge_type,
            add_reverse_edges=self.add_reverse_edges,
            transform=self.transform,
            pre_transform=self.pre_transform,
            skip_errors=self.skip_errors,
            seed=self.seed,
        )

        return train_dataset, val_dataset, test_dataset

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  num_threads={len(self)},\n"
            f"  stories={self.stories},\n"
            f"  rumour_types={self.rumour_types},\n"
            f"  split={self.split},\n"
            f"  user_edge_type={self.graph_builder.user_edge_type}\n"
            f")"
        )
