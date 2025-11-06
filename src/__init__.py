"""
PHEME Dataset Parser

A robust parser for the PHEME dataset with Pydantic validation
and PyTorch Geometric graph construction.
"""

from .models import (
    Annotation,
    AnnotationLink,
    ThreadStructure,
    Tweet,
    TweetThread,
    TwitterUser,
    TweetEntities,
)

from .parsers import (
    load_annotation,
    load_structure,
    load_tweet,
    load_tweet_thread,
    load_dataset,
    find_all_threads,
    verify_thread,
    ParseError,
)

from .graph_builder import (
    PHEMEGraphBuilder,
    thread_to_graph,
    print_graph_statistics,
)

from .pyg_dataset import (
    PHEMEDataset,
)

__all__ = [
    # Models
    "Annotation",
    "AnnotationLink",
    "ThreadStructure",
    "Tweet",
    "TweetThread",
    "TwitterUser",
    "TweetEntities",
    # Parsers
    "load_annotation",
    "load_structure",
    "load_tweet",
    "load_tweet_thread",
    "load_dataset",
    "find_all_threads",
    "verify_thread",
    "ParseError",
    # Graph Builder
    "PHEMEGraphBuilder",
    "thread_to_graph",
    "print_graph_statistics",
    # PyG Dataset
    "PHEMEDataset",
]
