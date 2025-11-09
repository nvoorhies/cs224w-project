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

from .link_prediction_dataset import (
    LinkPredictionDataset,
    create_story_split_datasets,
    split_stories,
    ALL_STORIES,
)

from .het_gat_model import (
    HeteroGATLayer,
    TemporalHeteroGAT,
    LinkPredictor,
    HeteroGATLinkPrediction,
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
    # Link Prediction Dataset
    "LinkPredictionDataset",
    "create_story_split_datasets",
    "split_stories",
    "ALL_STORIES",
    # Heterogeneous GAT Model
    "HeteroGATLayer",
    "TemporalHeteroGAT",
    "LinkPredictor",
    "HeteroGATLinkPrediction",
]
