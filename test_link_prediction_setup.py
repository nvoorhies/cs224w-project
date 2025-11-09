"""
Test script to verify the link prediction setup works correctly.

This script tests:
1. Dataset loading
2. Model creation
3. Forward pass
4. Link prediction generation
"""

import torch
from pathlib import Path
from src.link_prediction_dataset import LinkPredictionDataset, ALL_STORIES, split_stories
from src.het_gat_model import HeteroGATLinkPrediction


def test_dataset_loading():
    """Test that datasets can be loaded."""
    print("Testing dataset loading...")
    
    # Use a small subset for testing
    data_root = Path("data/all-rnr-annotated-threads")
    if not data_root.exists():
        print(f"Warning: Data root {data_root} does not exist. Skipping dataset test.")
        return None
    
    # Create a small dataset
    dataset = LinkPredictionDataset(
        root=data_root,
        stories=['germanwings-crash-all-rnr-threads'],
        max_threads=5,
        user_edge_type='replies',
    )
    
    print(f"  Loaded {len(dataset)} graphs")
    
    if len(dataset) > 0:
        # Get first graph
        graph = dataset[0]
        print(f"  Graph node types: {graph.node_types}")
        print(f"  Graph edge types: {graph.edge_types}")
        print(f"  Number of users: {graph['user'].x.shape[0]}")
        print(f"  Number of tweets: {graph['tweet'].x.shape[0]}")
        print(f"  Edge label index shape: {graph.edge_label_index.shape}")
        print(f"  Edge label shape: {graph.edge_label.shape}")
        print(f"  Positive samples: {graph.edge_label.sum().item()}")
        print(f"  Negative samples: {(graph.edge_label == 0).sum().item()}")
        return graph
    else:
        print("  No graphs loaded!")
        return None


def test_model_creation(graph):
    """Test that model can be created and run forward pass."""
    if graph is None:
        print("Skipping model test (no graph available)")
        return
    
    print("\nTesting model creation...")
    
    # Get feature dimensions
    in_channels_dict = {
        node_type: graph[node_type].x.shape[1] 
        for node_type in graph.node_types
    }
    print(f"  Input feature dimensions: {in_channels_dict}")
    
    # Create model
    model = HeteroGATLinkPrediction(
        in_channels_dict=in_channels_dict,
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        heads=2,
        dropout=0.5,
    )
    
    print(f"  Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
    edge_index_dict = {
        edge_type: graph[edge_type].edge_index 
        for edge_type in graph.edge_types
    }
    edge_label_index = graph.edge_label_index
    
    # Forward pass
    node_emb_dict, link_pred = model(x_dict, edge_index_dict, edge_label_index)
    
    print(f"  Node embeddings:")
    for node_type, emb in node_emb_dict.items():
        print(f"    {node_type}: {emb.shape}")
    print(f"  Link predictions: {link_pred.shape}")
    if link_pred.numel() > 0:
        print(f"  Link prediction range: [{link_pred.min().item():.4f}, {link_pred.max().item():.4f}]")
    else:
        print("  Link prediction range: [N/A] (no link samples)")
    
    # Test without link prediction
    node_emb_dict_only = model.encode_nodes(x_dict, edge_index_dict)
    print(f"  Node embeddings (encode only):")
    for node_type, emb in node_emb_dict_only.items():
        print(f"    {node_type}: {emb.shape}")


def test_story_splitting():
    """Test story splitting."""
    print("\nTesting story splitting...")
    
    train_stories, val_stories, test_stories = split_stories(
        ALL_STORIES,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    
    print(f"  Train stories ({len(train_stories)}): {train_stories}")
    print(f"  Val stories ({len(val_stories)}): {val_stories}")
    print(f"  Test stories ({len(test_stories)}): {test_stories}")
    print(f"  Total: {len(train_stories) + len(val_stories) + len(test_stories)}")
    print(f"  Expected: {len(ALL_STORIES)}")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Link Prediction Setup Test")
    print("=" * 80)
    
    # Test story splitting
    test_story_splitting()
    
    # Test dataset loading
    graph = test_dataset_loading()
    
    # Test model
    if graph is not None:
        test_model_creation(graph)
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests skipped (data not available)")


if __name__ == '__main__':
    main()

