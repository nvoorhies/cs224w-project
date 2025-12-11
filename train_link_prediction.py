"""
Training script for link prediction on PHEME dataset.

Trains a Heterogeneous GAT model to predict "follow_request_sent" links
from tweets to users in rumour-centric threads.
"""

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import networkx as nx
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.het_gat_model import HeteroGATLinkPrediction
from src.link_prediction_dataset import (
    LinkPredictionDataset,
    create_story_split_datasets,
    split_stories,
    ALL_STORIES,
)
from visualize import *

def filter_edges(x_dict, edge_index_dict):
    """Temporal filter: Remove nodes and edges involving nodes with delta_minutes cutoff t """
    tweet_node_features = x_dict['tweet']
    delta_minutes_idx = 13  # include_temporal_encoding = False
    # print(tweet_node_features.shape)
    delta_minutes = tweet_node_features[:, delta_minutes_idx]
    
    # Calculate cutoff_t as the median of all delta_minutes
    cutoff_t = torch.quantile(delta_minutes, 0.25, interpolation='lower').item()
    # print("Median cutoff_t:", cutoff_t)  # e.g. 8.966666221618652

    valid_nodes = delta_minutes <= cutoff_t

    # remove tweet to tweet reply edges
    for edge_type, edge_index in edge_index_dict.items():
        if edge_type[0] == 'tweet' and edge_type[2] == 'tweet':
            # print("edge_index shape before:", edge_index_dict[edge_type].shape)  # e.g. torch.Size([2, 18])
            valid_edges = valid_nodes[edge_index[0]] & valid_nodes[edge_index[1]]
            edge_index_dict[edge_type] = edge_index[:, valid_edges]
            # print("edge_index shape after (should be smaller):", edge_index_dict[edge_type].shape)  # e.g. torch.Size([2, 9])


def filter_labels(edge_index_dict, edge_label_index, edge_label):
    """
        Remove positive edges where the reply tweet is before cutoff time to avoid temporal leakage.
        
        If tweetB->tweetA reply_to edge exists in edge_index_dict (before cutoff),
        find userB (author of tweetB) and remove positive edge userB->tweetA.
        Then balance negative edges to match the number of remaining positive edges.
    """
    # Get tweet reply edges if they exist
    reply_edge_type = ('tweet', 'replies_to', 'tweet')
    if reply_edge_type not in edge_index_dict:
        return
    
    reply_edges = edge_index_dict[reply_edge_type]  # [2, num_reply_edges]
    
    # Separate positive and negative edges
    pos_mask = edge_label == 1
    neg_mask = edge_label == 0
    
    pos_edge_index = edge_label_index[:, pos_mask]  # [2, num_pos]
    neg_edge_index = edge_label_index[:, neg_mask]  # [2, num_neg]
    
    # Find positive edges to remove
    # For each positive edge (user->tweet), check if there's a reply (tweet->tweet) from user's tweet
    edges_to_keep = []
    for i in range(pos_edge_index.shape[1]):
        user_idx = pos_edge_index[0, i]
        tweet_idx = pos_edge_index[1, i]
        
        # Find tweets authored by this user (from edge_index_dict)
        # We need ('user', 'posts', 'tweet') edges to find user's tweets
        posts_edge_type = ('user', 'posts', 'tweet')
        if posts_edge_type in edge_index_dict:
            posts_edges = edge_index_dict[posts_edge_type]
            user_tweets = posts_edges[1, posts_edges[0] == user_idx]  # tweets by this user
            
            # Check if any of user's tweets reply to the target tweet
            # reply_edges: [source_tweet, target_tweet]
            has_reply = False
            for user_tweet in user_tweets:
                # Check if user_tweet -> tweet_idx exists in reply_edges
                if torch.any((reply_edges[0] == user_tweet) & (reply_edges[1] == tweet_idx)):
                    has_reply = True
                    break
            if not has_reply:
                edges_to_keep.append(i)
        else:
            # If we can't verify, keep the edge
            edges_to_keep.append(i)
    
    # Filter positive edges
    if len(edges_to_keep) < pos_edge_index.shape[1]:
        edges_to_keep = torch.tensor(edges_to_keep, dtype=torch.long, device=pos_edge_index.device)
        pos_edge_index = pos_edge_index[:, edges_to_keep]
        
        # Balance negative edges to match positive edges
        num_pos = pos_edge_index.shape[1]
        if neg_edge_index.shape[1] > num_pos:
            # Randomly sample negative edges
            perm = torch.randperm(neg_edge_index.shape[1])[:num_pos]
            neg_edge_index = neg_edge_index[:, perm]
        
        # Reconstruct edge_label_index and edge_label
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        pos_labels = torch.ones(pos_edge_index.shape[1], device=edge_label.device)
        neg_labels = torch.zeros(neg_edge_index.shape[1], device=edge_label.device)
        edge_label = torch.cat([pos_labels, neg_labels], dim=0)
        # print(f"Filtered positive edges: {pos_mask.sum().item()} -> {num_pos}")  # e.g. 35 -> 27
    
    return edge_label_index, edge_label
    

def train_epoch(model, train_loader, optimizer, device, vis):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_graphs = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        # Move batch to device
        batch = batch.to(device)
        
        # Get node features and edge indices
        x_dict = {node_type: batch[node_type].x for node_type in batch.node_types}
        edge_index_dict = {
            edge_type: batch[edge_type].edge_index 
            for edge_type in batch.edge_types
        }

        if vis and num_graphs < 10: 
            G = nx.DiGraph()
            pos = visualize_input("Full Thread HeteroData Graph (Green=Tweet node, Purple=User node)", edge_index_dict, G=G)

        filter_edges(x_dict, edge_index_dict)

        if vis and num_graphs < 10: 
            visualize_input("Thread HeteroData Graph Filtered by Time Cutoff", edge_index_dict, pos=pos, G=G)
        
        # Get link prediction labels
        # Note: For batched graphs, edge_label_index should be adjusted by PyG's collate
        # If using batch_size > 1, we need custom collate function or process individually
        if not hasattr(batch, 'edge_label_index') or batch.edge_label_index is None:
            continue
            
        edge_label_index = batch.edge_label_index
        edge_label = batch.edge_label.float()

        # if tweetB->tweetA reply_to edge is in the edge_index_dict (meaning it's before cutoff time and kept in the input graph to message passing), 
        # then we find the author (userB) of the reply (tweetB) and remove the positive edge that connects userB->tweetA reply relation.
        # Negatve edges don't need filtering but the quantity should match the filtered positve edges.
        edge_label_index, edge_label = filter_labels(edge_index_dict, edge_label_index, edge_label)

        if vis and num_graphs < 10: 
            visualize_input("Edge Labels for Link Prediction Task (Green=Positive edge, Orange=Negative)", edge_index_dict, edge_label_index, edge_label, pos=pos, G=G)
        
        # Skip graphs without link labels
        if edge_label.numel() == 0 or edge_label_index.numel() == 0:
            continue

        # Forward pass
        optimizer.zero_grad()
        node_emb_dict, att_dict, link_pred = model(x_dict, edge_index_dict, edge_label_index)
        if vis and num_graphs < 10: 
            visualize_graph(node_emb_dict, att_dict, pos=pos, G=G)

        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(link_pred, edge_label)
        
        # Backward pass
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_graphs += 1
    
    return total_loss / max(num_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_graphs = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        
        # Get node features and edge indices
        x_dict = {node_type: batch[node_type].x for node_type in batch.node_types}
        edge_index_dict = {
            edge_type: batch[edge_type].edge_index 
            for edge_type in batch.edge_types
        }

        filter_edges(x_dict, edge_index_dict)
        
        # Get link prediction labels
        if hasattr(batch, 'edge_label_index') and batch.edge_label_index is not None:
            edge_label_index = batch.edge_label_index
            edge_label = batch.edge_label.float()

            edge_label_index, edge_label = filter_labels(edge_index_dict, edge_label_index, edge_label)

        else:
            continue
        
        if edge_label.numel() == 0 or edge_label_index.numel() == 0:
            continue

        # Forward pass
        _, _, link_pred = model(x_dict, edge_index_dict, edge_label_index)
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(link_pred, edge_label)
        
        # Store predictions and labels
        pred_probs = torch.sigmoid(link_pred)
        all_preds.append(pred_probs.cpu().numpy())
        all_labels.append(edge_label.cpu().numpy())
        
        total_loss += loss.item()
        num_graphs += 1
    
    if len(all_preds) == 0:
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'auc_roc': 0.0,
            'auc_pr': 0.0,
        }
    
    # Compute metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Accuracy (using 0.5 threshold)
    pred_binary = (all_preds > 0.5).astype(int)
    accuracy = (pred_binary == all_labels).mean()
    
    # AUC-ROC
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    try:
        auc_roc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_roc = 0.0
    
    # AUC-PR
    try:
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        auc_pr = auc(recall, precision)
    except ValueError:
        auc_pr = 0.0
    
    avg_loss = total_loss / max(num_graphs, 1)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
    }


def main():
    parser = argparse.ArgumentParser(description='Train link prediction model on PHEME dataset')
    
    parser.add_argument('--visualize', type=bool, default=False,
                       help='visualize input graph, labels, node embeddings & attention weights')
    
    # Dataset arguments
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of PHEME dataset')
    parser.add_argument('--max-threads', type=int, default=None,
                       help='Maximum number of threads to load (None = all)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of stories for training')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Ratio of stories for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Ratio of stories for testing')
    parser.add_argument('--split-seed', type=int, default=42,
                       help='Random seed for story splitting')
    parser.add_argument('--user-edge-type', type=str, default='replies',
                       choices=['mentions', 'replies', 'both', 'none'],
                       help='How to construct user-user edges')
    parser.add_argument('--num-negative-samples', type=int, default=1,
                       help='Number of negative samples per positive sample')
    
    # Model arguments
    parser.add_argument('--hidden-channels', type=int, default=64,
                       help='Hidden dimension for GAT layers')
    parser.add_argument('--out-channels', type=int, default=32,
                       help='Output dimension for node embeddings')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of GAT layers')
    parser.add_argument('--heads', type=int, default=2,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    parser.add_argument('--link-pred-hidden-dim', type=int, default=64,
                       help='Hidden dimension for link predictor')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seeds
    torch.manual_seed(args.split_seed)
    np.random.seed(args.split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.split_seed)
    
    # Split stories
    train_stories, val_stories, test_stories = split_stories(
        ALL_STORIES,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    
    print(f"Train stories ({len(train_stories)}): {train_stories}")
    print(f"Val stories ({len(val_stories)}): {val_stories}")
    print(f"Test stories ({len(test_stories)}): {test_stories}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = create_story_split_datasets(
        root=args.data_root,
        train_stories=train_stories,
        val_stories=val_stories,
        test_stories=test_stories,
        max_threads=args.max_threads,
        user_edge_type=args.user_edge_type,
        num_negative_samples=args.num_negative_samples,
        seed=args.split_seed,
    )
    
    print(f"Train dataset: {len(train_dataset)} graphs")
    print(f"Val dataset: {len(val_dataset)} graphs")
    print(f"Test dataset: {len(test_dataset)} graphs")
    
    # Create data loaders
    # Note: For heterogeneous graphs with link prediction, batch_size=1 is often simpler
    # If batch_size > 1, PyG needs to handle edge_label_index offsets correctly
    # We use batch_size=1 for reliability, but you can increase if you implement custom collate
    effective_batch_size = 1 if args.batch_size > 1 else args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False)
    
    if args.batch_size > 1:
        print(f"Warning: Using batch_size=1 for heterogeneous link prediction. "
              f"Requested batch_size={args.batch_size} ignored.")
    
    # Get feature dimensions from first graph
    sample_graph = train_dataset[0]
    in_channels_dict = {
        node_type: sample_graph[node_type].x.shape[1] 
        for node_type in sample_graph.node_types
    }
    
    print(f"Input feature dimensions: {in_channels_dict}")
    
    # Create model
    model = HeteroGATLinkPrediction(
        in_channels_dict=in_channels_dict,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        link_pred_hidden_dim=args.link_pred_hidden_dim,
    ).to(args.device)
    
    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_metrics = []
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device, args.visualize)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics_dict = evaluate(model, val_loader, args.device)
        val_loss = val_metrics_dict['loss']
        val_losses.append(val_loss)
        val_metrics.append(val_metrics_dict)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        # TODO: get train acc
        # print(f"  Train Accuracy: {train_metrics_dict['accuracy']:.4f}")  # Added to display training accuracy
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_metrics_dict['accuracy']:.4f}")
        print(f"  Val AUC-ROC: {val_metrics_dict['auc_roc']:.4f}")
        print(f"  Val AUC-PR: {val_metrics_dict['auc_pr']:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics_dict,
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics_dict,
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print()
    
    # Load best model and evaluate on test set
    print("Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, args.device)
    
    print("Test Metrics:")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  Test AUC-PR: {test_metrics['auc_pr']:.4f}")
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

