"""
Heterogeneous Graph Attention Transformer Model for Link Prediction.

This module implements a Graph Attention Transformer (GAT) model adapted for
temporal heterogeneous networks, specifically for link prediction tasks on
the PHEME dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional


class HeteroGATLayer(nn.Module):
    """
    Single layer of heterogeneous GAT using PyTorch Geometric's HeteroConv.
    
    Applies GAT convolution to each edge type independently.
    """
    
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        heads: int = 2,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
    ):
        """
        Initialize HeteroGATLayer.
        
        Args:
            in_channels_dict: Dictionary mapping node type to input feature dim
            out_channels: Output dimension for each head
            heads: Number of attention heads
            dropout: Dropout probability
            negative_slope: Negative slope for LeakyReLU
            add_self_loops: Whether to add self-loops
        """
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        
        # Create a dictionary of GAT layers for each edge type
        conv_dict = {}
        
        # Tweet -> Tweet edges (homogeneous)
        if 'tweet' in in_channels_dict:
            tweet_dim = in_channels_dict['tweet']
            conv_dict[('tweet', 'replies_to', 'tweet')] = GATConv(
                tweet_dim, out_channels, heads=heads, dropout=dropout,
                negative_slope=negative_slope, add_self_loops=add_self_loops
            )
            conv_dict[('tweet', 'replied_by', 'tweet')] = GATConv(
                tweet_dim, out_channels, heads=heads, dropout=dropout,
                negative_slope=negative_slope, add_self_loops=add_self_loops
            )
        
        # User -> Tweet edges (heterogeneous)
        if 'user' in in_channels_dict and 'tweet' in in_channels_dict:
            user_dim = in_channels_dict['user']
            tweet_dim = in_channels_dict['tweet']
            conv_dict[('user', 'posts', 'tweet')] = GATConv(
                (user_dim, tweet_dim), out_channels, heads=heads, dropout=dropout,
                negative_slope=negative_slope, add_self_loops=False
            )
            conv_dict[('tweet', 'posted_by', 'user')] = GATConv(
                (tweet_dim, user_dim), out_channels, heads=heads, dropout=dropout,
                negative_slope=negative_slope, add_self_loops=False
            )
        
        # User -> User edges (homogeneous)
        if 'user' in in_channels_dict:
            user_dim = in_channels_dict['user']
            conv_dict[('user', 'interacts_with', 'user')] = GATConv(
                user_dim, out_channels, heads=heads, dropout=dropout,
                negative_slope=negative_slope, add_self_loops=add_self_loops
            )
            conv_dict[('user', 'interacted_by', 'user')] = GATConv(
                user_dim, out_channels, heads=heads, dropout=dropout,
                negative_slope=negative_slope, add_self_loops=add_self_loops
            )
        
        # Use HeteroConv to wrap the individual conv layers
        self.conv = HeteroConv(conv_dict, aggr='mean')
        # Store supported edge types for filtering
        self.supported_edge_types = set(conv_dict.keys())
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the heterogeneous GAT layer.
        
        Args:
            x_dict: Dictionary mapping node type to node features
            edge_index_dict: Dictionary mapping edge type to edge indices
            
        Returns:
            Dictionary mapping node type to updated node features
        """
        # Filter edge_index_dict to only include edge types we have layers for
        filtered_edge_index_dict = {
            edge_type: edge_index 
            for edge_type, edge_index in edge_index_dict.items()
            if edge_type in self.supported_edge_types
        }
        
        # If no valid edges, return input features
        if not filtered_edge_index_dict:
            return x_dict
        
        # HeteroConv automatically aggregates across edge types
        return self.conv(x_dict, filtered_edge_index_dict)


class TemporalHeteroGAT(nn.Module):
    """
    Temporal Heterogeneous Graph Attention Network for link prediction.
    
    This model processes heterogeneous graphs with temporal information
    to predict links between tweets and users.
    """
    
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.5,
        negative_slope: float = 0.2,
    ):
        """
        Initialize TemporalHeteroGAT.
        
        Args:
            in_channels_dict: Dictionary mapping node type to input feature dim
            hidden_channels: Hidden dimension for GAT layers
            out_channels: Output dimension for node embeddings
            num_layers: Number of GAT layers
            heads: Number of attention heads per layer
            dropout: Dropout probability
            negative_slope: Negative slope for LeakyReLU
        """
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Input projection layers
        self.input_proj = nn.ModuleDict()
        for node_type, in_dim in in_channels_dict.items():
            self.input_proj[node_type] = Linear(in_dim, hidden_channels * heads)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer: hidden_channels * heads -> hidden_channels * heads
                layer_in_dict = {
                    node_type: hidden_channels * heads 
                    for node_type in in_channels_dict.keys()
                }
            else:
                # Subsequent layers: hidden_channels * heads -> hidden_channels * heads
                layer_in_dict = {
                    node_type: hidden_channels * heads 
                    for node_type in in_channels_dict.keys()
                }
            
            self.gat_layers.append(
                HeteroGATLayer(
                    layer_in_dict,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout if i < num_layers - 1 else 0.0,
                    negative_slope=negative_slope,
                )
            )
        
        # Output projection layers
        self.output_proj = nn.ModuleDict()
        for node_type in in_channels_dict.keys():
            self.output_proj[node_type] = Linear(
                hidden_channels * heads, out_channels
            )
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x_dict: Dictionary mapping node type to node features
            edge_index_dict: Dictionary mapping edge type to edge indices
            
        Returns:
            Dictionary mapping node type to final node embeddings
        """
        # Input projection
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.input_proj[node_type](x)
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h_dict = gat_layer(h_dict, edge_index_dict)
            if i < self.num_layers - 1:
                # Apply activation and dropout (except last layer)
                h_dict = {
                    node_type: F.relu(h) 
                    for node_type, h in h_dict.items()
                }
                h_dict = {
                    node_type: F.dropout(h, p=self.dropout, training=self.training)
                    for node_type, h in h_dict.items()
                }
        
        # Output projection
        out_dict = {}
        for node_type, h in h_dict.items():
            out_dict[node_type] = self.output_proj[node_type](h)
        
        return out_dict
    
    @property
    def dropout(self) -> float:
        """Get dropout rate from first GAT layer."""
        if len(self.gat_layers) > 0:
            return self.gat_layers[0].dropout
        return 0.0


class LinkPredictor(nn.Module):
    """
    Link prediction head for predicting follow requests.
    
    Takes node embeddings and predicts the probability of a link
    between a tweet and a user (e.g., a reply tweet triggers a follow request).
    """
    
    def __init__(
        self,
        node_emb_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.5,
    ):
        """
        Initialize LinkPredictor.
        
        Args:
            node_emb_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout probability
        """
        super().__init__()
        self.node_emb_dim = node_emb_dim
        
        # MLP for link prediction
        self.mlp = nn.Sequential(
            nn.Linear(node_emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict link probability between source and destination nodes.
        
        Args:
            src_emb: Source node embeddings [num_pairs, node_emb_dim]
            dst_emb: Destination node embeddings [num_pairs, node_emb_dim]
            
        Returns:
            Link probability scores [num_pairs, 1]
        """
        # Concatenate source and destination embeddings
        pair_emb = torch.cat([src_emb, dst_emb], dim=1)
        
        # Predict link probability
        logits = self.mlp(pair_emb)
        
        return logits.squeeze(-1)  # [num_pairs]


class HeteroGATLinkPrediction(nn.Module):
    """
    Complete model for tweet-to-user link prediction on heterogeneous graphs.
    
    Combines TemporalHeteroGAT for node embeddings and LinkPredictor
    for link prediction.
    """
    
    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.5,
        negative_slope: float = 0.2,
        link_pred_hidden_dim: int = 64,
    ):
        """
        Initialize HeteroGATLinkPrediction.
        
        Args:
            in_channels_dict: Dictionary mapping node type to input feature dim
            hidden_channels: Hidden dimension for GAT layers
            out_channels: Output dimension for node embeddings
            num_layers: Number of GAT layers
            heads: Number of attention heads per layer
            dropout: Dropout probability
            negative_slope: Negative slope for LeakyReLU
            link_pred_hidden_dim: Hidden dimension for link predictor
        """
        super().__init__()
        
        # GAT encoder
        self.encoder = TemporalHeteroGAT(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            negative_slope=negative_slope,
        )
        
        # Link predictor for tweet->user links
        self.link_predictor = LinkPredictor(
            node_emb_dim=out_channels,
            hidden_dim=link_pred_hidden_dim,
            dropout=dropout,
        )
        
        self.out_channels = out_channels
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_label_index: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x_dict: Dictionary mapping node type to node features
            edge_index_dict: Dictionary mapping edge type to edge indices
            edge_label_index: Edge indices for link prediction [2, num_edges]
                             If None, only returns node embeddings
            
        Returns:
            Tuple of (node_embeddings_dict, link_predictions)
        """
        # Get node embeddings
        node_emb_dict = self.encoder(x_dict, edge_index_dict)
        
        # Predict links if edge_label_index is provided
        link_pred = None
        if edge_label_index is not None:
            tweet_emb = node_emb_dict['tweet']
            user_emb = node_emb_dict['user']
            
            src_emb = user_emb[edge_label_index[0]]
            dst_emb = tweet_emb[edge_label_index[1]]
            
            link_pred = self.link_predictor(src_emb, dst_emb)
        
        return node_emb_dict, link_pred
    
    def encode_nodes(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode nodes to embeddings without link prediction.
        
        Args:
            x_dict: Dictionary mapping node type to node features
            edge_index_dict: Dictionary mapping edge type to edge indices
            
        Returns:
            Dictionary mapping node type to node embeddings
        """
        return self.encoder(x_dict, edge_index_dict)

