"""
Heterogeneous Graph Attention Transformer Model for Link Prediction.

This module implements a Graph Attention Transformer (GAT) model adapted for
temporal heterogeneous networks, specifically for link prediction tasks on
the PHEME dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GraphConv, Linear, TransformerConv, GCN
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List

import warnings
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops


class GraphConvWithAttention(GraphConv):
    """GraphConv wrapper that returns uniform attention weights for compatibility.

    This wrapper enables GraphConv to work with the same interface as TransformerConv
    by optionally returning degree-normalized uniform attention weights. This maintains
    API compatibility with the heterogeneous GAT implementation.
    """

    def forward(self, x, edge_index, edge_weight=None, size=None,
                return_attention_weights=None):
        """Forward pass with optional attention weight generation.

        Args:
            x: Input node features
            edge_index: Edge indices
            edge_weight: Optional edge weights
            size: Size of bipartite graph (for heterogeneous edges)
            return_attention_weights: If True, return uniform attention weights

        Returns:
            If return_attention_weights is True:
                Tuple of (output, (edge_index, attention_weights))
            Otherwise:
                output
        """
        # Call parent forward
        out = super().forward(x, edge_index, edge_weight, size)

        # If attention weights requested, generate degree-normalized uniform weights
        if return_attention_weights:
            from torch_geometric.utils import degree

            num_edges = edge_index.shape[1]

            # Determine number of destination nodes
            if isinstance(x, tuple):
                num_dst = x[1].shape[0] if x[1] is not None else x[0].shape[0]
            else:
                num_dst = x.shape[0]

            # Calculate in-degree for normalization (avoid div by zero)
            in_deg = degree(edge_index[1], num_dst, dtype=torch.float)

            # Create uniform weights normalized by in-degree: sum to 1.0 per dst node
            uniform_weights = torch.zeros(num_edges, 1,
                                         dtype=torch.float,
                                         device=edge_index.device)
            for i in range(num_edges):
                dst = edge_index[1, i]
                # Use max(deg, 1.0) to handle zero in-degree nodes
                uniform_weights[i] = 1.0 / max(in_deg[dst].item(), 1.0)

            return out, (edge_index, uniform_weights)

        return out


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
        conv_type: str = 'transformer',
        skip_connections: bool = True,
    ):
        """
        Initialize HeteroGATLayer.

        Args:
            in_channels_dict: Dictionary mapping node type to input feature dim
            out_channels: Output dimension for each head
            heads: Number of attention heads
            dropout: Dropout probability
            negative_slope: Negative slope for LeakyReLU (used by GATConv)
            add_self_loops: Whether to add self-loops (used by GATConv)
            conv_type: Type of attention layer ('transformer' or 'gat')
        """
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv_type = conv_type

        # Create a dictionary of attention layers for each edge type
        conv_dict = {}

        # Tweet -> Tweet edges (homogeneous)
        if 'tweet' in in_channels_dict:
            tweet_dim = in_channels_dict['tweet']
            if conv_type == 'transformer':
                conv_dict[('tweet', 'replies_to', 'tweet')] = TransformerConv(
                    tweet_dim, out_channels, heads=heads, dropout=dropout, beta=skip_connections
                )
                conv_dict[('tweet', 'replied_by', 'tweet')] = TransformerConv(
                    tweet_dim, out_channels, heads=heads, dropout=dropout, beta=skip_connections
                )
            elif conv_type == 'gat':
                conv_dict[('tweet', 'replies_to', 'tweet')] = GATConv(
                    tweet_dim, out_channels, heads=heads, dropout=dropout,
                    negative_slope=negative_slope, add_self_loops=add_self_loops,
                    residual=skip_connections  # Replacement for beta gating
                )
                conv_dict[('tweet', 'replied_by', 'tweet')] = GATConv(
                    tweet_dim, out_channels, heads=heads, dropout=dropout,
                    negative_slope=negative_slope, add_self_loops=add_self_loops,
                    residual=skip_connections
                )
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")

        # User -> Tweet edges (heterogeneous)
        if 'user' in in_channels_dict and 'tweet' in in_channels_dict:
            user_dim = in_channels_dict['user']
            tweet_dim = in_channels_dict['tweet']
            if conv_type == 'transformer':
                conv_dict[('user', 'posts', 'tweet')] = TransformerConv(
                    (user_dim, tweet_dim), out_channels, heads=heads, dropout=dropout, beta=skip_connections
                )
                conv_dict[('tweet', 'posted_by', 'user')] = TransformerConv(
                    (tweet_dim, user_dim), out_channels, heads=heads, dropout=dropout, beta=skip_connections
                )
            elif conv_type == 'gat':
                # Note: add_self_loops=False for bipartite (heterogeneous) edges
                conv_dict[('user', 'posts', 'tweet')] = GATConv(
                    (user_dim, tweet_dim), out_channels, heads=heads, dropout=dropout,
                    negative_slope=negative_slope, add_self_loops=False,
                    residual=skip_connections
                )
                conv_dict[('tweet', 'posted_by', 'user')] = GATConv(
                    (tweet_dim, user_dim), out_channels, heads=heads, dropout=dropout,
                    negative_slope=negative_slope, add_self_loops=False,
                    residual=skip_connections
                )
        
        # # User -> User edges (homogeneous)
        # if 'user' in in_channels_dict:
        #     user_dim = in_channels_dict['user']
        #     conv_dict[('user', 'interacts_with', 'user')] = TransformerConv(
        #         (tweet_dim, user_dim), out_channels, heads=heads, dropout=dropout, beta=True
        #     )
        #     conv_dict[('user', 'interacted_by', 'user')] = TransformerConv(
        #         (tweet_dim, user_dim), out_channels, heads=heads, dropout=dropout, beta=True
        #     )
        
        # Use HeteroConv to wrap the individual conv layers
        self.conv = HeteroConv(conv_dict, aggr='mean')
        # Store supported edge types for filtering
        self.supported_edge_types = set(conv_dict.keys())
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        return_attention_weights: bool = True):
    # ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the heterogeneous GAT layer.
        
        Args:
            x_dict: Dictionary mapping node type to node features
            edge_index_dict: Dictionary mapping edge type to edge indices
            
        Returns:
            Tuple containing:
                - Dictionary mapping node type to updated node features
                - Dictionary mapping edge type to attention weights
        """
        # Filter edge_index_dict to only include edge types we have layers for
        filtered_edge_index_dict = {
            edge_type: edge_index 
            for edge_type, edge_index in edge_index_dict.items()
            if edge_type in self.supported_edge_types
        }
        
        # If no valid edges, return input features and empty attention weights
        if not filtered_edge_index_dict:
            return x_dict, {}
        
        return_attention_weights_dict = {
            edge_type: return_attention_weights
            for edge_type in edge_index_dict
        }
        
        # HeteroConv automatically aggregates across edge types
        conv_outputs = self.conv(x_dict, filtered_edge_index_dict, return_attention_weights_dict=return_attention_weights_dict)
        
        return conv_outputs  # out_dict, attention_weights_dict


class HeteroGraphConvLayer(nn.Module):
    """
    Single layer of heterogeneous GCN using PyTorch Geometric's HeteroConv.

    Applies GCN convolution to each edge type independently with uniform attention.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        dropout: float = 0.0,
    ):
        """
        Initialize HeteroGraphConvLayer.

        Args:
            in_channels_dict: Dictionary mapping node type to input feature dim
            out_channels: Output dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.out_channels = out_channels
        self.dropout = dropout

        # Create a dictionary of GraphConv layers for each edge type
        conv_dict = {}

        # Tweet -> Tweet edges (homogeneous)
        if 'tweet' in in_channels_dict:
            tweet_dim = in_channels_dict['tweet']
            conv_dict[('tweet', 'replies_to', 'tweet')] = GraphConvWithAttention(
                tweet_dim, out_channels, aggr='add'
            )
            conv_dict[('tweet', 'replied_by', 'tweet')] = GraphConvWithAttention(
                tweet_dim, out_channels, aggr='add'
            )

        # User -> Tweet edges (heterogeneous)
        if 'user' in in_channels_dict and 'tweet' in in_channels_dict:
            user_dim = in_channels_dict['user']
            tweet_dim = in_channels_dict['tweet']
            conv_dict[('user', 'posts', 'tweet')] = GraphConvWithAttention(
                (user_dim, tweet_dim), out_channels, aggr='add'
            )
            conv_dict[('tweet', 'posted_by', 'user')] = GraphConvWithAttention(
                (tweet_dim, user_dim), out_channels, aggr='add'
            )

        # Use HeteroConv to wrap the individual conv layers
        self.conv = HeteroConv(conv_dict, aggr='mean')
        # Store supported edge types for filtering
        self.supported_edge_types = set(conv_dict.keys())

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        return_attention_weights: bool = True):
        """
        Forward pass through the heterogeneous GCN layer.

        Args:
            x_dict: Dictionary mapping node type to node features
            edge_index_dict: Dictionary mapping edge type to edge indices
            return_attention_weights: Whether to return attention weights

        Returns:
            Tuple containing:
                - Dictionary mapping node type to updated node features
                - Dictionary mapping edge type to uniform attention weights
        """
        # Filter edge_index_dict to only include edge types we have layers for
        filtered_edge_index_dict = {
            edge_type: edge_index
            for edge_type, edge_index in edge_index_dict.items()
            if edge_type in self.supported_edge_types
        }

        # If no valid edges, return input features and empty attention weights
        if not filtered_edge_index_dict:
            return x_dict, {}

        return_attention_weights_dict = {
            edge_type: return_attention_weights
            for edge_type in edge_index_dict
        }

        # HeteroConv automatically aggregates across edge types
        conv_outputs = self.conv(x_dict, filtered_edge_index_dict, return_attention_weights_dict=return_attention_weights_dict)

        return conv_outputs  # out_dict, uniform_attention_weights_dict


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
        conv_type: str = 'transformer',
        skip_connections: bool = True,
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
            conv_type: Type of attention layer ('transformer' or 'gat')
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
                    conv_type=conv_type,
                    skip_connections=skip_connections,
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
            h_dict, att_dict = gat_layer(h_dict, edge_index_dict)
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
        
        # if edge_index_dict[("tweet", "replies_to", "tweet")].shape[1] > 5:  # more than this many edges
        #     print()
        #     print("Interesting att_dict with multiple incoming edges splitting attention", att_dict)
        
        return out_dict, att_dict
    
    @property
    def dropout(self) -> float:
        """Get dropout rate from first GAT layer."""
        if len(self.gat_layers) > 0:
            return self.gat_layers[0].dropout
        return 0.0


class TemporalHeteroGCN(nn.Module):
    """
    Temporal Heterogeneous Graph Convolutional Network for link prediction.

    This model processes heterogeneous graphs with temporal information
    to predict links between tweets and users, using GCN layers instead of GAT.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Initialize TemporalHeteroGCN.

        Args:
            in_channels_dict: Dictionary mapping node type to input feature dim
            hidden_channels: Hidden dimension for GCN layers
            out_channels: Output dimension for node embeddings
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Input projection layers - NOTE: NO multiplication by heads for GCN
        self.input_proj = nn.ModuleDict()
        for node_type, in_dim in in_channels_dict.items():
            self.input_proj[node_type] = Linear(in_dim, hidden_channels)

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            # All layers: hidden_channels -> hidden_channels
            layer_in_dict = {
                node_type: hidden_channels
                for node_type in in_channels_dict.keys()
            }

            self.gcn_layers.append(
                HeteroGraphConvLayer(
                    layer_in_dict,
                    hidden_channels,
                    dropout=dropout if i < num_layers - 1 else 0.0,
                )
            )

        # Output projection layers
        self.output_proj = nn.ModuleDict()
        for node_type in in_channels_dict.keys():
            self.output_proj[node_type] = Linear(
                hidden_channels, out_channels
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

        # Apply GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            h_dict, att_dict = gcn_layer(h_dict, edge_index_dict)
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

        return out_dict, att_dict

    @property
    def dropout(self) -> float:
        """Get dropout rate from first GCN layer."""
        if len(self.gcn_layers) > 0:
            return self.gcn_layers[0].dropout
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
        conv_type: str = 'transformer',
        skip_connections: bool = True,
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
            conv_type: Type of attention layer ('transformer' or 'gat')
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
            conv_type=conv_type,
            skip_connections=skip_connections,
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
        node_emb_dict, att_dict = self.encoder(x_dict, edge_index_dict)
        
        # Predict links if edge_label_index is provided
        link_pred = None
        if edge_label_index is not None:
            tweet_emb = node_emb_dict['tweet']
            user_emb = node_emb_dict['user']
            
            src_emb = user_emb[edge_label_index[0]]
            dst_emb = tweet_emb[edge_label_index[1]]
            
            link_pred = self.link_predictor(src_emb, dst_emb)
        
        return node_emb_dict, att_dict, link_pred
    
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


class HeteroGCNLinkPrediction(nn.Module):
    """
    Complete model for tweet-to-user link prediction on heterogeneous graphs.

    Combines TemporalHeteroGCN for node embeddings and LinkPredictor
    for link prediction.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.5,
        link_pred_hidden_dim: int = 64,
    ):
        """
        Initialize HeteroGCNLinkPrediction.

        Args:
            in_channels_dict: Dictionary mapping node type to input feature dim
            hidden_channels: Hidden dimension for GCN layers
            out_channels: Output dimension for node embeddings
            num_layers: Number of GCN layers
            dropout: Dropout probability
            link_pred_hidden_dim: Hidden dimension for link predictor
        """
        super().__init__()

        # GCN encoder
        self.encoder = TemporalHeteroGCN(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
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
            Tuple of (node_embeddings_dict, attention_weights_dict, link_predictions)
        """
        # Get node embeddings
        node_emb_dict, att_dict = self.encoder(x_dict, edge_index_dict)

        # Predict links if edge_label_index is provided
        link_pred = None
        if edge_label_index is not None:
            tweet_emb = node_emb_dict['tweet']
            user_emb = node_emb_dict['user']

            src_emb = user_emb[edge_label_index[0]]
            dst_emb = tweet_emb[edge_label_index[1]]

            link_pred = self.link_predictor(src_emb, dst_emb)

        return node_emb_dict, att_dict, link_pred

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


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HeteroConv(torch.nn.Module):
    r"""A generic wrapper for computing graph convolution on heterogeneous
    graphs.
    This layer will pass messages from source nodes to target nodes based on
    the bipartite GNN layer given for a specific edge type.
    If multiple relations point to the same destination, their results will be
    aggregated according to :attr:`aggr`.
    In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is
    especially useful if you want to apply different message passing modules
    for different edge types.

    .. code-block:: python

        hetero_conv = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, 64),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
            ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
        }, aggr='sum')

        out_dict = hetero_conv(x_dict, edge_index_dict)

        print(list(out_dict.keys()))
        >>> ['paper', 'author']

    Args:
        convs (Dict[Tuple[str, str, str], MessagePassing]): A dictionary
            holding a bipartite
            :class:`~torch_geometric.nn.conv.MessagePassing` layer for each
            individual edge type.
        aggr (str, optional): The aggregation scheme to use for grouping node
            embeddings generated by different relations
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"cat"`, :obj:`None`). (default: :obj:`"sum"`)
    """
    def __init__(
        self,
        convs: Dict[EdgeType, MessagePassing],
        aggr: Optional[str] = "sum",
    ):
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = {key[0] for key in convs.keys()}
        dst_node_types = {key[-1] for key in convs.keys()}
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.",
                stacklevel=2)

        self.convs = ModuleDict(convs)
        self.aggr = aggr

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        *args_dict,
        **kwargs_dict,
    ) -> Tuple[Dict[NodeType, Tensor], Dict[EdgeType, Tuple[Tensor, Tensor]]]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.
            *args_dict (optional): Additional forward arguments of individual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict: Dict[str, List[Tensor]] = {}
        att_dict: Dict[EdgeType, Tuple[Tensor, Tensor]] = {}

        for edge_type, conv in self.convs.items():
            src, rel, dst = edge_type

            has_edge_level_arg = False

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append((
                        value_dict.get(src, None),
                        value_dict.get(dst, None),
                    ))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                if not arg.endswith('_dict'):
                    raise ValueError(
                        f"Keyword arguments in '{self.__class__.__name__}' "
                        f"need to end with '_dict' (got '{arg}')")

                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (
                        value_dict.get(src, None),
                        value_dict.get(dst, None),
                    )

            if not has_edge_level_arg:
                continue

            out, att = conv(*args, **kwargs)

            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)

            att_dict[edge_type] = att  # (edge_index, attention_weights)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict, att_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'



