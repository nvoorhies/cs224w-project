import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_input(title, edge_index_dict, edge_label_index = None, edge_label = None):
    """Visualize input graph structure for the given thread with edges defined in edge_index_dict."""
    G = nx.DiGraph()

    # Add edges from edge_index_dict
    for edge_type, edge_index in edge_index_dict.items():
        for i in range(edge_index.shape[1]):
            src = (edge_type[0], edge_index[0, i].item())
            dst = (edge_type[2], edge_index[1, i].item())
            G.add_edge(src, dst, color='grey', edge_type=edge_type)

    if edge_label_index != None:  # include label edges in the visualization
        # Add edges from edge_label_index with colors based on edge_label
        for i in range(edge_label_index.shape[1]):
            src = ('user', edge_label_index[0, i].item())
            dst = ('tweet', edge_label_index[1, i].item())
            label = edge_label[i].item()
            color = 'green' if label == 1 else 'orange'
            G.add_edge(src, dst, label=label, color=color)

    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # Color nodes by type
    node_colors = []
    for node in G.nodes():
        if node[0] == 'tweet':
            node_colors.append('green')
        else:  # user
            node_colors.append('purple')
    
    # Draw the graph
    pos = nx.spring_layout(G, k=0.5, seed=42)  # Adjust layout to reduce overlap

    # Offset reverse edges slightly for visibility
    for edge in G.edges:
        if (edge[1], edge[0]) in G.edges:  # Check for reverse edge
            pos[edge[0]] += (0.5, 0.5)  # Offset source node slightly
            pos[edge[1]] -= (0.5, 0.5)  # Offset target node slightly

    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, 
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    

def visualize_graph(h_dict, att_dict):
    """
    Visualize node embeddings and edge attention weights.

    Args:
        h_dict: Dictionary of node embeddings {node_type: Tensor}.
        att_dict: Dictionary of edge attention weights {edge_type: (edge_index, attention_weights)}.
        edge_index_dict: Dictionary of edge indices {edge_type: Tensor}.
    """
    # Create a graph
    G = nx.DiGraph()

    # Add nodes with embedding-based color
    node_colors = {}
    node_outlines = {}
    for node_type, embeddings in h_dict.items():
        for i, embedding in enumerate(embeddings):
            # Normalize embedding for color scaling
            color_intensity = np.linalg.norm(embedding.detach().numpy())
            node_colors[(node_type, i)] = color_intensity

            if node_type == 'tweet':
                outline_color = "green"
            else:
                outline_color = "purple"
            node_outlines[(node_type, i)] = outline_color
            G.add_node((node_type, i), color=color_intensity)

    # Add edges with attention-based color
    edge_colors = []
    edge_styles = []  # Store edge styles (e.g., thickness)
    edge_list = []
    for edge_type, (edge_index, att_weights) in att_dict.items():
        for i, (src, dst) in enumerate(edge_index.T.detach().numpy()):
            # Use (node_type, node_id) format for edges
            src_node = (edge_type[0], src)
            dst_node = (edge_type[2], dst)

            # Ensure nodes exist in the graph
            if src_node in G.nodes and dst_node in G.nodes:
                # Normalize attention weight for color scaling
                weight = att_weights[i][0].item()  # use the first head
                if weight == 1.0:
                    edge_colors.append("black")  # Black for weight 1
                    edge_styles.append(0.5)  # Thin edge for weight 1
                else:
                    edge_colors.append(plt.cm.Reds(weight / 2))  # Map weight to RGBA color
                    edge_styles.append(2.0)  # Thicker edge for other weights
                G.add_edge(src_node, dst_node, weight=weight)
                edge_list.append((src_node, dst_node))

    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)  # Adjust layout to reduce overlap

    # Offset reverse edges slightly for visibility
    for edge in G.edges:
        if (edge[1], edge[0]) in G.edges:  # Check for reverse edge
            pos[edge[0]] += (0.5, 0.5)  # Offset source node slightly
            pos[edge[1]] -= (0.5, 0.5)  # Offset target node slightly

    # Draw nodes with black outlines
    node_color_values = [node_colors[node] for node in G.nodes]
    node_outline_values = [node_outlines[node] for node in G.nodes]

    nx.draw_networkx_nodes(
        G, pos, node_color=node_color_values, cmap=plt.cm.Blues, node_size=300,
        edgecolors=node_outline_values
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list, edge_color=edge_colors, width=edge_styles,
        connectionstyle="arc3,rad=0.2"  # Add curvature to edges for better visibility
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)


    # Add colorbars for node colors
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_color_values), vmax=max(node_color_values)))
    sm_nodes._A = []
    plt.colorbar(sm_nodes, label="Node Embedding Intensity (Blue)", ax=plt.gca())

    # Add colorbars for edge colors
    sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm_edges._A = []
    plt.colorbar(sm_edges, label="Edge Attention Weight (Red)", ax=plt.gca())

    plt.axis("off")
    plt.title("Edge Attention Weights")
    # for edge_type, (edge_index, att_weights) in att_dict.items():
    #     print(f"Edge Type: {edge_type}")
    #     print(f"Edge Index: {edge_index}")
    #     print(f"Attention Weights: {att_weights}")

    plt.show()
