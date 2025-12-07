import networkx as nx
import matplotlib.pyplot as plt
import json
import os

def visualize_thread(graph_data):
    """
    Visualize a graph for a thread with a legend for node and edge types.

    Parameters:
        graph_data (dict): A dictionary containing nodes and edges information.
    """
    # Create a directed graph
    G = nx.Graph()  # Use an undirected graph

    # Add nodes with their types
    for node in graph_data["nodes"]:
        G.add_node(node["id"], type=node["type"])

    # Add edges with their types
    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], type=edge["type"])

    # Define node colors based on type
    node_colors = {
        "user": "blue",
        "tweet": "green",
        "reply": "orange"
    }

    # Map node colors
    colors = [node_colors[G.nodes[n]["type"]] for n in G.nodes]

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=True, labels={n: n for n in G.nodes}, node_color=colors, node_size=500, font_size=10, font_color="black"
    )

    # Draw edges with styles and colors
    edge_styles = {
        "authored": "dotted",
        "replied": "dashed"
    }
    edge_colors = {
        "authored": "blue",
        "replied": "orange"
    }
    for edge in G.edges(data=True):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(edge[0], edge[1])],
            style=edge_styles.get(edge[2]["type"], "solid"),
            edge_color=edge_colors.get(edge[2]["type"], "black")
        )

    # Create a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='User', markersize=10, markerfacecolor='blue'),
        plt.Line2D([0], [0], marker='o', color='w', label='Tweet', markersize=10, markerfacecolor='green'),
        plt.Line2D([0], [0], marker='o', color='w', label='Reply', markersize=10, markerfacecolor='orange'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Authored'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Replied')
    ]
    plt.legend(handles=legend_elements, loc='best')

    # Show the plot
    plt.title("Thread Graph Visualization")
    plt.show()

def load_thread_structure(file_path):
    """Load thread structure from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def parse_structure_to_graph(structure, user_mapping):
    """
    Parse the thread structure into nodes and edges.

    Parameters:
        structure (dict): Thread structure as a nested dictionary.
        user_mapping (dict): Mapping of tweet/reply IDs to user IDs.

    Returns:
        dict: Graph data with nodes and edges.
    """
    nodes = []
    edges = []

    def add_nodes_and_edges(parent_id, children, is_root=False):
        if isinstance(children, dict):  # Ensure children is a dictionary
            for child_id, grandchildren in children.items():
                # Add tweet/reply nodes
                if is_root:
                    nodes.append({"id": parent_id, "type": "tweet"})  # Root tweet
                else:
                    nodes.append({"id": parent_id, "type": "reply"})  # Replies

                nodes.append({"id": child_id, "type": "reply"})

                # Add user nodes for authorship using user_mapping
                # Fetch user ID from JSON file for parent_id
                user_id_parent = user_mapping.get(parent_id)
                if not user_id_parent:
                    parent_file = os.path.join(user_mapping_file, f"{parent_id}.json")
                    if os.path.exists(parent_file):
                        with open(parent_file, "r") as f:
                            parent_data = json.load(f)
                            user_id_parent = parent_data["user"]["id"]
                            user_mapping[parent_id] = user_id_parent
                    else:
                        user_id_parent = f"user_{parent_id}"

                # Fetch user ID from JSON file for child_id
                user_id_child = user_mapping.get(child_id)
                if not user_id_child:
                    child_file = os.path.join(user_mapping_file, f"{child_id}.json")
                    if os.path.exists(child_file):
                        with open(child_file, "r") as f:
                            child_data = json.load(f)
                            user_id_child = child_data["user"]["id"]
                            user_mapping[child_id] = user_id_child
                    else:
                        user_id_child = f"user_{child_id}"

                nodes.append({"id": user_id_parent, "type": "user"})
                nodes.append({"id": user_id_child, "type": "user"})

                # Add authorship edges (undirected)
                edges.append({"source": user_id_parent, "target": parent_id, "type": "authored"})
                edges.append({"source": user_id_child, "target": child_id, "type": "authored"})

                # Add reply edge (undirected)
                edges.append({"source": parent_id, "target": child_id, "type": "replied"})

                # Recursively add grandchildren
                add_nodes_and_edges(child_id, grandchildren)
        elif isinstance(children, list):  # Handle empty lists
            pass

    for root_id, replies in structure.items():
        # Add root tweet node
        nodes.append({"id": root_id, "type": "tweet"})

        # Add user node for root tweet using user_mapping
        user_id_root = user_mapping.get(root_id)
        if not user_id_root:
            root_file = os.path.join(os.path.dirname(user_mapping_file), "source-tweets", f"{root_id}.json")
            if os.path.exists(root_file):
                with open(root_file, "r") as f:
                    root_data = json.load(f)
                    user_id_root = root_data["user"]["id"]
                    user_mapping[root_id] = user_id_root
            else:
                user_id_root = f"user_{root_id}"
        nodes.append({"id": user_id_root, "type": "user"})

        # Add authorship edge for root tweet (undirected)
        edges.append({"source": user_id_root, "target": root_id, "type": "authored"})

        add_nodes_and_edges(root_id, replies, is_root=True)

    # Remove duplicate nodes
    unique_nodes = {node["id"]: node for node in nodes}.values()

    return {"nodes": list(unique_nodes), "edges": edges}

# Example usage
if __name__ == "__main__":
    structure_file = "data/all-rnr-annotated-threads/charliehebdo-all-rnr-threads/rumours/552783238415265792/structure.json"
    user_mapping_file = "data/all-rnr-annotated-threads/charliehebdo-all-rnr-threads/rumours/552783238415265792/reactions"

    # Load structure
    structure = load_thread_structure(structure_file)

    # Load user mapping from reaction files
    user_mapping = {}
    for reaction_file in os.listdir(user_mapping_file):
        if reaction_file.endswith(".json"):
            with open(os.path.join(user_mapping_file, reaction_file), "r") as f:
                reaction_data = json.load(f)
                tweet_id = reaction_data["id"]
                user_id = reaction_data["user"]["id"]
                user_mapping[tweet_id] = user_id

    graph_data = parse_structure_to_graph(structure, user_mapping)

    # Visualize the graph
    visualize_thread(graph_data)