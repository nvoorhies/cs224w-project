#!/usr/bin/env python3
"""
Graph Visualization Examples.

This script demonstrates how to visualize PHEME dataset graphs using
NetworkX, matplotlib, and plotly for interactive exploration.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    load_tweet_thread,
    thread_to_graph,
    visualize_graph,
    visualize_graph_static,
    visualize_graph_interactive,
    print_graph_info,
    hetero_to_networkx,
)


def example_1_basic_visualization():
    """Example 1: Basic static visualization."""
    print("=" * 80)
    print("Example 1: Basic Static Visualization")
    print("=" * 80)
    print()

    # Load a thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    
    if not thread_path.exists():
        print(f"⚠ Thread path does not exist: {thread_path}")
        print("Please ensure the PHEME dataset is available in the data directory.")
        return
    
    print(f"Loading thread from: {thread_path.name}")
    thread = load_tweet_thread(thread_path)
    print(f"Thread has {len(thread.get_all_tweets())} tweets and {len(thread.get_all_users())} users")
    print()

    # Convert to graph
    graph = thread_to_graph(thread, user_edge_type="replies")
    
    # Print graph information
    print_graph_info(graph)
    
    # Create static visualization
    print("Creating static visualization...")
    fig = visualize_graph_static(
        graph,
        layout='spring',
        node_size=500,
        figsize=(14, 10),
        show_labels=True,
        save_path="graph_static.png"
    )
    print("Static graph saved to graph_static.png")
    print("Close the matplotlib window to continue...")
    import matplotlib.pyplot as plt
    plt.show()


def example_2_interactive_visualization():
    """Example 2: Interactive visualization with plotly."""
    print("=" * 80)
    print("Example 2: Interactive Visualization")
    print("=" * 80)
    print()

    # Load a thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    
    if not thread_path.exists():
        print(f"⚠ Thread path does not exist: {thread_path}")
        return
    
    thread = load_tweet_thread(thread_path)
    graph = thread_to_graph(thread, user_edge_type="both")  # Include mentions and replies
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    fig = visualize_graph_interactive(
        graph,
        layout='spring',
        node_size=15,
        height=900,
        show_labels=True,
        save_html="graph_interactive.html"
    )
    print("Interactive graph saved to graph_interactive.html")
    print("Opening in browser...")
    fig.show()


def example_3_networkx_analysis():
    """Example 3: NetworkX graph analysis."""
    print("=" * 80)
    print("Example 3: NetworkX Graph Analysis")
    print("=" * 80)
    print()

    # Load a thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    
    if not thread_path.exists():
        print(f"⚠ Thread path does not exist: {thread_path}")
        return
    
    thread = load_tweet_thread(thread_path)
    graph = thread_to_graph(thread, user_edge_type="replies")
    
    # Convert to NetworkX
    import networkx as nx
    G = hetero_to_networkx(graph, include_reverse_edges=False)
    
    print("NetworkX Graph Analysis:")
    print(f"  Number of nodes: {G.number_of_nodes()}")
    print(f"  Number of edges: {G.number_of_edges()}")
    print()
    
    # Analyze node types
    node_types = nx.get_node_attributes(G, 'node_type')
    tweet_nodes = [n for n, t in node_types.items() if t == 'tweet']
    user_nodes = [n for n, t in node_types.items() if t == 'user']
    
    print(f"  Tweet nodes: {len(tweet_nodes)}")
    print(f"  User nodes: {len(user_nodes)}")
    print()
    
    # Analyze edge types
    edge_relations = [attr['relation'] for _, _, attr in G.edges(data=True)]
    from collections import Counter
    relation_counts = Counter(edge_relations)
    
    print("Edge relations:")
    for rel, count in relation_counts.items():
        print(f"  {rel:20s}: {count:4d} edges")
    print()
    
    # Centrality measures (for tweet nodes only, as example)
    if tweet_nodes:
        tweet_subgraph = G.subgraph(tweet_nodes)
        if len(tweet_subgraph) > 1:
            try:
                degree_centrality = nx.degree_centrality(tweet_subgraph)
                top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Top 5 tweet nodes by degree centrality:")
                for node, centrality in top_nodes:
                    print(f"  {node:30s}: {centrality:.4f}")
            except Exception as e:
                print(f"Could not compute centrality: {e}")


def example_4_different_layouts():
    """Example 4: Visualize with different layout algorithms."""
    print("=" * 80)
    print("Example 4: Different Layout Algorithms")
    print("=" * 80)
    print()

    # Load a thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    
    if not thread_path.exists():
        print(f"⚠ Thread path does not exist: {thread_path}")
        return
    
    thread = load_tweet_thread(thread_path)
    graph = thread_to_graph(thread, user_edge_type="replies")
    
    layouts = ['spring', 'circular', 'kamada_kawai']
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(layouts), figsize=(18, 6))
    
    for i, layout in enumerate(layouts):
        print(f"Creating {layout} layout...")
        fig_layout = visualize_graph_static(
            graph,
            layout=layout,
            node_size=300,
            figsize=(6, 6),
            show_labels=False
        )
        # Copy to subplot
        # Note: This is a simplified version - in practice you'd recreate for each layout
        axes[i].set_title(f"{layout.capitalize()} Layout")
    
    plt.tight_layout()
    plt.savefig("graph_layouts.png", dpi=300, bbox_inches='tight')
    print("Layouts saved to graph_layouts.png")
    plt.show()


def example_5_custom_colors():
    """Example 5: Custom node and edge colors."""
    print("=" * 80)
    print("Example 5: Custom Colors")
    print("=" * 80)
    print()

    # Load a thread
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    
    if not thread_path.exists():
        print(f"⚠ Thread path does not exist: {thread_path}")
        return
    
    thread = load_tweet_thread(thread_path)
    graph = thread_to_graph(thread, user_edge_type="replies")
    
    # Custom colors
    node_colors = {
        'tweet': '#1DA1F2',  # Twitter blue
        'user': '#FF6B6B',   # Coral red
    }
    
    edge_colors = {
        'replies_to': '#888888',      # Gray
        'posts': '#4ECDC4',            # Teal
        'interacts_with': '#FFE66D',   # Yellow
    }
    
    print("Creating visualization with custom colors...")
    fig = visualize_graph_interactive(
        graph,
        node_colors=node_colors,
        edge_colors=edge_colors,
        layout='spring',
        height=800,
        save_html="graph_custom_colors.html"
    )
    print("Custom colored graph saved to graph_custom_colors.html")
    fig.show()


def main():
    """Run all examples."""
    print("PHEME Graph Visualization Examples")
    print("=" * 80)
    print()
    print("Available examples:")
    print("  1. Basic static visualization")
    print("  2. Interactive visualization")
    print("  3. NetworkX graph analysis")
    print("  4. Different layout algorithms")
    print("  5. Custom colors")
    print()
    
    try:
        choice = input("Enter example number (1-5) or 'all' to run all: ").strip().lower()
        
        if choice == '1':
            example_1_basic_visualization()
        elif choice == '2':
            example_2_interactive_visualization()
        elif choice == '3':
            example_3_networkx_analysis()
        elif choice == '4':
            example_4_different_layouts()
        elif choice == '5':
            example_5_custom_colors()
        elif choice == 'all':
            example_1_basic_visualization()
            example_2_interactive_visualization()
            example_3_networkx_analysis()
            example_4_different_layouts()
            example_5_custom_colors()
        else:
            print("Invalid choice. Running example 2 (interactive visualization)...")
            example_2_interactive_visualization()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



