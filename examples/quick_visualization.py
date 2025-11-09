#!/usr/bin/env python3
"""
Quick visualization example for PHEME graphs.

This is a simple script to quickly visualize a graph from the PHEME dataset.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    try:
        from src import load_tweet_thread, thread_to_graph, visualize_graph_interactive
    except ImportError as e:
        print(f"Error importing visualization functions: {e}")
        print("\nPlease install visualization dependencies:")
        print("  uv pip install -e '.[viz]'")
        sys.exit(1)
    
    # Example thread path (adjust as needed)
    thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
    
    if not thread_path.exists():
        print(f"⚠ Thread path does not exist: {thread_path}")
        print("\nPlease ensure:")
        print("1. The PHEME dataset is downloaded")
        print("2. The data directory structure is correct")
        print("3. Adjust the thread_path in this script if needed")
        return
    
    print("Loading thread...")
    thread = load_tweet_thread(thread_path)
    print(f"✓ Loaded thread with {len(thread.get_all_tweets())} tweets and {len(thread.get_all_users())} users")
    
    print("Converting to graph...")
    graph = thread_to_graph(thread, user_edge_type="replies")
    print("✓ Graph created")
    
    print("Creating interactive visualization...")
    print("  (This will open in your browser)")
    fig = visualize_graph_interactive(
        graph,
        layout='spring',
        node_size=15,
        height=900,
        show_labels=True,
        save_html="graph_interactive.html"
    )
    
    print("✓ Interactive graph saved to graph_interactive.html")
    print("  Opening in browser...")
    fig.show()
    
    print("\n✓ Done! You can:")
    print("  - Interact with the graph in your browser")
    print("  - Zoom, pan, and hover over nodes")
    print("  - Share the HTML file with others")

if __name__ == "__main__":
    main()



