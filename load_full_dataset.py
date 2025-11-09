"""
Load the entire PHEME dataset.

This script loads all threads from all 9 news stories in the PHEME dataset,
following the instructions in the README.
"""

from pathlib import Path
from src import load_dataset
from src.link_prediction_dataset import ALL_STORIES


def load_full_dataset(data_root: str = "data/all-rnr-annotated-threads"):
    """
    Load the entire PHEME dataset from all stories.
    
    This follows the README pattern:
    ```python
    from src import load_dataset
    from pathlib import Path
    
    story_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads")
    threads = load_dataset(
        story_path,
        load_reactions=True,
        max_threads=10,
        skip_errors=True
    )
    ```
    
    Args:
        data_root: Root directory of the PHEME dataset
        
    Returns:
        Dictionary mapping story name to list of threads
    """
    data_path = Path(data_root)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    all_threads = {}
    total_threads = 0
    
    print("Loading PHEME dataset from all stories...")
    print("=" * 80)
    
    for story in ALL_STORIES:
        story_path = data_path / story
        if not story_path.exists():
            print(f"⚠️  Story not found: {story}")
            continue
        
        print(f"\nLoading {story}...")
        # Following README pattern: load_dataset with load_reactions=True, skip_errors=True
        threads = load_dataset(
            story_path,
            load_reactions=True,
            max_threads=None,  # Load all threads (None = no limit)
            skip_errors=True
        )
        
        all_threads[story] = threads
        total_threads += len(threads)
        print(f"  Loaded {len(threads)} threads")
    
    print("=" * 80)
    print(f"\nTotal threads loaded: {total_threads}")
    print(f"Stories processed: {len(all_threads)}/{len(ALL_STORIES)}")
    
    # Print summary by story
    print("\nSummary by story:")
    for story, threads in all_threads.items():
        print(f"  {story}: {len(threads)} threads")
    
    return all_threads


if __name__ == '__main__':
    import sys
    
    # Allow custom data root path
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/all-rnr-annotated-threads"
    
    try:
        print(f"Looking for dataset at: {data_root}")
        all_threads = load_full_dataset(data_root)
        print("\n✅ Dataset loading complete!")
        print(f"\nYou can now use this dataset for training.")
        print(f"To train the model, run:")
        print(f"  uv run python train_link_prediction.py --data-root {data_root}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease make sure the PHEME dataset is downloaded and placed in:")
        print(f"  {data_root}/")
        print("\nThe dataset should contain the following directories:")
        for story in ALL_STORIES:
            print(f"  - {story}/")
        print("\nYou can download the dataset from:")
        print("  https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/2068650")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

