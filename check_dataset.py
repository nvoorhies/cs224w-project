"""
Check dataset structure and provide guidance.

This script checks if the dataset has the required structure for the link prediction model.
"""

from pathlib import Path


def check_dataset_structure(data_root: str):
    """Check if dataset has the required structure."""
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"❌ Dataset not found at: {data_path}")
        return False
    
    print(f"✅ Dataset found at: {data_path}")
    print("\nChecking dataset structure...")
    
    # Check for stories
    stories_found = []
    expected_stories = [
        'charliehebdo-all-rnr-threads',
        'ebola-essien-all-rnr-threads',
        'ferguson-all-rnr-threads',
        'germanwings-crash-all-rnr-threads',
        'gurlitt-all-rnr-threads',
        'ottawashooting-all-rnr-threads',
        'prince-toronto-all-rnr-threads',
        'putinmissing-all-rnr-threads',
        'sydneysiege-all-rnr-threads',
    ]
    
    # Also check for alternative naming (without -all-rnr-threads suffix)
    alt_stories = [
        'charliehebdo',
        'ebola-essien',
        'ferguson',
        'germanwings-crash',
        'gurlitt',
        'ottawashooting',
        'prince-toronto',
        'putinmissing',
        'sydneysiege',
    ]
    
    for story in list(data_path.iterdir()):
        if story.is_dir():
            stories_found.append(story.name)
    
    print(f"\nStories found: {len(stories_found)}")
    for story in stories_found:
        print(f"  - {story}")
    
    # Check a sample story for required files
    sample_story = stories_found[0] if stories_found else None
    if sample_story:
        sample_path = data_path / sample_story
        print(f"\nChecking sample story: {sample_story}")
        
        # Look for threads
        threads_found = 0
        annotation_files = 0
        structure_files = 0
        
        for item in sample_path.rglob("*"):
            if item.is_dir():
                # Check if it looks like a thread directory
                if (item / "annotation.json").exists():
                    annotation_files += 1
                if (item / "structure.json").exists():
                    structure_files += 1
                if (item / "source-tweets").exists() or (item / "source-tweet").exists():
                    threads_found += 1
        
        print(f"  Thread directories found: {threads_found}")
        print(f"  annotation.json files: {annotation_files}")
        print(f"  structure.json files: {structure_files}")
        
        if annotation_files > 0 and structure_files > 0:
            print("\n✅ Dataset has the correct structure (annotated version)")
            return True
        else:
            print("\n⚠️  Dataset structure differs from expected")
            print("   This appears to be the non-annotated version of PHEME")
            print("   The link prediction model requires the annotated version with:")
            print("   - annotation.json files")
            print("   - structure.json files")
            return False
    
    return False


if __name__ == '__main__':
    import sys
    
    # Check multiple possible locations
    possible_paths = [
        "data/all-rnr-annotated-threads",
        "/Users/hong/Desktop/pheme-rnr-dataset",
        sys.argv[1] if len(sys.argv) > 1 else None,
    ]
    
    for path in possible_paths:
        if path and Path(path).exists():
            print(f"\n{'='*80}")
            print(f"Checking: {path}")
            print(f"{'='*80}")
            if check_dataset_structure(path):
                print(f"\n✅ Ready to use! You can run:")
                print(f"   uv run python train_link_prediction.py --data-root {path}")
                break
            else:
                print(f"\n❌ Dataset at {path} doesn't have the required structure")
                print("\nPlease download the annotated PHEME dataset from:")
                print("  https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/2068650")
                print("\nThe dataset should be extracted to:")
                print("  data/all-rnr-annotated-threads/")
                print("\nAnd should contain directories like:")
                print("  data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/")
                print("  data/all-rnr-annotated-threads/charliehebdo-all-rnr-threads/")
                print("  etc.")
    else:
        print("\n❌ No dataset found in expected locations")
        print("\nPlease download the annotated PHEME dataset and place it in:")
        print("  data/all-rnr-annotated-threads/")

