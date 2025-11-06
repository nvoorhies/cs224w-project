"""
Parsers for loading PHEME dataset JSON files into Pydantic models.

These parsers provide robust loading with error handling and validation
using the Pydantic models defined in models.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from .models import (
    Annotation,
    AnnotationLink,
    ThreadStructure,
    Tweet,
    TweetThread,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Custom exception for parsing errors."""
    pass


# ============================================================================
# Individual file parsers
# ============================================================================

def load_annotation(annotation_path: Path) -> Annotation:
    """
    Load and parse an annotation.json file.

    Args:
        annotation_path: Path to annotation.json

    Returns:
        Parsed Annotation object

    Raises:
        ParseError: If file cannot be loaded or validated
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return Annotation(**data)

    except FileNotFoundError:
        raise ParseError(f"Annotation file not found: {annotation_path}")
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON in {annotation_path}: {e}")
    except ValidationError as e:
        logger.error(f"Validation error in {annotation_path}: {e}")
        raise ParseError(f"Invalid annotation format: {e}")


def load_structure(structure_path: Path) -> ThreadStructure:
    """
    Load and parse a structure.json file.

    Args:
        structure_path: Path to structure.json

    Returns:
        Parsed ThreadStructure object

    Raises:
        ParseError: If file cannot be loaded or validated
    """
    try:
        with open(structure_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return ThreadStructure.from_dict(data)

    except FileNotFoundError:
        raise ParseError(f"Structure file not found: {structure_path}")
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON in {structure_path}: {e}")
    except ValidationError as e:
        logger.error(f"Validation error in {structure_path}: {e}")
        raise ParseError(f"Invalid structure format: {e}")


def load_tweet(tweet_path: Path) -> Tweet:
    """
    Load and parse a tweet JSON file.

    Args:
        tweet_path: Path to tweet JSON file

    Returns:
        Parsed Tweet object

    Raises:
        ParseError: If file cannot be loaded or validated
    """
    try:
        with open(tweet_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return Tweet(**data)

    except FileNotFoundError:
        raise ParseError(f"Tweet file not found: {tweet_path}")
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON in {tweet_path}: {e}")
    except ValidationError as e:
        logger.error(f"Validation error in {tweet_path}: {e}")
        raise ParseError(f"Invalid tweet format: {e}")


# ============================================================================
# Thread-level parser
# ============================================================================

def load_tweet_thread(thread_dir: Path, load_reactions: bool = True) -> TweetThread:
    """
    Load a complete tweet thread from a directory.

    Expected directory structure:
        thread_dir/
        ├── annotation.json
        ├── structure.json
        ├── source-tweets/
        │   └── <tweet_id>.json
        └── reactions/
            ├── <tweet_id_1>.json
            ├── <tweet_id_2>.json
            └── ...

    Args:
        thread_dir: Path to the thread directory
        load_reactions: Whether to load reaction tweets (default: True)

    Returns:
        Parsed TweetThread object

    Raises:
        ParseError: If required files cannot be loaded
    """
    thread_dir = Path(thread_dir)
    thread_id = thread_dir.name

    logger.info(f"Loading thread: {thread_id}")

    # Load annotation
    annotation_path = thread_dir / "annotation.json"
    annotation = load_annotation(annotation_path)

    # Load structure
    structure_path = thread_dir / "structure.json"
    structure = load_structure(structure_path)

    # Load source tweet
    source_tweets_dir = thread_dir / "source-tweets"
    source_tweet_path = source_tweets_dir / f"{thread_id}.json"

    if not source_tweet_path.exists():
        # Try to find any JSON file in source-tweets
        source_tweet_files = list(source_tweets_dir.glob("*.json"))
        if not source_tweet_files:
            raise ParseError(f"No source tweet found in {source_tweets_dir}")
        source_tweet_path = source_tweet_files[0]
        logger.warning(f"Using {source_tweet_path.name} as source tweet")

    source_tweet = load_tweet(source_tweet_path)

    # Load reaction tweets
    reaction_tweets = {}
    if load_reactions:
        reactions_dir = thread_dir / "reactions"
        if reactions_dir.exists():
            reaction_files = list(reactions_dir.glob("*.json"))
            logger.info(f"Loading {len(reaction_files)} reaction tweets")

            for reaction_file in reaction_files:
                try:
                    tweet = load_tweet(reaction_file)
                    reaction_tweets[tweet.id_str] = tweet
                except ParseError as e:
                    logger.warning(f"Skipping reaction {reaction_file.name}: {e}")
                    continue

    # Create TweetThread object
    thread = TweetThread(
        thread_id=thread_id,
        annotation=annotation,
        structure=structure,
        source_tweet=source_tweet,
        reaction_tweets=reaction_tweets,
    )

    logger.info(
        f"Successfully loaded thread {thread_id}: "
        f"{len(reaction_tweets)} reactions, "
        f"{len(structure.get_all_tweet_ids())} total tweets in structure"
    )

    return thread


# ============================================================================
# Dataset-level parsers
# ============================================================================

def find_all_threads(dataset_root: Path) -> List[Path]:
    """
    Find all thread directories in the PHEME dataset.

    Args:
        dataset_root: Root directory of the dataset

    Returns:
        List of paths to thread directories
    """
    dataset_root = Path(dataset_root)
    thread_dirs = []

    # Look for directories containing annotation.json
    for annotation_file in dataset_root.rglob("annotation.json"):
        thread_dir = annotation_file.parent
        thread_dirs.append(thread_dir)

    logger.info(f"Found {len(thread_dirs)} threads in {dataset_root}")
    return thread_dirs


def load_dataset(
    dataset_root: Path,
    load_reactions: bool = True,
    max_threads: Optional[int] = None,
    skip_errors: bool = True,
) -> List[TweetThread]:
    """
    Load all threads from the PHEME dataset.

    Args:
        dataset_root: Root directory of the dataset
        load_reactions: Whether to load reaction tweets
        max_threads: Maximum number of threads to load (None = all)
        skip_errors: If True, skip threads that fail to load; if False, raise error

    Returns:
        List of TweetThread objects

    Raises:
        ParseError: If skip_errors=False and any thread fails to load
    """
    thread_dirs = find_all_threads(dataset_root)

    if max_threads is not None:
        thread_dirs = thread_dirs[:max_threads]
        logger.info(f"Limiting to {max_threads} threads")

    threads = []
    failed_count = 0

    for thread_dir in thread_dirs:
        try:
            thread = load_tweet_thread(thread_dir, load_reactions=load_reactions)
            threads.append(thread)
        except ParseError as e:
            failed_count += 1
            if skip_errors:
                logger.warning(f"Skipping thread {thread_dir.name}: {e}")
                continue
            else:
                raise

    logger.info(
        f"Successfully loaded {len(threads)} threads "
        f"({failed_count} failed)"
    )

    return threads


# ============================================================================
# Verification and statistics
# ============================================================================

def verify_thread(thread: TweetThread) -> Dict[str, any]:
    """
    Verify the integrity of a loaded thread and return statistics.

    Args:
        thread: TweetThread to verify

    Returns:
        Dictionary with verification results and statistics
    """
    results = {
        "thread_id": thread.thread_id,
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # Check that all tweets mentioned in structure exist
    structure_tweet_ids = set(thread.structure.get_all_tweet_ids())
    loaded_tweet_ids = set(thread.get_all_tweets().keys())

    missing_tweets = structure_tweet_ids - loaded_tweet_ids
    extra_tweets = loaded_tweet_ids - structure_tweet_ids

    if missing_tweets:
        results["warnings"].append(
            f"Tweets in structure but not loaded: {missing_tweets}"
        )

    if extra_tweets:
        results["warnings"].append(
            f"Loaded tweets not in structure: {extra_tweets}"
        )

    # Collect statistics
    all_tweets = thread.get_all_tweets()
    all_users = thread.get_all_users()

    results["stats"] = {
        "num_tweets": len(all_tweets),
        "num_reaction_tweets": len(thread.reaction_tweets),
        "num_users": len(all_users),
        "num_edges": len(thread.structure.get_reply_edges()),
        "rumour_type": thread.annotation.is_rumour,
        "category": thread.annotation.category,
        "num_links": len(thread.annotation.links),
    }

    # Check for orphaned nodes
    edges = thread.structure.get_reply_edges()
    if edges:
        connected_nodes = set()
        for parent, child in edges:
            connected_nodes.add(parent)
            connected_nodes.add(child)

        orphaned = structure_tweet_ids - connected_nodes
        if orphaned and len(structure_tweet_ids) > 1:
            results["warnings"].append(f"Orphaned tweet nodes: {orphaned}")

    return results
