"""
Tests for PHEME dataset parsers.

These tests validate that the parsers can correctly load and validate
JSON files from the PHEME dataset.
"""

import json
from pathlib import Path
import pytest

from src.models import (
    Annotation,
    ThreadStructure,
    Tweet,
    TweetThread,
)
from src.parsers import (
    load_annotation,
    load_structure,
    load_tweet,
    load_tweet_thread,
    find_all_threads,
    verify_thread,
    ParseError,
)


# ============================================================================
# Test data paths
# ============================================================================

DATA_ROOT = Path(__file__).parent.parent / "data" / "all-rnr-annotated-threads"
SAMPLE_THREAD_1 = (
    DATA_ROOT
    / "germanwings-crash-all-rnr-threads"
    / "rumours"
    / "580320995266936832"
)
SAMPLE_THREAD_2 = (
    DATA_ROOT
    / "germanwings-crash-all-rnr-threads"
    / "rumours"
    / "580697361799876608"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_annotation_path():
    """Path to a sample annotation.json file."""
    return SAMPLE_THREAD_1 / "annotation.json"


@pytest.fixture
def sample_structure_path():
    """Path to a sample structure.json file."""
    return SAMPLE_THREAD_1 / "structure.json"


@pytest.fixture
def sample_tweet_path():
    """Path to a sample tweet JSON file."""
    return SAMPLE_THREAD_1 / "source-tweets" / "580320995266936832.json"


@pytest.fixture
def sample_thread_dir():
    """Path to a complete sample thread directory."""
    return SAMPLE_THREAD_1


@pytest.fixture
def sample_thread_with_reactions_dir():
    """Path to a sample thread with reaction tweets."""
    return SAMPLE_THREAD_2


# ============================================================================
# Test individual parsers
# ============================================================================

def test_load_annotation(sample_annotation_path):
    """Test loading an annotation.json file."""
    annotation = load_annotation(sample_annotation_path)

    assert isinstance(annotation, Annotation)
    assert annotation.is_rumour in ["rumour", "non-rumour", "unverified"]
    assert isinstance(annotation.category, str)
    assert isinstance(annotation.links, list)


def test_load_annotation_nonexistent():
    """Test loading a nonexistent annotation file."""
    with pytest.raises(ParseError, match="not found"):
        load_annotation(Path("/nonexistent/annotation.json"))


def test_load_structure(sample_structure_path):
    """Test loading a structure.json file."""
    structure = load_structure(sample_structure_path)

    assert isinstance(structure, ThreadStructure)
    assert structure.get_root_tweet_id() is not None

    # Check that we can extract tweet IDs
    tweet_ids = structure.get_all_tweet_ids()
    assert len(tweet_ids) >= 1
    assert all(isinstance(tid, str) for tid in tweet_ids)


def test_load_structure_edges(sample_thread_with_reactions_dir):
    """Test extracting edges from structure."""
    structure_path = sample_thread_with_reactions_dir / "structure.json"
    structure = load_structure(structure_path)

    edges = structure.get_reply_edges()
    assert isinstance(edges, list)
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges)

    # All edges should have string tweet IDs
    for parent, child in edges:
        assert isinstance(parent, str)
        assert isinstance(child, str)


def test_load_tweet(sample_tweet_path):
    """Test loading a tweet JSON file."""
    tweet = load_tweet(sample_tweet_path)

    assert isinstance(tweet, Tweet)
    assert tweet.id > 0
    assert tweet.id_str
    assert tweet.text
    assert tweet.user is not None
    assert tweet.user.screen_name
    assert tweet.entities is not None


def test_load_tweet_fields(sample_tweet_path):
    """Test that all expected tweet fields are loaded."""
    tweet = load_tweet(sample_tweet_path)

    # Basic fields
    assert hasattr(tweet, 'id')
    assert hasattr(tweet, 'text')
    assert hasattr(tweet, 'created_at')

    # User fields
    assert hasattr(tweet.user, 'id')
    assert hasattr(tweet.user, 'screen_name')
    assert hasattr(tweet.user, 'followers_count')

    # Engagement metrics
    assert hasattr(tweet, 'retweet_count')
    assert hasattr(tweet, 'favorite_count')

    # Entities
    assert hasattr(tweet.entities, 'hashtags')
    assert hasattr(tweet.entities, 'urls')
    assert hasattr(tweet.entities, 'user_mentions')


# ============================================================================
# Test thread-level parser
# ============================================================================

def test_load_tweet_thread_basic(sample_thread_dir):
    """Test loading a basic thread without reactions."""
    thread = load_tweet_thread(sample_thread_dir, load_reactions=False)

    assert isinstance(thread, TweetThread)
    assert thread.thread_id == "580320995266936832"
    assert thread.annotation is not None
    assert thread.structure is not None
    assert thread.source_tweet is not None


def test_load_tweet_thread_with_reactions(sample_thread_with_reactions_dir):
    """Test loading a thread with reaction tweets."""
    thread = load_tweet_thread(sample_thread_with_reactions_dir, load_reactions=True)

    assert isinstance(thread, TweetThread)
    assert len(thread.reaction_tweets) > 0

    # All reaction tweets should be Tweet objects
    for tweet_id, tweet in thread.reaction_tweets.items():
        assert isinstance(tweet, Tweet)
        assert tweet.id_str == tweet_id


def test_tweet_thread_get_all_tweets(sample_thread_with_reactions_dir):
    """Test getting all tweets from a thread."""
    thread = load_tweet_thread(sample_thread_with_reactions_dir)

    all_tweets = thread.get_all_tweets()
    assert len(all_tweets) >= 1
    assert thread.thread_id in all_tweets
    assert all(isinstance(tweet, Tweet) for tweet in all_tweets.values())


def test_tweet_thread_get_all_users(sample_thread_with_reactions_dir):
    """Test getting all unique users from a thread."""
    thread = load_tweet_thread(sample_thread_with_reactions_dir)

    all_users = thread.get_all_users()
    assert len(all_users) >= 1
    assert all(isinstance(user_id, int) for user_id in all_users.keys())


# ============================================================================
# Test dataset-level functions
# ============================================================================

def test_find_all_threads():
    """Test finding all thread directories in the dataset."""
    thread_dirs = find_all_threads(DATA_ROOT)

    assert len(thread_dirs) > 0
    assert all(isinstance(path, Path) for path in thread_dirs)
    assert all((path / "annotation.json").exists() for path in thread_dirs)


def test_find_all_threads_germanwings():
    """Test finding threads in a specific story."""
    story_dir = DATA_ROOT / "germanwings-crash-all-rnr-threads"
    thread_dirs = find_all_threads(story_dir)

    assert len(thread_dirs) > 0
    # All threads should be from germanwings story
    assert all("germanwings" in str(path) for path in thread_dirs)


# ============================================================================
# Test verification
# ============================================================================

def test_verify_thread(sample_thread_with_reactions_dir):
    """Test thread verification and statistics."""
    thread = load_tweet_thread(sample_thread_with_reactions_dir)
    results = verify_thread(thread)

    assert "thread_id" in results
    assert "valid" in results
    assert "stats" in results
    assert "errors" in results
    assert "warnings" in results

    # Check statistics
    stats = results["stats"]
    assert stats["num_tweets"] >= 1
    assert stats["num_users"] >= 1
    assert stats["rumour_type"] in ["rumour", "non-rumour", "unverified"]


def test_verify_thread_stats_consistency(sample_thread_with_reactions_dir):
    """Test that verification statistics are consistent."""
    thread = load_tweet_thread(sample_thread_with_reactions_dir)
    results = verify_thread(thread)

    stats = results["stats"]

    # Number of tweets should equal source + reactions
    expected_tweets = 1 + len(thread.reaction_tweets)
    assert stats["num_tweets"] == expected_tweets
    assert stats["num_reaction_tweets"] == len(thread.reaction_tweets)


# ============================================================================
# Test edge cases and error handling
# ============================================================================

def test_load_annotation_invalid_json(tmp_path):
    """Test loading an invalid JSON file."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{ invalid json ")

    with pytest.raises(ParseError, match="Invalid JSON"):
        load_annotation(invalid_file)


def test_thread_structure_nested():
    """Test ThreadStructure with nested replies."""
    # Create a structure with multiple levels
    structure_data = {
        "1": {
            "2": {
                "3": {},
                "4": {}
            },
            "5": {}
        }
    }

    structure = ThreadStructure.from_dict(structure_data)

    # Should find all 5 tweets
    tweet_ids = structure.get_all_tweet_ids()
    assert len(tweet_ids) == 5
    assert set(tweet_ids) == {"1", "2", "3", "4", "5"}

    # Should find 4 edges: 1->2, 2->3, 2->4, 1->5
    edges = structure.get_reply_edges()
    assert len(edges) == 4
    assert ("1", "2") in edges
    assert ("2", "3") in edges
    assert ("2", "4") in edges
    assert ("1", "5") in edges


def test_thread_structure_empty():
    """Test ThreadStructure with no replies."""
    structure_data = {"1": {}}
    structure = ThreadStructure.from_dict(structure_data)

    tweet_ids = structure.get_all_tweet_ids()
    assert tweet_ids == ["1"]

    edges = structure.get_reply_edges()
    assert len(edges) == 0


# ============================================================================
# Integration tests
# ============================================================================

def test_end_to_end_single_thread(sample_thread_with_reactions_dir):
    """Test complete workflow for loading and verifying a thread."""
    # Load the thread
    thread = load_tweet_thread(sample_thread_with_reactions_dir)

    # Verify it
    results = verify_thread(thread)

    # Check that it's valid
    assert results["valid"]

    # Get all data
    all_tweets = thread.get_all_tweets()
    all_users = thread.get_all_users()
    edges = thread.structure.get_reply_edges()

    # Basic consistency checks
    assert len(all_tweets) > 0
    assert len(all_users) > 0

    # All edge endpoints should refer to tweets that exist in structure
    structure_ids = set(thread.structure.get_all_tweet_ids())
    for parent, child in edges:
        assert parent in structure_ids
        assert child in structure_ids


def test_load_multiple_threads():
    """Test loading multiple threads from the dataset."""
    story_dir = DATA_ROOT / "germanwings-crash-all-rnr-threads"
    thread_dirs = find_all_threads(story_dir)

    # Load first 3 threads
    loaded_count = 0
    for thread_dir in thread_dirs[:3]:
        try:
            thread = load_tweet_thread(thread_dir)
            assert isinstance(thread, TweetThread)
            loaded_count += 1
        except ParseError:
            # Some threads might have issues, that's okay
            pass

    assert loaded_count >= 2, "Should be able to load at least 2 threads"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
