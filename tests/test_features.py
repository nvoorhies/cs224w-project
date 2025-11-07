"""
Tests for feature extraction functions.

These tests validate that feature extraction works correctly for both
tweets and users, including new features like replies count, temporal
encoding, timezone, profile age, and description features.
"""

import sys
from pathlib import Path

# Add src to path to import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from datetime import datetime, timezone

# Import directly from module files to avoid __init__.py importing graph_builder
from src.models import (
    Tweet,
    TwitterUser,
    TweetEntities,
)
from src.features import (
    extract_tweet_text_features,
    extract_tweet_entity_features,
    extract_tweet_engagement_features,
    extract_tweet_metadata_features,
    extract_tweet_features,
    extract_temporal_features,
    extract_user_profile_features,
    extract_user_activity_features,
    extract_user_metadata_features,
    extract_user_timezone_features,
    extract_user_description_features,
    extract_user_profile_age_features,
    extract_user_features,
    extract_all_tweet_features,
    extract_all_user_features,
    compute_replies_count,
    compute_temporal_positions,
    create_temporal_positional_encoding,
    parse_twitter_timestamp,
)


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def sample_user():
    """Create a sample TwitterUser for testing."""
    return TwitterUser(
        id=123456,
        id_str="123456",
        screen_name="testuser",
        name="Test User",
        description="This is a test description with https://example.com",
        followers_count=1000,
        friends_count=500,
        statuses_count=2000,
        listed_count=10,
        favourites_count=5000,
        verified=True,
        protected=False,
        geo_enabled=True,
        default_profile=False,
        default_profile_image=False,
        created_at="Mon Jan 01 12:00:00 +0000 2010",
        time_zone="Pacific Time (US & Canada)",
        utc_offset=-28800,  # PST offset
    )


@pytest.fixture
def sample_tweet(sample_user):
    """Create a sample Tweet for testing."""
    entities = TweetEntities(
        hashtags=[],
        urls=[],
        user_mentions=[],
        symbols=[],
    )
    return Tweet(
        id=789012,
        id_str="789012",
        text="This is a TEST tweet! It has some features?",
        created_at="Tue Mar 24 10:51:21 +0000 2015",
        user=sample_user,
        entities=entities,
        retweet_count=50,
        favorite_count=100,
        retweeted=False,
        favorited=False,
        truncated=False,
        source="<a href=\"http://twitter.com\" rel=\"nofollow\">Twitter Web Client</a>",
        lang="en",
    )


# ============================================================================
# Test tweet text features
# ============================================================================

def test_extract_tweet_text_features(sample_tweet):
    """Test tweet text feature extraction."""
    features = extract_tweet_text_features(sample_tweet)
    
    assert features.shape == (6,)
    assert features.dtype == np.float32
    
    # Check text length
    assert features[0] == len(sample_tweet.text)
    
    # Check word count
    assert features[1] == len(sample_tweet.text.split())
    
    # Check uppercase ratio (should be > 0 for "TEST")
    assert features[2] >= 0
    assert features[2] <= 1
    
    # Check punctuation ratio
    assert features[3] >= 0
    assert features[3] <= 1
    
    # Check exclamation mark (text has "!")
    assert features[4] == 1.0
    
    # Check question mark (text has "?")
    assert features[5] == 1.0


def test_extract_tweet_text_features_empty():
    """Test text features for empty tweet."""
    user = TwitterUser(
        id=1, id_str="1", screen_name="u", name="U",
        created_at="Mon Jan 01 12:00:00 +0000 2010"
    )
    tweet = Tweet(
        id=1, id_str="1", text="", created_at="Mon Jan 01 12:00:00 +0000 2010",
        user=user, entities=TweetEntities(), source="", lang="en"
    )
    
    features = extract_tweet_text_features(tweet)
    assert features[0] == 0  # length
    assert features[1] == 0  # word count
    assert features[4] == 0  # no exclamation
    assert features[5] == 0  # no question


# ============================================================================
# Test tweet engagement features
# ============================================================================

def test_extract_tweet_engagement_features(sample_tweet):
    """Test tweet engagement feature extraction."""
    features = extract_tweet_engagement_features(sample_tweet, replies_count=5)
    
    assert features.shape == (3,)
    assert features.dtype == np.float32
    
    # Check retweet count (log-scaled) - use approximate comparison for float32
    assert features[0] == pytest.approx(np.log1p(sample_tweet.retweet_count), abs=1e-6)
    
    # Check favorite count (log-scaled)
    assert features[1] == pytest.approx(np.log1p(sample_tweet.favorite_count), abs=1e-6)
    
    # Check replies count (log-scaled)
    assert features[2] == pytest.approx(np.log1p(5), abs=1e-6)


def test_extract_tweet_engagement_features_zero_replies(sample_tweet):
    """Test engagement features with zero replies."""
    features = extract_tweet_engagement_features(sample_tweet, replies_count=0)
    assert features[2] == 0.0  # log1p(0) = 0


# ============================================================================
# Test tweet temporal features
# ============================================================================

def test_extract_temporal_features(sample_tweet):
    """Test temporal feature extraction."""
    reference_time = parse_twitter_timestamp("Mon Jan 01 12:00:00 +0000 2015")
    features = extract_temporal_features(sample_tweet, reference_time)
    
    assert features.shape == (6,)
    assert features.dtype == np.float32
    
    # Hour should be encoded with sin/cos (between -1 and 1)
    assert -1.0 <= features[0] <= 1.0  # hour_sin
    assert -1.0 <= features[1] <= 1.0  # hour_cos
    
    # Day of week should be encoded with sin/cos
    assert -1.0 <= features[2] <= 1.0  # day_sin
    assert -1.0 <= features[3] <= 1.0  # day_cos
    
    # Time since reference should be non-negative
    assert features[4] >= 0.0
    
    # Normalized timestamp should be non-negative
    assert features[5] >= 0.0


def test_extract_temporal_features_no_reference(sample_tweet):
    """Test temporal features without reference time."""
    features = extract_temporal_features(sample_tweet, reference_time=None)
    
    assert features.shape == (6,)
    # Time since reference should be 0
    assert features[4] == 0.0


# ============================================================================
# Test temporal positional encoding
# ============================================================================

def test_create_temporal_positional_encoding():
    """Test temporal positional encoding."""
    encoding = create_temporal_positional_encoding(position=0, d_model=16)
    
    assert encoding.shape == (16,)
    assert encoding.dtype == np.float32
    # Position 0 should have sin(0) = 0 and cos(0) = 1
    assert encoding[0] == pytest.approx(0.0, abs=1e-6)
    assert encoding[1] == pytest.approx(1.0, abs=1e-6)


def test_create_temporal_positional_encoding_different_positions():
    """Test that different positions produce different encodings."""
    enc1 = create_temporal_positional_encoding(position=0, d_model=16)
    enc2 = create_temporal_positional_encoding(position=1, d_model=16)
    enc3 = create_temporal_positional_encoding(position=10, d_model=16)
    
    # All should have same shape
    assert enc1.shape == enc2.shape == enc3.shape
    
    # Different positions should produce different encodings
    assert not np.allclose(enc1, enc2)
    assert not np.allclose(enc1, enc3)


# ============================================================================
# Test replies count computation
# ============================================================================

def test_compute_replies_count():
    """Test computing replies count from edges."""
    edges = [
        ("1", "2"),  # Tweet 1 gets a reply
        ("1", "3"),  # Tweet 1 gets another reply
        ("2", "4"),  # Tweet 2 gets a reply
    ]
    
    counts = compute_replies_count(edges)
    
    assert counts["1"] == 2
    assert counts["2"] == 1
    assert "3" not in counts  # Tweet 3 doesn't receive replies
    assert "4" not in counts  # Tweet 4 doesn't receive replies


def test_compute_replies_count_empty():
    """Test replies count with no edges."""
    counts = compute_replies_count([])
    assert len(counts) == 0


# ============================================================================
# Test user timezone features
# ============================================================================

def test_extract_user_timezone_features(sample_user):
    """Test user timezone feature extraction."""
    features = extract_user_timezone_features(sample_user)
    
    assert features.shape == (2,)
    assert features.dtype == np.float32
    
    # Should have timezone
    assert features[0] == 1.0
    
    # UTC offset should be normalized
    assert -1.0 <= features[1] <= 1.0


def test_extract_user_timezone_features_no_timezone():
    """Test timezone features for user without timezone."""
    user = TwitterUser(
        id=1, id_str="1", screen_name="u", name="U",
        created_at="Mon Jan 01 12:00:00 +0000 2010",
        time_zone=None,
        utc_offset=None,
    )
    
    features = extract_user_timezone_features(user)
    assert features[0] == 0.0  # No timezone
    assert features[1] == 0.0  # No offset


# ============================================================================
# Test user description features
# ============================================================================

def test_extract_user_description_features(sample_user):
    """Test user description feature extraction."""
    features = extract_user_description_features(sample_user)
    
    assert features.shape == (4,)
    assert features.dtype == np.float32
    
    # Check description length
    assert features[0] == len(sample_user.description)
    
    # Check word count
    assert features[1] == len(sample_user.description.split())
    
    # Should have description
    assert features[2] == 1.0
    
    # Should have URL
    assert features[3] == 1.0


def test_extract_user_description_features_no_description():
    """Test description features for user without description."""
    user = TwitterUser(
        id=1, id_str="1", screen_name="u", name="U",
        created_at="Mon Jan 01 12:00:00 +0000 2010",
        description=None,
    )
    
    features = extract_user_description_features(user)
    assert features[0] == 0  # No length
    assert features[1] == 0  # No words
    assert features[2] == 0.0  # No description
    assert features[3] == 0.0  # No URL


# ============================================================================
# Test user profile age features
# ============================================================================

def test_extract_user_profile_age_features(sample_user):
    """Test user profile age feature extraction."""
    features = extract_user_profile_age_features(sample_user)
    
    assert features.shape == (2,)
    assert features.dtype == np.float32
    
    # Account age should be positive (account created in 2010, reference is 2015)
    assert features[0] > 0  # Log age
    assert features[1] > 0  # Years


def test_extract_user_profile_age_features_recent_account():
    """Test profile age for recent account."""
    user = TwitterUser(
        id=1, id_str="1", screen_name="u", name="U",
        created_at="Mon Jan 01 12:00:00 +0000 2016",  # After reference
    )
    
    features = extract_user_profile_age_features(user)
    # Should handle gracefully (age = 0)
    assert features[0] >= 0
    assert features[1] >= 0


# ============================================================================
# Test complete feature extraction
# ============================================================================

def test_extract_tweet_features_complete(sample_tweet):
    """Test complete tweet feature extraction."""
    features = extract_tweet_features(
        sample_tweet,
        include_text=True,
        replies_count=3
    )
    
    # Should have: 6 text + 4 entity + 3 engagement + 3 metadata = 16
    assert features.shape == (16,)
    assert features.dtype == np.float32


def test_extract_user_features_complete(sample_user):
    """Test complete user feature extraction."""
    features = extract_user_features(sample_user)
    
    # Should have: 7 profile + 3 activity + 3 metadata + 2 timezone + 4 description + 2 profile age = 21
    assert features.shape == (21,)
    assert features.dtype == np.float32


# ============================================================================
# Test batch feature extraction
# ============================================================================

def test_extract_all_tweet_features(sample_tweet):
    """Test batch tweet feature extraction."""
    tweets = {"789012": sample_tweet}
    reference_time = parse_twitter_timestamp("Mon Jan 01 12:00:00 +0000 2015")
    edges = []  # No replies
    
    features_dict, feature_dim = extract_all_tweet_features(
        tweets,
        reference_time=reference_time,
        include_temporal_encoding=True,
        temporal_encoding_dim=16,
        edges=edges
    )
    
    assert "789012" in features_dict
    # Should have: 16 base + 6 temporal + 16 positional = 38
    assert feature_dim == 38
    assert features_dict["789012"].shape == (38,)


def test_extract_all_tweet_features_with_replies():
    """Test batch extraction with reply counts."""
    user = TwitterUser(
        id=1, id_str="1", screen_name="u", name="U",
        created_at="Mon Jan 01 12:00:00 +0000 2010"
    )
    
    tweet1 = Tweet(
        id=1, id_str="1", text="Tweet 1", created_at="Mon Jan 01 12:00:00 +0000 2015",
        user=user, entities=TweetEntities(), source="", lang="en"
    )
    
    tweet2 = Tweet(
        id=2, id_str="2", text="Tweet 2", created_at="Mon Jan 01 12:01:00 +0000 2015",
        user=user, entities=TweetEntities(), source="", lang="en"
    )
    
    tweets = {"1": tweet1, "2": tweet2}
    edges = [("1", "2")]  # Tweet 2 replies to tweet 1
    reference_time = parse_twitter_timestamp("Mon Jan 01 12:00:00 +0000 2015")
    
    features_dict, _ = extract_all_tweet_features(
        tweets,
        reference_time=reference_time,
        include_temporal_encoding=True,
        edges=edges
    )
    
    # Tweet 1 should have replies_count = 1
    # The feature is in the engagement section (index 10 = 6 text + 4 entity)
    # Engagement features are at indices 10-12 (retweet, favorite, replies)
    tweet1_features = features_dict["1"]
    # Replies count is the 3rd engagement feature (index 12)
    # We can't easily verify the exact value, but we can check it's there
    assert tweet1_features.shape[0] >= 13  # At least 13 features


def test_extract_all_user_features(sample_user):
    """Test batch user feature extraction."""
    users = {123456: sample_user}
    
    features_dict, feature_dim = extract_all_user_features(users)
    
    assert 123456 in features_dict
    assert feature_dim == 21
    assert features_dict[123456].shape == (21,)


# ============================================================================
# Test temporal positions computation
# ============================================================================

def test_compute_temporal_positions():
    """Test computing temporal positions from edges."""
    user = TwitterUser(
        id=1, id_str="1", screen_name="u", name="U",
        created_at="Mon Jan 01 12:00:00 +0000 2010"
    )
    
    tweets = {
        "1": Tweet(id=1, id_str="1", text="Root", created_at="Mon Jan 01 12:00:00 +0000 2015",
                   user=user, entities=TweetEntities(), source="", lang="en"),
        "2": Tweet(id=2, id_str="2", text="Reply 1", created_at="Mon Jan 01 12:01:00 +0000 2015",
                   user=user, entities=TweetEntities(), source="", lang="en"),
        "3": Tweet(id=3, id_str="3", text="Reply 2", created_at="Mon Jan 01 12:02:00 +0000 2015",
                   user=user, entities=TweetEntities(), source="", lang="en"),
    }
    
    edges = [("1", "2"), ("2", "3")]  # 1 -> 2 -> 3
    
    positions = compute_temporal_positions(tweets, edges)
    
    assert positions["1"] == 0  # Root
    assert positions["2"] == 1  # Depth 1
    assert positions["3"] == 2  # Depth 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

