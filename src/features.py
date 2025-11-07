"""
Feature extraction functions for tweets and users.

This module provides functions to extract numerical features from
tweets and users for use in graph neural networks.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

from .models import Tweet, TwitterUser


def parse_twitter_timestamp(timestamp_str: str) -> datetime:
    """
    Parse Twitter's timestamp format to datetime.

    Format: "Tue Mar 24 10:51:21 +0000 2015"

    Args:
        timestamp_str: Twitter timestamp string

    Returns:
        datetime object
    """
    return datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %z %Y")


# ============================================================================
# Tweet Feature Extraction
# ============================================================================

def extract_tweet_text_features(tweet: Tweet) -> np.ndarray:
    """
    Extract text-based features from a tweet.

    Args:
        tweet: Tweet object

    Returns:
        Array of text features [6 features]
    """
    text = tweet.text or ""
    
    # Text length features
    text_length = len(text)
    word_count = len(text.split()) if text else 0
    
    # Character-level features
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / max(text_length, 1)
    
    # Punctuation count
    punctuation_chars = ".,!?;:-\"\'()[]{}"
    punctuation_count = sum(1 for c in text if c in punctuation_chars)
    punctuation_ratio = punctuation_count / max(text_length, 1)
    
    # Exclamation/question marks (engagement indicators)
    has_exclamation = 1.0 if "!" in text else 0.0
    has_question = 1.0 if "?" in text else 0.0
    
    return np.array([
        text_length,              # Character count
        word_count,               # Word count
        uppercase_ratio,          # Uppercase ratio
        punctuation_ratio,        # Punctuation ratio
        has_exclamation,          # Contains exclamation mark
        has_question,             # Contains question mark
    ], dtype=np.float32)


def extract_tweet_entity_features(tweet: Tweet) -> np.ndarray:
    """
    Extract entity-based features from a tweet.

    Args:
        tweet: Tweet object

    Returns:
        Array of entity features [4 features]
    """
    entities = tweet.entities

    return np.array([
        len(entities.hashtags),              # Number of hashtags
        len(entities.urls),                  # Number of URLs
        len(entities.user_mentions),         # Number of mentions
        len(entities.symbols),               # Number of symbols
    ], dtype=np.float32)


def extract_tweet_engagement_features(
    tweet: Tweet, 
    replies_count: int = 0
) -> np.ndarray:
    """
    Extract engagement features from a tweet.

    Args:
        tweet: Tweet object
        replies_count: Number of replies this tweet received (from thread structure)

    Returns:
        Array of engagement features [3 features]
    """
    return np.array([
        np.log1p(tweet.retweet_count),       # Log retweet count
        np.log1p(tweet.favorite_count),      # Log favorite count
        np.log1p(replies_count),             # Log replies count
    ], dtype=np.float32)


def extract_tweet_metadata_features(tweet: Tweet) -> np.ndarray:
    """
    Extract metadata features from a tweet.

    Args:
        tweet: Tweet object

    Returns:
        Array of metadata features [3 features]
    """
    return np.array([
        1.0 if tweet.retweeted else 0.0,     # Is retweet
        1.0 if tweet.truncated else 0.0,     # Is truncated
        1.0 if tweet.in_reply_to_status_id else 0.0,  # Is reply
    ], dtype=np.float32)


def extract_tweet_features(
    tweet: Tweet, 
    include_text: bool = True,
    replies_count: int = 0
) -> np.ndarray:
    """
    Extract all numerical features from a tweet.

    Args:
        tweet: Tweet object
        include_text: Whether to include text-based features
        replies_count: Number of replies this tweet received

    Returns:
        Array of tweet features [16 base features: 6 text + 4 entity + 3 engagement + 3 metadata]
        Note: Temporal features (6) and positional encoding (16) are added separately
    """
    features = []

    if include_text:
        features.append(extract_tweet_text_features(tweet))

    features.append(extract_tweet_entity_features(tweet))
    features.append(extract_tweet_engagement_features(tweet, replies_count))
    features.append(extract_tweet_metadata_features(tweet))

    return np.concatenate(features)


# ============================================================================
# User Feature Extraction
# ============================================================================

def extract_user_profile_features(user: TwitterUser) -> np.ndarray:
    """
    Extract profile features from a user.

    Args:
        user: TwitterUser object

    Returns:
        Array of profile features [7 features]
    """
    return np.array([
        np.log1p(user.followers_count),      # Log followers
        np.log1p(user.friends_count),        # Log friends
        np.log1p(user.statuses_count),       # Log statuses
        np.log1p(user.listed_count),         # Log listed count
        np.log1p(user.favourites_count),     # Log favorites
        1.0 if user.verified else 0.0,       # Is verified
        1.0 if user.protected else 0.0,      # Is protected
    ], dtype=np.float32)


def extract_user_activity_features(user: TwitterUser) -> np.ndarray:
    """
    Extract activity-based features from a user.

    Args:
        user: TwitterUser object

    Returns:
        Array of activity features [3 features]
    """
    # Calculate ratios
    followers = max(user.followers_count, 1)
    friends = max(user.friends_count, 1)

    return np.array([
        friends / followers,                  # Friends/followers ratio
        user.statuses_count / max(followers, 1),  # Tweets per follower
        user.favourites_count / max(user.statuses_count, 1),  # Favorites per tweet
    ], dtype=np.float32)


def extract_user_metadata_features(user: TwitterUser) -> np.ndarray:
    """
    Extract metadata features from a user.

    Args:
        user: TwitterUser object

    Returns:
        Array of metadata features [3 features]
    """
    return np.array([
        1.0 if user.geo_enabled else 0.0,    # Geo enabled
        1.0 if user.default_profile else 0.0,  # Default profile
        1.0 if user.default_profile_image else 0.0,  # Default image
    ], dtype=np.float32)


def extract_user_timezone_features(user: TwitterUser) -> np.ndarray:
    """
    Extract timezone features from a user.
    
    Args:
        user: TwitterUser object
        
    Returns:
        Array of timezone features [2 features]
    """
    # Timezone (encode as hash-based feature or one-hot, but use simple numeric encoding)
    # Most timezones are None, so we encode presence and use offset if available
    has_timezone = 1.0 if user.time_zone else 0.0
    
    # UTC offset (in seconds, normalized)
    # Most offsets are None, so default to 0
    if user.utc_offset is not None:
        # Normalize to [-1, 1] range (offsets are typically -43200 to 50400 seconds)
        # Divide by 43200 (12 hours in seconds) to normalize
        normalized_offset = user.utc_offset / 43200.0
        # Clamp to [-1, 1]
        normalized_offset = max(-1.0, min(1.0, normalized_offset))
    else:
        normalized_offset = 0.0
    
    return np.array([
        has_timezone,          # Has timezone set
        normalized_offset,     # Normalized UTC offset
    ], dtype=np.float32)


def extract_user_description_features(user: TwitterUser) -> np.ndarray:
    """
    Extract description text features from a user.
    
    Args:
        user: TwitterUser object
        
    Returns:
        Array of description features [4 features]
    """
    description = user.description or ""
    
    # Description length features
    desc_length = len(description)
    desc_word_count = len(description.split()) if description else 0
    
    # Has description
    has_description = 1.0 if description else 0.0
    
    # Description has URL
    has_url = 1.0 if description and ("http://" in description or "https://" in description) else 0.0
    
    return np.array([
        desc_length,           # Description character count
        desc_word_count,       # Description word count
        has_description,       # Has description
        has_url,              # Description contains URL
    ], dtype=np.float32)


def extract_user_profile_age_features(user: TwitterUser) -> np.ndarray:
    """
    Extract profile creation time features from a user.
    
    Args:
        user: TwitterUser object
        
    Returns:
        Array of profile age features [2 features]
    """
    try:
        created_at = parse_twitter_timestamp(user.created_at)
        # Reference time: 2015-01-01 (approximate dataset timeframe)
        if created_at.tzinfo:
            reference_epoch = datetime(2015, 1, 1, tzinfo=created_at.tzinfo)
        else:
            reference_epoch = datetime(2015, 1, 1)
        
        # Account age in days (how old the account was at reference time)
        # created_at is when account was created, reference_epoch is when we're measuring from
        age_delta = (reference_epoch - created_at).total_seconds() / 86400.0  # Convert to days
        # Handle negative deltas (account created after reference)
        age_delta = max(0, age_delta)
        # Log-scale for better distribution (accounts can be years old)
        account_age_log = np.log1p(age_delta)
        
        # Years since creation (for context)
        account_age_years = age_delta / 365.25
        
    except (ValueError, AttributeError, TypeError):
        # If creation time is missing or invalid, use defaults
        account_age_log = 0.0
        account_age_years = 0.0
    
    return np.array([
        account_age_log,       # Log account age in days
        account_age_years,     # Account age in years
    ], dtype=np.float32)


def extract_user_features(user: TwitterUser) -> np.ndarray:
    """
    Extract all numerical features from a user.

    Args:
        user: TwitterUser object

    Returns:
        Array of user features [21 features total: 7 profile + 3 activity + 3 metadata + 2 timezone + 4 description + 2 profile age]
    """
    return np.concatenate([
        extract_user_profile_features(user),      # 7 features
        extract_user_activity_features(user),     # 3 features
        extract_user_metadata_features(user),     # 3 features
        extract_user_timezone_features(user),     # 2 features
        extract_user_description_features(user),  # 4 features
        extract_user_profile_age_features(user),  # 2 features
    ])


# ============================================================================
# Temporal Features
# ============================================================================

def extract_temporal_features(
    tweet: Tweet,
    reference_time: Optional[datetime] = None
) -> np.ndarray:
    """
    Extract temporal features from a tweet.

    Args:
        tweet: Tweet object
        reference_time: Reference time for relative features (e.g., thread start time)

    Returns:
        Array of temporal features [6 features]
    """
    created_at = parse_twitter_timestamp(tweet.created_at)
    
    # Hour of day (0-23)
    hour = created_at.hour
    # Normalize using sin/cos encoding for cyclical nature
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (0=Monday, 6=Sunday)
    day_of_week = created_at.weekday()
    # Normalize using sin/cos encoding
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Time since reference (in seconds, log-scaled)
    if reference_time is not None:
        time_delta = (created_at - reference_time).total_seconds()
        # Handle negative deltas (shouldn't happen, but be safe)
        time_since_ref = np.log1p(max(time_delta, 0))
    else:
        time_since_ref = 0.0
    
    # Unix timestamp (normalized, log-scaled for better distribution)
    timestamp = created_at.timestamp()
    # Normalize by subtracting a reference epoch (e.g., 2015-01-01) and log-scale
    if created_at.tzinfo:
        epoch_2015 = datetime(2015, 1, 1, tzinfo=created_at.tzinfo).timestamp()
    else:
        epoch_2015 = datetime(2015, 1, 1).timestamp()
    normalized_timestamp = np.log1p(max(timestamp - epoch_2015, 0))
    
    return np.array([
        hour_sin,                  # Hour (sin encoding)
        hour_cos,                  # Hour (cos encoding)
        day_sin,                   # Day of week (sin encoding)
        day_cos,                   # Day of week (cos encoding)
        time_since_ref,            # Time since reference (log seconds)
        normalized_timestamp,      # Normalized timestamp
    ], dtype=np.float32)


def compute_replies_count(
    edges: List[tuple[str, str]]
) -> Dict[str, int]:
    """
    Compute the number of replies received by each tweet.
    
    Args:
        edges: List of (parent_id, child_id) reply edges
        
    Returns:
        Dictionary of tweet_id -> number of replies received
    """
    replies_count = {}
    for parent_id, child_id in edges:
        replies_count[parent_id] = replies_count.get(parent_id, 0) + 1
    return replies_count


def compute_temporal_positions(
    tweets: Dict[str, Tweet],
    edges: List[tuple[str, str]]
) -> Dict[str, int]:
    """
    Compute temporal positions for tweets based on reply structure.

    Args:
        tweets: Dictionary of tweet_id -> Tweet
        edges: List of (parent_id, child_id) reply edges

    Returns:
        Dictionary of tweet_id -> position (0 for root, incrementing by depth)
    """
    # Build adjacency list
    children = {}
    for parent, child in edges:
        if parent not in children:
            children[parent] = []
        children[parent].append(child)

    # Find root (tweet with no parent)
    all_children = set(child for _, child in edges)
    all_tweets = set(tweets.keys())
    roots = all_tweets - all_children

    if not roots:
        # All tweets in a cycle or single tweet
        return {tid: 0 for tid in tweets.keys()}

    # BFS to assign positions
    positions = {}
    queue = [(root, 0) for root in roots]

    while queue:
        tweet_id, depth = queue.pop(0)
        if tweet_id in positions:
            continue
        positions[tweet_id] = depth

        if tweet_id in children:
            for child_id in children[tweet_id]:
                queue.append((child_id, depth + 1))

    # Handle any disconnected tweets
    for tweet_id in tweets:
        if tweet_id not in positions:
            positions[tweet_id] = 0

    return positions


def create_temporal_positional_encoding(
    position: int,
    d_model: int = 16
) -> np.ndarray:
    """
    Create sinusoidal positional encoding for a temporal position.
    
    Uses the same approach as Transformer positional encodings:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        position: Temporal position (0, 1, 2, ...)
        d_model: Dimension of the encoding

    Returns:
        Positional encoding vector [d_model dimensions]
    """
    encoding = np.zeros(d_model, dtype=np.float32)
    
    for i in range(0, d_model, 2):
        # Calculate the divisor for this dimension
        div_term = 10000 ** (i / d_model)
        # Even indices: sin
        encoding[i] = np.sin(position / div_term)
        # Odd indices: cos (if not the last element)
        if i + 1 < d_model:
            encoding[i + 1] = np.cos(position / div_term)
    
    return encoding


# ============================================================================
# Batch Feature Extraction
# ============================================================================

def extract_all_tweet_features(
    tweets: Dict[str, Tweet],
    reference_time: Optional[datetime] = None,
    include_temporal_encoding: bool = True,
    temporal_encoding_dim: int = 16,
    edges: Optional[List[tuple[str, str]]] = None
) -> tuple[Dict[str, np.ndarray], int]:
    """
    Extract features for all tweets.

    Args:
        tweets: Dictionary of tweet_id -> Tweet
        reference_time: Reference time for temporal features
        include_temporal_encoding: Whether to include positional encoding
        temporal_encoding_dim: Dimension of temporal encoding
        edges: Reply edges for computing temporal positions and reply counts

    Returns:
        Tuple of (features_dict, total_feature_dim)
    """
    features_dict = {}

    # Compute replies count for each tweet
    replies_count_dict = {}
    if edges:
        replies_count_dict = compute_replies_count(edges)
        # Initialize all tweets with 0 replies if not in dict
        for tweet_id in tweets.keys():
            if tweet_id not in replies_count_dict:
                replies_count_dict[tweet_id] = 0

    # Compute temporal positions if needed
    positions = None
    if include_temporal_encoding and edges:
        positions = compute_temporal_positions(tweets, edges)

    for tweet_id, tweet in tweets.items():
        # Get replies count for this tweet
        replies_count = replies_count_dict.get(tweet_id, 0)
        
        # Base features (16: 6 text + 4 entity + 3 engagement + 3 metadata)
        base_features = extract_tweet_features(tweet, replies_count=replies_count)
        # Temporal features (6)
        temporal = extract_temporal_features(tweet, reference_time)

        feature_parts = [base_features, temporal]

        # Add temporal positional encoding (16)
        if include_temporal_encoding:
            if positions:
                position = positions.get(tweet_id, 0)
            else:
                position = 0
            encoding = create_temporal_positional_encoding(
                position, temporal_encoding_dim
            )
            feature_parts.append(encoding)

        features_dict[tweet_id] = np.concatenate(feature_parts)

    # Calculate total dimension
    sample_features = next(iter(features_dict.values()))
    feature_dim = len(sample_features)

    return features_dict, feature_dim


def extract_all_user_features(
    users: Dict[int, TwitterUser]
) -> tuple[Dict[int, np.ndarray], int]:
    """
    Extract features for all users.

    Args:
        users: Dictionary of user_id -> TwitterUser

    Returns:
        Tuple of (features_dict, total_feature_dim)
    """
    features_dict = {}

    for user_id, user in users.items():
        features_dict[user_id] = extract_user_features(user)

    # Calculate total dimension
    sample_features = next(iter(features_dict.values()))
    feature_dim = len(sample_features)

    return features_dict, feature_dim


# ============================================================================
# Feature Normalization
# ============================================================================

def normalize_features(
    features: np.ndarray,
    method: str = "standardize"
) -> np.ndarray:
    """
    Normalize a feature matrix.

    Args:
        features: Feature matrix [num_samples, num_features]
        method: Normalization method ("standardize", "minmax", or "none")

    Returns:
        Normalized feature matrix
    """
    if method == "none":
        return features

    if method == "standardize":
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        return (features - mean) / std

    elif method == "minmax":
        min_val = features.min(axis=0)
        max_val = features.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # Avoid division by zero
        return (features - min_val) / range_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")
