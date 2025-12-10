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
        Array of text features [1 dummy feature]
    """
    text = tweet.text

    #TODO: Implement actual text feature extraction (e.g., sentiment, length, embeddings) if useful

    return np.zeros(1, dtype=np.float32)


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


def extract_tweet_engagement_features(tweet: Tweet) -> np.ndarray:
    """
    Extract engagement features from a tweet.

    Args:
        tweet: Tweet object

    Returns:
        Array of engagement features [2 features]
    """
    return np.array([
        np.log1p(tweet.retweet_count),       # Log retweet count
        np.log1p(tweet.favorite_count),      # Log favorite count
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


def extract_tweet_features(tweet: Tweet, include_text: bool = True) -> np.ndarray:
    """
    Extract all numerical features from a tweet.

    Args:
        tweet: Tweet object
        include_text: Whether to include text-based features

    Returns:
        Array of tweet features [15 features total]
    """
    features = []

    if include_text:
        features.append(extract_tweet_text_features(tweet))

    features.append(extract_tweet_entity_features(tweet))
    features.append(extract_tweet_engagement_features(tweet))
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


def extract_user_features(user: TwitterUser) -> np.ndarray:
    """
    Extract all numerical features from a user.

    Args:
        user: TwitterUser object

    Returns:
        Array of user features [13 features total]
    """
    return np.concatenate([
        extract_user_profile_features(user),
        extract_user_activity_features(user),
        extract_user_metadata_features(user),
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
        Array of temporal features [hour_of_day, day_of_week, unix_minutes, delta_minutes]
    """
    created_at = parse_twitter_timestamp(tweet.created_at)

    # Temporal context
    hour_of_day = created_at.hour / 24.0  # Normalize to [0, 1]
    day_of_week = created_at.weekday() / 7.0  # Normalize to [0, 1]
    unix_minutes = created_at.timestamp() / 60_000_000.0  # Scale to keep float32 stable

    if reference_time:
        delta_minutes = (created_at - reference_time).total_seconds() / 60.0
    else:
        delta_minutes = 0.0

    return np.array(
        [hour_of_day, day_of_week, unix_minutes, delta_minutes],
        dtype=np.float32
    )


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
    Create positional encoding for a temporal position.

    Args:
        position: Temporal position (0, 1, 2, ...)
        d_model: Dimension of the encoding

    Returns:
        Positional encoding vector [d_model dimensions]
    """
    # Sinusoidal positional encoding
    encoding = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = 10000 ** (i / d_model)
        encoding[i] = np.sin(position / div_term)
        if i + 1 < d_model:
            encoding[i + 1] = np.cos(position / div_term)    

    return encoding.astype(np.float32)


# ============================================================================
# Batch Feature Extraction
# ============================================================================

def extract_all_tweet_features(
    tweets: Dict[str, Tweet],
    reference_time: Optional[datetime] = None,
    include_temporal_encoding: bool = False,
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
        edges: Reply edges for computing temporal positions

    Returns:
        Tuple of (features_dict, total_feature_dim)
    """
    features_dict = {}

    # Compute temporal positions if needed
    positions = None
    # Use reply structure for positions
    if include_temporal_encoding and edges:
        positions = compute_temporal_positions(tweets, edges)
        # set position to 0 for tweets that're created after the median time
        timestamps = []
        for tweet_id, tweet in tweets.items():
            created_at = parse_twitter_timestamp(tweet.created_at)
            timestamps.append(created_at)
        median_time = sorted(timestamps)[len(timestamps) // 2]
        for tweet_id, tweet in tweets.items():
            created_at = parse_twitter_timestamp(tweet.created_at)
            if created_at > median_time:
                positions[tweet_id] = 0
    elif include_temporal_encoding:
        # Fallback: assign position 0 when no edges are present
        positions = {tweet_id: 0 for tweet_id in tweets.keys()}
    
    # Use timestamp ranking for positions
    # if include_temporal_encoding:
    #     # Collect timestamps for all tweets
    #     timestamps = []  # List[Tup[tweet_id,created_at]]
    #     for tweet_id, tweet in tweets.items():
    #         created_at = parse_twitter_timestamp(tweet.created_at)
    #         timestamps.append((tweet_id, created_at))
        
    #     # Sort by timestamp
    #     timestamps.sort(key=lambda x: x[1])
        
    #     # Find median time
    #     median_time = timestamps[len(timestamps) // 2][1]
        
    #     # Assign positions based on sorted order
    #     positions = {}
    #     rank = 1
    #     for tweet_id, created_at in timestamps:
    #         if created_at > median_time:
    #             positions[tweet_id] = 0
    #         else:
    #             positions[tweet_id] = rank
    #             rank += 1

    for tweet_id, tweet in tweets.items():
        # Base features
        base_features = extract_tweet_features(tweet)
        temporal = extract_temporal_features(tweet, reference_time)

        feature_parts = [base_features, temporal]

        # Add temporal positional encoding
        if include_temporal_encoding and positions:
            position = positions.get(tweet_id, 0)
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
