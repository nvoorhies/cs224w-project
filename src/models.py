"""
Pydantic models for PHEME dataset JSON structures.

These models represent the various JSON files found in the PHEME dataset:
- Tweet JSON (source-tweets and reactions)
- Annotation JSON
- Structure JSON
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Tweet-related models
# ============================================================================

class HashtagEntity(BaseModel):
    """Hashtag entity in a tweet."""
    text: str
    indices: List[int]


class URLEntity(BaseModel):
    """URL entity in a tweet."""
    url: str
    indices: List[int]
    expanded_url: Optional[str] = None
    display_url: Optional[str] = None


class UserMentionEntity(BaseModel):
    """User mention entity in a tweet."""
    id: int
    id_str: str
    screen_name: str
    name: str
    indices: List[int]


class SymbolEntity(BaseModel):
    """Symbol entity (e.g., $AAPL) in a tweet."""
    text: str
    indices: List[int]


class MediaSize(BaseModel):
    """Size information for media."""
    h: int  # height
    w: int  # width
    resize: str  # "fit" or "crop"


class MediaEntity(BaseModel):
    """Media entity (photo, video, GIF) in a tweet."""
    id: int
    id_str: str
    url: str
    media_url: str
    media_url_https: str
    expanded_url: str
    display_url: str
    type: str  # "photo", "video", "animated_gif"
    indices: List[int]
    sizes: Dict[str, MediaSize]  # "large", "medium", "small", "thumb"

    # Optional fields for retweets/quoted tweets
    source_user_id: Optional[int] = None
    source_user_id_str: Optional[str] = None
    source_status_id: Optional[int] = None
    source_status_id_str: Optional[str] = None

    # Video-specific fields
    video_info: Optional[Dict[str, Any]] = None


class TweetEntities(BaseModel):
    """Entities extracted from a tweet (hashtags, URLs, mentions, etc.)."""
    hashtags: List[HashtagEntity] = Field(default_factory=list)
    urls: List[URLEntity] = Field(default_factory=list)
    user_mentions: List[UserMentionEntity] = Field(default_factory=list)
    symbols: List[SymbolEntity] = Field(default_factory=list)
    media: List[MediaEntity] = Field(default_factory=list)
    trends: List[Any] = Field(default_factory=list)  # Rare field, structure varies


class URLEntityShort(BaseModel):
    """URL entity in user profile."""
    url: str
    indices: List[int]
    expanded_url: Optional[str] = None
    display_url: Optional[str] = None


class UserEntities(BaseModel):
    """Entities in user profile."""
    url: Optional[Dict[str, List[URLEntityShort]]] = None
    description: Optional[Dict[str, List[URLEntityShort]]] = None


class TwitterUser(BaseModel):
    """Twitter user information."""
    id: int
    id_str: str
    screen_name: str
    name: str
    location: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None

    # Profile settings
    protected: bool = False
    verified: bool = False

    # Counts
    followers_count: int = 0
    friends_count: int = 0
    listed_count: int = 0
    favourites_count: int = 0
    statuses_count: int = 0

    # Profile appearance
    profile_image_url: Optional[str] = None
    profile_image_url_https: Optional[str] = None
    profile_banner_url: Optional[str] = None
    profile_background_image_url: Optional[str] = None
    profile_background_image_url_https: Optional[str] = None
    profile_background_color: Optional[str] = None
    profile_background_tile: Optional[bool] = None
    profile_link_color: Optional[str] = None
    profile_sidebar_border_color: Optional[str] = None
    profile_sidebar_fill_color: Optional[str] = None
    profile_text_color: Optional[str] = None
    profile_use_background_image: Optional[bool] = None

    # Other metadata
    created_at: str  # Twitter date string format
    lang: Optional[str] = None
    time_zone: Optional[str] = None
    utc_offset: Optional[int] = None
    geo_enabled: Optional[bool] = None

    # Relationship indicators
    following: Optional[bool] = None
    follow_request_sent: Optional[bool] = None
    notifications: Optional[bool] = None

    # Settings
    default_profile: Optional[bool] = None
    default_profile_image: Optional[bool] = None
    is_translator: Optional[bool] = None
    is_translation_enabled: Optional[bool] = None
    contributors_enabled: Optional[bool] = None

    # Entities
    entities: Optional[UserEntities] = None
    profile_location: Optional[Any] = None  # Can be null or object


class Coordinates(BaseModel):
    """Geographic coordinates."""
    type: str
    coordinates: List[float]  # [longitude, latitude]


class Place(BaseModel):
    """Place information for a tweet."""
    id: str
    url: Optional[str] = None
    place_type: Optional[str] = None
    name: Optional[str] = None
    full_name: Optional[str] = None
    country_code: Optional[str] = None
    country: Optional[str] = None
    bounding_box: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, Any]] = None


class ExtendedEntities(BaseModel):
    """Extended entities with additional media information."""
    media: List[MediaEntity] = Field(default_factory=list)


class Tweet(BaseModel):
    """Complete tweet object."""
    id: int
    id_str: str
    text: str
    created_at: str  # Twitter date string format

    # User who posted the tweet
    user: TwitterUser

    # Entities
    entities: TweetEntities

    # Engagement metrics
    retweet_count: int = 0
    favorite_count: int = 0

    # Reply/conversation information
    in_reply_to_status_id: Optional[int] = None
    in_reply_to_status_id_str: Optional[str] = None
    in_reply_to_user_id: Optional[int] = None
    in_reply_to_user_id_str: Optional[str] = None
    in_reply_to_screen_name: Optional[str] = None

    # Flags
    retweeted: bool = False
    favorited: bool = False
    truncated: bool = False

    # Location
    geo: Optional[Any] = None  # Can be null or Coordinates
    coordinates: Optional[Coordinates] = None
    place: Optional[Place] = None

    # Other metadata
    source: str
    lang: str
    contributors: Optional[List[Any]] = None

    # Content flags
    possibly_sensitive: Optional[bool] = None
    possibly_sensitive_appealable: Optional[bool] = None
    filter_level: Optional[str] = None

    # Extended media information
    extended_entities: Optional[ExtendedEntities] = None

    # Quoted/retweeted status (for retweets/quotes)
    quoted_status_id: Optional[int] = None
    quoted_status_id_str: Optional[str] = None
    quoted_status: Optional['Tweet'] = None
    retweeted_status: Optional['Tweet'] = None

    @field_validator('created_at')
    @classmethod
    def parse_created_at(cls, v: str) -> str:
        """Validate and potentially parse the created_at timestamp."""
        # Keep as string for now, but validate format
        # Format: "Tue Mar 24 10:51:21 +0000 2015"
        return v


# Need to update forward references for recursive Tweet model
Tweet.model_rebuild()


# ============================================================================
# Annotation models
# ============================================================================

class AnnotationLink(BaseModel):
    """Link in annotation with metadata about the source."""
    link: str
    mediatype: Optional[str] = None  # e.g., "news-media", "twitter", etc. (can be null)
    position: Optional[str] = None  # e.g., "for", "against", "neutral" (can be null)


class Annotation(BaseModel):
    """Annotation metadata for a tweet thread."""
    is_rumour: str  # "rumour", "non-rumour", "unverified"
    category: str  # Description/category of the rumour
    misinformation: int = 0  # 0 or 1 (coerced from int or string in data)
    true: int = 0  # 0 or 1 (coerced from int or string in data)
    is_turnaround: int = 0  # 0 or 1
    links: List[AnnotationLink] = Field(default_factory=list)

    @field_validator('misinformation', 'true', mode='before')
    @classmethod
    def coerce_to_int(cls, v: Any) -> int:
        """Convert string or int to int for consistency."""
        if isinstance(v, str):
            return int(v) if v.isdigit() else 0
        return int(v) if v is not None else 0


# ============================================================================
# Structure models
# ============================================================================

class ThreadStructure(BaseModel):
    """
    Represents the tree structure of tweet replies.

    This is a nested dictionary where:
    - Keys are tweet IDs (as strings)
    - Values are dictionaries of child tweets that replied to that tweet

    Example:
    {
        "580697361799876608": {
            "580697644227518464": [],
            "580698989974020096": {
                "580699224720875520": []
            }
        }
    }
    """
    structure: Dict[str, Any]  # Recursive structure, tweet_id -> {tweet_id -> {...}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThreadStructure':
        """Create ThreadStructure from raw dictionary."""
        return cls(structure=data)

    def get_root_tweet_id(self) -> Optional[str]:
        """Get the root tweet ID (should be the only top-level key)."""
        keys = list(self.structure.keys())
        return keys[0] if keys else None

    def get_all_tweet_ids(self) -> List[str]:
        """Recursively extract all tweet IDs from the structure."""
        def extract_ids(d: Dict[str, Any]) -> List[str]:
            ids = []
            for key, value in d.items():
                ids.append(key)
                # Handle both dict and list formats
                if isinstance(value, dict) and value:
                    ids.extend(extract_ids(value))
                # Lists mean no children, so skip
            return ids

        return extract_ids(self.structure)

    def get_reply_edges(self) -> List[tuple[str, str]]:
        """
        Extract all reply relationships as edges (parent_id, child_id).

        Returns:
            List of tuples (parent_tweet_id, child_tweet_id)
        """
        edges = []

        def traverse(parent_id: str, children: Any):
            # Handle both dict and list formats in the data
            if isinstance(children, list):
                # Empty list means no children
                return
            elif isinstance(children, dict):
                for child_id, grandchildren in children.items():
                    edges.append((parent_id, child_id))
                    if grandchildren:  # Recursively process if not empty
                        traverse(child_id, grandchildren)

        root_id = self.get_root_tweet_id()
        if root_id:
            children = self.structure[root_id]
            traverse(root_id, children)

        return edges


# ============================================================================
# Container models for a complete thread
# ============================================================================

class TweetThread(BaseModel):
    """Complete representation of a tweet thread."""
    thread_id: str  # The root tweet ID
    annotation: Annotation
    structure: ThreadStructure
    source_tweet: Tweet
    reaction_tweets: Dict[str, Tweet] = Field(default_factory=dict)  # tweet_id -> Tweet

    def get_all_tweets(self) -> Dict[str, Tweet]:
        """Get all tweets (source + reactions) as a dictionary."""
        all_tweets = {self.thread_id: self.source_tweet}
        all_tweets.update(self.reaction_tweets)
        return all_tweets

    def get_all_users(self) -> Dict[int, TwitterUser]:
        """Get all unique users from all tweets."""
        users = {}
        for tweet in self.get_all_tweets().values():
            users[tweet.user.id] = tweet.user
        return users
