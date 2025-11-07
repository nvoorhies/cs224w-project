# PHEME Dataset Parser

A robust, Pydantic-based parser for the PHEME dataset with comprehensive validation and PyTorch Geometric (PyG) support.

## Features

- **Pydantic Validation**: All JSON data validated with comprehensive Pydantic models
- **Robust Error Handling**: Graceful handling of inconsistent or malformed data
- **Complete Data Access**: Access to all tweet, user, and annotation fields
- **Graph Structure Extraction**: Easy extraction of reply edges for graph neural networks
- **PyG-Ready**: Designed to facilitate conversion to PyTorch Geometric HeteroData objects
- **Comprehensive Tests**: Full test coverage with 19 passing tests
- **Dataset Verification**: Built-in verification script to validate and explore the dataset

## Installation

```bash
# Install dependencies
uv pip install -e .

# Install development dependencies (for tests)
uv pip install -e ".[dev]"
```

## Quick Start

### Load a Single Thread

```python
from pathlib import Path
from src import load_tweet_thread

# Load a complete tweet thread
thread_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads/rumours/580697361799876608")
thread = load_tweet_thread(thread_path)

# Access annotation data
print(f"Rumour type: {thread.annotation.is_rumour}")
print(f"Category: {thread.annotation.category}")

# Access tweets
all_tweets = thread.get_all_tweets()
print(f"Total tweets: {len(all_tweets)}")

# Access users
all_users = thread.get_all_users()
print(f"Unique users: {len(all_users)}")

# Get graph structure
edges = thread.structure.get_reply_edges()
print(f"Reply edges: {len(edges)}")
```

### Load Multiple Threads

```python
from src import load_dataset
from pathlib import Path

# Load threads from a specific story
story_path = Path("data/all-rnr-annotated-threads/germanwings-crash-all-rnr-threads")
threads = load_dataset(
    story_path,
    load_reactions=True,
    max_threads=10,
    skip_errors=True
)

print(f"Loaded {len(threads)} threads")
```

## Dataset Structure

The PHEME dataset contains 6,425 tweet threads across 9 different news stories:

```
data/all-rnr-annotated-threads/
├── charliehebdo-all-rnr-threads/     (2,079 threads)
├── ebola-essien-all-rnr-threads/     (14 threads)
├── ferguson-all-rnr-threads/         (1,143 threads)
├── germanwings-crash-all-rnr-threads/ (469 threads)
├── gurlitt-all-rnr-threads/          (138 threads)
├── ottawashooting-all-rnr-threads/   (890 threads)
├── prince-toronto-all-rnr-threads/   (233 threads)
├── putinmissing-all-rnr-threads/     (238 threads)
└── sydneysiege-all-rnr-threads/      (1,221 threads)
```

## Verification Script

Explore and verify the dataset:

```bash
# Verify first 20 threads
uv run python verify_dataset.py --max-threads 20

# Verify a specific story
uv run python verify_dataset.py --story germanwings-crash-all-rnr-threads

# Full dataset verification
uv run python verify_dataset.py
```

## Running Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run with coverage
uv run python -m pytest tests/ --cov=src --cov-report=html
```

All 19 tests pass ✅

## Examples

See `examples/basic_usage.py` for comprehensive usage examples:

```bash
uv run python examples/basic_usage.py
```

## Next Steps

For building the heterogeneous GAT:

1. **Feature Engineering**: Convert text to embeddings, extract temporal features
2. **PyG Graph Builder**: Create `HeteroData` objects for each thread
3. **Target Label Creation**: Define prediction task and labels
4. **Model Implementation**: Build heterogeneous GAT with temporal encoding

## License

MIT

## PyG Graph Builder

The graph builder converts PHEME threads to PyTorch Geometric HeteroData objects with full support for heterogeneous GAT training.

### Quick Start

```python
from src import load_tweet_thread, thread_to_graph, PHEMEDataset
from pathlib import Path

# Convert a single thread to a graph
thread = load_tweet_thread(Path("data/.../thread_id"))
graph = thread_to_graph(thread, user_edge_type="replies")

# Or use the Dataset class
dataset = PHEMEDataset(
    root='data/all-rnr-annotated-threads',
    stories=['germanwings-crash-all-rnr-threads'],
    max_threads=100,
    user_edge_type='replies'
)

graph = dataset[0]
print(graph)
```

### Graph Schema

The heterogeneous graph includes:

**Node Types:**
- `tweet`: Tweet nodes with 38 features (16 base + 6 temporal + 16 positional encoding)
- `user`: User nodes with 21 features

**Edge Types:**
- `(tweet, replies_to, tweet)`: Reply relationships
- `(tweet, replied_by, tweet)`: Reverse reply edges  
- `(user, posts, tweet)`: Authorship edges
- `(tweet, posted_by, user)`: Reverse authorship
- `(user, interacts_with, user)`: User interactions (optional)
- `(user, interacted_by, user)`: Reverse user interactions (optional)

### Features

**Tweet Features (38 dimensions):**
- Text features (6): length, word count, uppercase ratio, punctuation ratio, has exclamation, has question
- Entity features (4): hashtags, URLs, mentions, symbols
- Engagement features (3): log retweet count, log favorite count, log replies count
- Metadata (3): is retweet, is truncated, is reply
- Temporal (6): hour (sin/cos), day of week (sin/cos), time since thread start, normalized timestamp
- Positional encoding (16): sinusoidal encoding based on position in thread (Transformer-style)

**User Features (21 dimensions):**
- Profile (7): log followers, log friends, log statuses, log listed, log favorites, verified, protected
- Activity (3): friends/followers ratio, tweets per follower, favorites per tweet
- Metadata (3): geo enabled, default profile, default image
- Timezone (2): has timezone, normalized UTC offset
- Description (4): length, word count, has description, has URL
- Profile age (2): log account age in days, account age in years

### User-User Edges

The builder supports multiple strategies for constructing user→user edges:

```python
# Based on reply relationships (default)
graph = thread_to_graph(thread, user_edge_type="replies")

# Based on @mentions
graph = thread_to_graph(thread, user_edge_type="mentions")

# Combine both
graph = thread_to_graph(thread, user_edge_type="both")

# No user-user edges
graph = thread_to_graph(thread, user_edge_type="none")
```

### PyG Dataset

The `PHEMEDataset` class provides a PyG-compatible dataset with:
- Lazy loading of graphs
- Train/val/test splitting
- Filtering by story and rumour type
- Automatic feature extraction
- Ground truth labels

```python
# Create dataset
dataset = PHEMEDataset(
    root='data/all-rnr-annotated-threads',
    stories=['germanwings-crash-all-rnr-threads'],
    max_threads=100
)

# Create splits
train, val, test = dataset.get_split_datasets()

# Use with DataLoader
from torch_geometric.loader import DataLoader
loader = DataLoader(train, batch_size=32, shuffle=True)

for batch in loader:
    # batch is a batched HeteroData object
    print(batch['tweet'].x.shape)  # [num_tweets_in_batch, 38]
    print(batch['user'].x.shape)   # [num_users_in_batch, 21]
```

### Labels

Each graph includes multiple labels from the annotation:

```python
graph = dataset[0]

print(graph.y_rumour)         # 0=rumour, 1=non-rumour, 2=unverified
print(graph.y_misinformation) # 0 or 1
print(graph.y_true)           # 0 or 1
print(graph.y_turnaround)     # 0 or 1
```

### Examples

See `examples/graph_builder_usage.py` for comprehensive examples:

```bash
uv run python examples/graph_builder_usage.py
```

Examples include:
1. Converting single threads to graphs
2. Different user edge strategies
3. Using the PHEMEDataset
4. Creating train/val/test splits
5. Batching with DataLoader
6. Feature inspection
7. Exploring edge types
