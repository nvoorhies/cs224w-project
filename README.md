# PHEME Dataset Parser & Link Prediction Pipeline

A robust, Pydantic-based parser for the annotated PHEME dataset, complete with utilities to build PyTorch Geometric (PyG) heterogeneous graphs and a full link-prediction training pipeline.

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

## Requirements

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- GPU optional (training runs on CPU, but GPU accelerates experiments)

## Installation

```bash
# Install runtime dependencies
uv pip install -e .

# (Optional) install development dependencies
uv pip install -e ".[dev]"
```

## Dataset Setup

1. Download the annotated PHEME release from Figshare:
   ```bash
   mkdir -p data
   curl -L -o data/all-rnr-annotated-threads.tar.bz2 \
     https://ndownloader.figshare.com/files/11767817
   ```
2. Extract the archive (≈ 6,425 threads across nine stories):
   ```bash
   cd data
   tar -xjf all-rnr-annotated-threads.tar.bz2
   rm all-rnr-annotated-threads.tar.bz2  # optional cleanup
   cd ..
   ```
3. Verify the structure and annotations:
   ```bash
   uv run python check_dataset.py
   ```
   Expected directory tree (counts in parentheses indicate number of threads):

   ```
   data/all-rnr-annotated-threads/
   ├── charliehebdo-all-rnr-threads/     (2,079)
   ├── ebola-essien-all-rnr-threads/     (14)
   ├── ferguson-all-rnr-threads/         (1,143)
   ├── germanwings-crash-all-rnr-threads/ (469)
   ├── gurlitt-all-rnr-threads/          (138)
   ├── ottawashooting-all-rnr-threads/   (890)
  ├── prince-toronto-all-rnr-threads/   (233)
   ├── putinmissing-all-rnr-threads/     (238)
   └── sydneysiege-all-rnr-threads/      (1,221)
   ```

## Quick Start (Parsing Utilities)

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

## End-to-End Link Prediction Pipeline

Run the following from the project root once the dataset is in place.

```bash
# 1. Sanity check (loads a story, builds a model, runs forward pass)
uv run python test_link_prediction_setup.py

# 2. (Optional) Inspect dataset statistics/story splits
uv run python load_full_dataset.py

# 3. Train the heterogeneous GAT link predictor
#    --max-threads limits each split for quick experimentation.
#    Remove it for a full-data run.
uv run python train_link_prediction.py \
  --data-root data/all-rnr-annotated-threads \
  --epochs 1 \
  --batch-size 1 \
  --max-threads 100
```

Outputs are written to `outputs/` (`args.json`, `best_model.pt`, `results.json`).

### Latest Accuracy Metrics

100-epoch training run on the full dataset (batch size = 1, hidden = 64, heads = 2). Best checkpoint observed at epoch 28:

| Split       | Loss  | Accuracy | AUC-ROC | AUC-PR |
|-------------|-------|----------|---------|--------|
| Validation* | 0.0387 | 0.979 | 0.997 | 0.997 |
| Test        | 0.0644 | 0.964 | 0.994 | 0.993 |

\*Best-validation checkpoint is used for the reported test metrics.

For quick smoke tests, you can still run a single epoch with `--max-threads 100`; expect materially lower scores.

### Resuming from Checkpoints

The training script saves `outputs/best_model.pt` containing:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "val_loss": float,
    "val_metrics": dict,
}
```

To resume (or run inference), recreate the model with the same hyperparameters, then load the state:

```python
import torch
from pathlib import Path
from src.het_gat_model import HeteroGATLinkPrediction

in_channels_dict = {"tweet": 30, "user": 13}
model = HeteroGATLinkPrediction(
    in_channels_dict=in_channels_dict,
    hidden_channels=64,
    out_channels=32,
    num_layers=2,
    heads=2,
    dropout=0.5,
    link_pred_hidden_dim=64,
)

checkpoint = torch.load(Path("outputs/best_model.pt"), weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

For continued training, also restore the optimiser and optional scheduler:

```python
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

Resume your training loop from `start_epoch`. If you frequently reuse checkpoints, you can add a small CLI flag that wraps these calls before the main loop.

## Model Architecture & Design Choices

- **Encoder:** Two-layer heterogeneous Graph Attention Network (`HeteroConv` with `GATConv`) covering six edge relations (tweet↔tweet, user↔tweet, user↔user).
- **Hidden dimensions:** 64 hidden channels with 2 attention heads per layer (concatenated); ReLU activations and dropout between layers.
- **Link predictor:** Multi-layer perceptron (`64 → 32 → 1`) applied to concatenated user embeddings; dropout (0.5) for regularisation.
- **Feature design:**
  - Tweets: 30-dimensional vector (entities, engagement, metadata, temporal context, positional encodings).
  - Users: 13-dimensional profile/activity/metadata features.
- **Training strategy:** Adaptive negative sampling per graph (ratio = 1), binary cross-entropy loss on link logits, `Adam` optimiser with `ReduceLROnPlateau` scheduler and gradient clipping.
- **Temporal handling:** Parsed timestamps converted into continuous features (hour, weekday, scaled epoch, minutes since thread start) to capture dynamics.
- **Why this setup:** Heterogeneous attention allows tweets and users to exchange information across different relation types, while a lightweight MLP head keeps link prediction focused on user-user interactions.

### What else to try

- Increase `--num-layers`, `--hidden-channels`, or attention `--heads` for richer expressivity (with corresponding regularisation).
- Add textual embeddings (e.g., sentence transformers) into the tweet feature vectors.
- Experiment with additional negative sampling ratios or curriculum schedules.
- Fine-tune dropout/weight decay, or introduce edge dropout to mitigate overfitting.
- Swap the MLP link predictor for dot-product or bilinear forms if you prefer parameter sharing.

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
- `tweet`: Tweet nodes with 35 features (15 base + 4 temporal + 16 positional encoding)
- `user`: User nodes with 13 features

**Edge Types:**
- `(tweet, replies_to, tweet)`: Reply relationships
- `(tweet, replied_by, tweet)`: Reverse reply edges  
- `(user, posts, tweet)`: Authorship edges
- `(tweet, posted_by, user)`: Reverse authorship
- `(user, interacts_with, user)`: User interactions (optional)
- `(user, interacted_by, user)`: Reverse user interactions (optional)

### Features

**Tweet Features (30 dimensions):**
- Text placeholder (1): reserved slot for future text embeddings
- Entity features (4): hashtags, URLs, mentions, symbols
- Engagement features (2): log retweet count, log favorite count
- Metadata (3): is retweet, is truncated, is reply
- Temporal context (4): hour of day, day of week, scaled timestamp, minutes since thread start
- Positional encoding (16): sinusoidal encoding based on depth in the reply tree

**User Features (13 dimensions):**
- Profile (7): log followers, friends, statuses, listed, favourites, plus verified & protected flags
- Activity (3): friends/followers ratio, tweets per follower, favourites per tweet
- Metadata (3): geo enabled, default profile, default profile image

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
    print(batch['tweet'].x.shape)  # [num_tweets_in_batch, 35]
    print(batch['user'].x.shape)   # [num_users_in_batch, 13]
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
