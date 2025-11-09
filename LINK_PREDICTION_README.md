# Link Prediction Training Guide

This guide explains how to train a Graph Attention Transformer (GAT) model for link prediction on the PHEME dataset.

## Overview

The task is to predict whether a tweet creates a "requests to follow" link between users. The model uses a heterogeneous GAT architecture that processes temporal heterogeneous networks with tweet and user nodes.

## Architecture

The model consists of:
1. **Heterogeneous GAT Encoder**: Processes heterogeneous graphs with multiple node types (tweet, user) and edge types (replies, authorship, interactions)
2. **Link Predictor**: Predicts the probability of a link (follow request) between two users based on their embeddings

## Dataset

The PHEME dataset contains 9 news stories:
- charliehebdo-all-rnr-threads
- ebola-essien-all-rnr-threads
- ferguson-all-rnr-threads
- germanwings-crash-all-rnr-threads
- gurlitt-all-rnr-threads
- ottawashooting-all-rnr-threads
- prince-toronto-all-rnr-threads
- putinmissing-all-rnr-threads
- sydneysiege-all-rnr-threads

The stories are split into train/val/test sets (default: 70%/15%/15%).

## Installation

Make sure you have all dependencies installed:

```bash
uv pip install -e .
```

## Usage

### Basic Training

```bash
python train_link_prediction.py \
    --data-root data/all-rnr-annotated-threads \
    --epochs 100 \
    --batch-size 1 \
    --lr 0.001
```

### Full Training with All Options

```bash
python train_link_prediction.py \
    --data-root data/all-rnr-annotated-threads \
    --max-threads 1000 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --split-seed 42 \
    --user-edge-type replies \
    --num-negative-samples 1 \
    --hidden-channels 64 \
    --out-channels 32 \
    --num-layers 2 \
    --heads 2 \
    --dropout 0.5 \
    --link-pred-hidden-dim 64 \
    --batch-size 1 \
    --epochs 100 \
    --lr 0.001 \
    --weight-decay 5e-4 \
    --patience 10 \
    --device cuda \
    --output-dir ./outputs
```

### Arguments

#### Dataset Arguments
- `--data-root`: Root directory of PHEME dataset (required)
- `--max-threads`: Maximum number of threads to load (None = all)
- `--train-ratio`: Ratio of stories for training (default: 0.7)
- `--val-ratio`: Ratio of stories for validation (default: 0.15)
- `--test-ratio`: Ratio of stories for testing (default: 0.15)
- `--split-seed`: Random seed for story splitting (default: 42)
- `--user-edge-type`: How to construct user-user edges: `mentions`, `replies`, `both`, `none` (default: `replies`)
- `--num-negative-samples`: Number of negative samples per positive sample (default: 1)

#### Model Arguments
- `--hidden-channels`: Hidden dimension for GAT layers (default: 64)
- `--out-channels`: Output dimension for node embeddings (default: 32)
- `--num-layers`: Number of GAT layers (default: 2)
- `--heads`: Number of attention heads (default: 2)
- `--dropout`: Dropout probability (default: 0.5)
- `--link-pred-hidden-dim`: Hidden dimension for link predictor (default: 64)

#### Training Arguments
- `--batch-size`: Batch size (default: 1, note: for heterogeneous graphs, batch_size=1 is recommended)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay (default: 5e-4)
- `--patience`: Early stopping patience (default: 10)
- `--device`: Device to use: `cuda` or `cpu` (default: auto-detect)

#### Output Arguments
- `--output-dir`: Output directory for checkpoints and logs (default: `./outputs`)
- `--save-interval`: Save checkpoint every N epochs (default: 10)

## Testing the Setup

Before training, you can test the setup:

```bash
python test_link_prediction_setup.py
```

This will test:
1. Dataset loading
2. Model creation
3. Forward pass
4. Link prediction generation

## Output

The training script saves:
- `args.json`: Training arguments
- `best_model.pt`: Best model checkpoint (lowest validation loss)
- `checkpoint_epoch_N.pt`: Checkpoints every N epochs
- `results.json`: Training results and metrics

## Evaluation Metrics

The model is evaluated using:
- **Loss**: Binary cross-entropy loss
- **Accuracy**: Classification accuracy (threshold=0.5)
- **AUC-ROC**: Area under the ROC curve
- **AUC-PR**: Area under the precision-recall curve

## Link Prediction Task

The model predicts whether a user will send a "follow request" to another user based on:
- User features (profile, activity, metadata)
- Tweet features (text, entities, engagement, temporal)
- Graph structure (reply relationships, mentions, interactions)
- Temporal information (position in thread, timestamp)

Positive samples are generated from:
- Existing user-user interaction edges (replies, mentions)
- User pairs who interact in the same thread

Negative samples are randomly sampled user pairs that don't interact.

## Model Architecture Details

### Heterogeneous GAT Layer
- Uses PyTorch Geometric's `HeteroConv` to handle multiple edge types
- Applies GAT convolution to each edge type independently
- Aggregates node embeddings across edge types using mean pooling

### Temporal Encoding
- Includes temporal positional encoding based on tweet position in thread
- Uses temporal features (hour, day of week, timestamp, time since thread start)

### Link Predictor
- MLP with 3 hidden layers
- Takes concatenated source and destination user embeddings
- Outputs probability of a link between users

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` to 1 (recommended for heterogeneous graphs)
- Reduce `--max-threads` to limit dataset size
- Reduce `--hidden-channels` or `--out-channels`

### Poor Performance
- Try different `--user-edge-type` (e.g., `both` for mentions and replies)
- Increase `--num-layers` or `--heads`
- Adjust `--lr` and `--weight-decay`
- Increase `--num-negative-samples` for better class balance

### No Positive Samples
- Check that graphs have user-user interaction edges
- Try `--user-edge-type both` to include more interactions
- Verify dataset is loaded correctly

## Example Results

After training, you should see output like:

```
Epoch 1/100
  Train Loss: 0.6931
  Val Loss: 0.6928
  Val Accuracy: 0.5000
  Val AUC-ROC: 0.5123
  Val AUC-PR: 0.5034
```

## Next Steps

1. **Hyperparameter Tuning**: Experiment with different model architectures and training parameters
2. **Feature Engineering**: Add more features (e.g., text embeddings, sentiment analysis)
3. **Evaluation**: Analyze predictions on specific stories or user types
4. **Visualization**: Visualize attention weights and link predictions

## References

- PHEME Dataset: [https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/2068650](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/2068650)
- PyTorch Geometric: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
- Graph Attention Networks: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

