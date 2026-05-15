# Tuned Lens

Trains a per-layer affine transform (tuned lens) on a VLM's unembedding head, allowing you to read out predicted tokens at any intermediate layer.

## Concept

The final unembedding head `U_model` maps hidden states to vocabulary logits. For each target layer `l`, we learn an affine transform:

```
U_l = U_model @ W_l.T + b_l
```

where `W_l` (D×D, initialized to identity) and `b_l` (D, initialized to zero) are trainable. The lens for layer `l` with hidden state `h_l` predicts:

```
logits_l = h_l @ U_l.T
```

**Training objective** (text tokens only, wikitext-103):

```
Loss = KLD(P_l || P_model) + λ * (||W_l - I||_F² + ||b_l||²)
```

The KLD measures how far the layer's predicted distribution is from the final model's distribution. The regularization keeps `W_l` close to identity (and `b_l` close to zero), so the lens degrades gracefully toward the final unembedding when not needed.

Multiple layers are trained jointly in a single run. Because the backbone is frozen and each layer's lens parameters are independent, training all layers together is mathematically identical to training each layer separately.

## Layer Numbering

Layers are **1-indexed** and correspond directly to `outputs.hidden_states[l]`:

| Index | What it is |
|---|---|
| 0 | Embedding output — not a transformer layer, do not use |
| 1 | Output of transformer layer 1 (first layer) |
| ... | ... |
| 28 | Output of transformer layer 28 (last layer for Qwen2.5-VL-7B) |

Valid range: `layers = [1, ..., 28]`.

## Setup

```bash
pip install -r ../requirements.txt
```

The model (`Qwen/Qwen2.5-VL-7B-Instruct`) and dataset (`Salesforce/wikitext`) are downloaded automatically by HuggingFace on first run, or loaded from cache if already present.

## Usage

```bash
# Use all defaults
python run_train.py

# Override specific fields
python run_train.py --layers 1 5 10 15 20 27 --batch_size 8 --num_epochs 5

# Load from a JSON config file (CLI args take precedence)
python run_train.py --config my_config.json --learning_rate 5e-4
```

**Example JSON config file:**

```json
{
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "dataset_config": "wikitext-103-raw-v1",
    "layers": [1, 5, 10, 15, 20, 25, 28],
    "batch_size": 8,
    "num_epochs": 3,
    "learning_rate": 1e-3,
    "lambda_reg": 1e-4,
    "output_dir": "/path/on/server/lens_checkpoints",
    "tensorboard_dir": "/path/on/server/runs"
}
```

## Monitoring with TensorBoard

```bash
tensorboard --logdir ./runs
```

Logged metrics:

| Tag | Description |
|---|---|
| `train/total_loss` | Total loss summed over all layers |
| `train/layer_NN_kld` | Per-layer KLD (train) |
| `train/layer_NN_reg` | Per-layer regularization term (train) |
| `val/mean_kld` | Mean KLD across all layers (validation) — used for best model selection |
| `val/layer_NN_kld` | Per-layer KLD (validation) |
| `val/layer_NN_ce` | Per-layer cross-entropy on next-token prediction (validation) |

Train metrics are logged every `log_every` steps. Validation metrics are logged at the end of each epoch.

## Checkpoints

Only the best checkpoint (lowest mean validation KLD across all trained layers) is saved:

```
output_dir/
└── lens_best.pt
```

Load a saved lens:

```python
from lens import TunedLens
lens = TunedLens.load("lens_checkpoints/lens_best.pt", device="cuda")
```

## Configuration Reference

All options are in `config.py` and can be set via CLI or JSON file:

| Field | Default | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID |
| `dataset_name` | `Salesforce/wikitext` | HuggingFace dataset ID |
| `dataset_config` | `wikitext-103-raw-v1` | Dataset config (`wikitext-2-raw-v1` for smaller) |
| `layers` | `[1,5,10,15,20,25,27]` | Which layers to train (1–28 for this model) |
| `seq_len` | `512` | Sequence length for chunked text |
| `batch_size` | `4` | Training batch size |
| `num_epochs` | `3` | Number of training epochs |
| `learning_rate` | `1e-3` | AdamW learning rate |
| `lambda_reg` | `1e-4` | Regularization weight |
| `output_dir` | `./lens_checkpoints` | Where to save the best checkpoint |
| `tensorboard_dir` | `./runs` | TensorBoard log directory |
| `log_every` | `100` | Steps between console/TensorBoard train logs |
| `eval_every_epoch` | `True` | Run validation and checkpoint at end of each epoch |
| `dtype` | `bfloat16` | Model dtype (`bfloat16`, `float16`, `float32`) |
| `device` | `cuda` | Device string |

## File Structure

```
tuned_lens/
├── config.py       — All hyperparameters in one dataclass
├── lens.py         — TunedLens nn.Module
├── data.py         — Dataset loading and tokenization
├── train.py        — Training loop
├── evaluate.py     — Validation evaluation
└── run_train.py    — Entry point
```
