from dataclasses import dataclass, field
from typing import List


@dataclass
class TunedLensConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Dataset
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"

    # Which transformer layers to train lenses for.
    # Indexing (Qwen2.5-VL, @capture_outputs convention):
    #   hidden_states[0]       = embedding
    #   hidden_states[1..27]   = pre-norm output of transformer layers 1..27
    #   hidden_states[28]      = last_hidden_state = POST-NORM output of layer 28
    layers: List[int] = field(default_factory=lambda: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])

    # Indices in `layers` whose hidden states are already post-norm (final_norm must NOT be applied).
    # For Qwen2.5-VL-7B: hidden_states[28] = last_hidden_state is post-norm.
    postnorm_layers: List[int] = field(default_factory=lambda: [28])

    # Training
    seq_len: int = 512
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-3
    lambda_reg: float = 1e-5       # weight on ||W-I||_F^2 + ||b||^2 regularization
    max_grad_norm: float = 1.0     # gradient clipping — prevents W divergence for late layers

    # Checkpointing and logging
    output_dir: str = "./outputs/lens_checkpoints"
    tensorboard_dir: str = "./runs"
    log_every: int = 100           # steps between console log lines and TensorBoard train logs
    eval_every_epoch: bool = True  # run evaluation after each epoch; saves best checkpoint

    # Hardware
    dtype: str = "bfloat16"        # model dtype: "bfloat16" or "float16" or "float32"
    device: str = "cuda"
