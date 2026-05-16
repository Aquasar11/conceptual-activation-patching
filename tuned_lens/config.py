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
    # These are 1-indexed and correspond to outputs.hidden_states indices:
    #   hidden_states[0]  = embedding output
    #   hidden_states[l]  = output of transformer layer l  (l = 1 ... num_layers)
    layers: List[int] = field(default_factory=lambda: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])

    # Training
    seq_len: int = 512
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-3
    lambda_reg: float = 1e-5       # weight on ||W-I||_F^2 + ||b||^2 regularization

    # Checkpointing and logging
    output_dir: str = "./outputs/lens_checkpoints"
    tensorboard_dir: str = "./runs"
    log_every: int = 100           # steps between console log lines and TensorBoard train logs
    eval_every_epoch: bool = True  # run evaluation after each epoch; saves best checkpoint

    # Hardware
    dtype: str = "bfloat16"        # model dtype: "bfloat16" or "float16" or "float32"
    device: str = "cuda"
