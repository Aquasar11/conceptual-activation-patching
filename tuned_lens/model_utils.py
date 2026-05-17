import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from typing import List, Tuple

from config import TunedLensConfig


def load_model(config: TunedLensConfig):
    """
    Load the VLM backbone, freeze all parameters, and extract the unembedding weight in float32.

    Returns:
        model:          Frozen language model on config.device.
        unembed_weight: model.lm_head.weight in model native dtype (bfloat16). Shape: (V, D)
        hidden_dim:     D
        final_norm:     model.model.norm — the RMSNorm applied before lm_head.
    """
    dtype = getattr(torch, config.dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        dtype=dtype,
        device_map=config.device,
    )

    # Drop the vision tower — text-only training never uses it
    del model.model.visual
    torch.cuda.empty_cache()

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Keep in model's native bfloat16 — lens forward runs under autocast
    unembed_weight = model.lm_head.weight.detach()
    hidden_dim = unembed_weight.shape[1]
    final_norm = model.model.language_model.norm  # frozen RMSNorm; must be applied before lm_head

    return model, unembed_weight, hidden_dim, final_norm


@torch.no_grad()
def get_model_outputs(
    model,
    input_ids: torch.Tensor,
    layer_indices: List[int],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single frozen-model forward pass and return what the lens needs.

    Args:
        model:         Frozen language model.
        input_ids:     Token ids. Shape: (B, S)
        layer_indices: Which transformer layers to extract (1-indexed).
        dtype:         Model compute dtype (e.g. torch.bfloat16).

    Returns:
        log_P_model: Log-softmax of final logits, shifted.  Shape: (B, S-1, V)
        H:           Stacked hidden states for target layers, shifted, float32.
                     Shape: (L, B, S-1, D)
    """
    with torch.autocast(device_type="cuda", dtype=dtype):
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    # Shift: position t predicts token t+1
    # .detach() is redundant under @no_grad but makes the intent explicit
    log_P_model = F.log_softmax(
        outputs.logits[:, :-1, :].detach().float(), dim=-1
    ).contiguous()  # (B, S-1, V)

    # Keep bfloat16 — lens forward runs under autocast, no float32 cast needed here
    H = torch.stack(
        [outputs.hidden_states[l][:, :-1, :].detach() for l in layer_indices],
        dim=0,
    )  # (L, B, S-1, D)

    return log_P_model, H
