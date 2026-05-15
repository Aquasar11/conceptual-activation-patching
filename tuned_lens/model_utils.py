import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import List, Tuple

from config import TunedLensConfig


def load_model(config: TunedLensConfig):
    """
    Load the VLM backbone (text-only via AutoModelForCausalLM), freeze all parameters,
    and extract the unembedding weight in float32.

    Returns:
        model:          Frozen language model on config.device.
        unembed_weight: model.lm_head.weight as float32. Shape: (V, D)
        hidden_dim:     D
    """
    dtype = getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Cast to float32 once — lens parameters and hidden states are float32
    unembed_weight = model.lm_head.weight.detach().float()
    hidden_dim = unembed_weight.shape[1]

    return model, unembed_weight, hidden_dim


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
    log_P_model = F.log_softmax(
        outputs.logits[:, :-1, :].float(), dim=-1
    ).contiguous()  # (B, S-1, V)

    # Extract and stack only the layers we need — avoid casting all 29
    H = torch.stack(
        [outputs.hidden_states[l][:, :-1, :].float() for l in layer_indices],
        dim=0,
    )  # (L, B, S-1, D)

    return log_P_model, H
