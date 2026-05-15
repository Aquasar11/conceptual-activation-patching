import torch
import torch.nn.functional as F
from typing import Tuple


def tuned_lens_loss(
    logits_all: torch.Tensor,
    log_P_model: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    lambda_reg: float,
    hidden_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized tuned lens loss across all layers.

    Args:
        logits_all:  Raw logits from the lens.          Shape: (L, B, S, V)
        log_P_model: Log-probs of the frozen model.     Shape: (B, S, V)
        W:           Lens weight matrices.              Shape: (L, D, D)
        b:           Lens bias vectors.                 Shape: (L, D)
        lambda_reg:  Regularization coefficient.
        hidden_dim:  D (needed to build the identity reference).

    Returns:
        total_loss:      Scalar — sum of KLD + lambda_reg * reg across all layers.
        kld_per_layer:   Shape (L,) — mean KLD(P_l || P_model) per layer.
        reg_per_layer:   Shape (L,) — ||W_l - I||_F^2 + ||b_l||^2 per layer.
    """
    # KLD(P_l || P_model) = sum_v P_l * (log P_l - log P_model)
    log_P_all = F.log_softmax(logits_all, dim=-1)                        # (L, B, S, V)
    kld_per_layer = (log_P_all.exp() * (log_P_all - log_P_model)).sum(-1).mean((1, 2))  # (L,)

    # ||W_l - I||_F^2 + ||b_l||^2  for each layer
    I = torch.eye(hidden_dim, device=W.device, dtype=W.dtype).unsqueeze(0)  # (1, D, D)
    reg_per_layer = (W - I).pow(2).sum((1, 2)) + b.pow(2).sum(1)            # (L,)

    total_loss = kld_per_layer.sum() + lambda_reg * reg_per_layer.sum()

    return total_loss, kld_per_layer, reg_per_layer
