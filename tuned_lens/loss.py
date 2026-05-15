import torch
import torch.nn.functional as F
from typing import Tuple


def tuned_lens_loss_layer(
    logits_l: torch.Tensor,
    log_P_model: torch.Tensor,
    W_l: torch.Tensor,
    b_l: torch.Tensor,
    lambda_reg: float,
    hidden_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss for a single layer. Keeps peak memory at (B, S, V) instead of (L, B, S, V).

    Args:
        logits_l:    Raw logits from the lens for one layer. Shape: (B, S, V)
        log_P_model: Log-probs of the frozen model.          Shape: (B, S, V)
        W_l:         Lens weight matrix for this layer.      Shape: (D, D)
        b_l:         Lens bias for this layer.               Shape: (D,)
        lambda_reg:  Regularization coefficient.
        hidden_dim:  D (needed to build the identity reference).

    Returns:
        total_loss: Scalar.
        kld:        Scalar — mean KLD(P_l || P_model).
        reg:        Scalar — ||W_l - I||_F^2 + ||b_l||^2.
    """
    log_P_l = F.log_softmax(logits_l, dim=-1)                               # (B, S, V)
    kld = F.kl_div(log_P_model, log_P_l, reduction="none", log_target=True).sum(-1).mean()

    I = torch.eye(hidden_dim, device=W_l.device, dtype=W_l.dtype)
    reg = (W_l - I).pow(2).sum() + b_l.pow(2).sum()

    return kld + lambda_reg * reg, kld, reg
