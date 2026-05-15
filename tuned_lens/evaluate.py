import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TunedLensConfig
from lens import TunedLens
from loss import tuned_lens_loss
from model_utils import get_model_outputs


@torch.no_grad()
def evaluate(
    model,
    lens: TunedLens,
    val_loader: DataLoader,
    config: TunedLensConfig,
) -> dict:
    """
    Evaluate the tuned lens on the validation set.

    Returns a dict with:
        "kld": Tensor (L,) — mean KLD(P_l || P_model) per layer
        "ce":  Tensor (L,) — mean cross-entropy (next-token prediction) per layer
    """
    model.eval()
    lens.eval()

    dtype = getattr(torch, config.dtype)
    device = config.device
    unembed_weight = model.lm_head.weight.detach().float()
    hidden_dim = unembed_weight.shape[1]
    L = len(config.layers)

    kld_sum = torch.zeros(L)
    ce_sum = torch.zeros(L)
    num_batches = 0

    for input_ids in val_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = input_ids[:, 1:].contiguous()  # (B, S-1)

        log_P_model, H = get_model_outputs(model, input_ids, config.layers, dtype)

        logits_all = lens(H, unembed_weight)   # (L, B, S-1, V)

        # KLD via shared loss function (lambda_reg=0 — no regularization during eval)
        _, kld_per_layer, _ = tuned_lens_loss(
            logits_all, log_P_model, lens.W, lens.b, 0.0, hidden_dim
        )

        # Cross-entropy: how well does each layer predict the actual next token?
        LB, B, S, V = logits_all.shape
        targets_expanded = targets.unsqueeze(0).expand(L, B, S).reshape(L * B * S)
        ce_per_layer = (
            F.cross_entropy(logits_all.reshape(L * B * S, V), targets_expanded, reduction="none")
            .view(L, B, S)
            .mean((1, 2))
        )  # (L,)

        kld_sum += kld_per_layer.cpu()
        ce_sum += ce_per_layer.cpu()
        num_batches += 1

    return {
        "kld": kld_sum / num_batches,
        "ce": ce_sum / num_batches,
    }
