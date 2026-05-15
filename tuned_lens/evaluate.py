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
    unembed_weight: torch.Tensor,
    hidden_dim: int,
) -> dict:
    """
    Evaluate the tuned lens on the validation set.

    Args:
        unembed_weight: model.lm_head.weight as float32 (V, D) — pass the one already
                        held by the caller to avoid a duplicate GPU allocation.
        hidden_dim:     D, passed for the same reason.

    Returns a dict with Tensors of shape (L,), one value per trained layer:
        "kld":  mean KLD(P_l || P_model)
        "ce":   mean cross-entropy against ground-truth next token
        "top1": fraction of positions where lens top-1 == model top-1
        "top5": fraction of positions where model top-1 is in lens top-5
    """
    model.eval()
    lens.eval()

    dtype = getattr(torch, config.dtype)
    device = config.device
    L = len(config.layers)

    kld_sum  = torch.zeros(L)
    ce_sum   = torch.zeros(L)
    top1_sum = torch.zeros(L)
    top5_sum = torch.zeros(L)
    num_batches = 0

    for input_ids in val_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = input_ids[:, 1:].contiguous()  # (B, S-1)

        log_P_model, H = get_model_outputs(model, input_ids, config.layers, dtype)

        logits_all = lens(H, unembed_weight)  # (L, B, S-1, V)

        # KLD — reuse shared loss function (lambda_reg=0: no reg during eval)
        _, kld_per_layer, _ = tuned_lens_loss(
            logits_all, log_P_model, lens.W, lens.b, 0.0, hidden_dim
        )

        _, B, S, V = logits_all.shape

        # Top-1 / Top-5 agreement with the model's own predicted token
        model_top1 = log_P_model.argmax(dim=-1)                                           # (B, S)
        top1 = (logits_all.argmax(-1) == model_top1.unsqueeze(0)).float().mean((1, 2))    # (L,)
        top5 = (
            logits_all.topk(5, dim=-1).indices == model_top1.unsqueeze(0).unsqueeze(-1)
        ).any(-1).float().mean((1, 2))                                                     # (L,)

        # Cross-entropy against ground-truth next token
        targets_expanded = targets.unsqueeze(0).expand(L, B, S).reshape(L * B * S)
        ce_per_layer = (
            F.cross_entropy(logits_all.reshape(L * B * S, V), targets_expanded, reduction="none")
            .view(L, B, S)
            .mean((1, 2))
        )  # (L,)

        kld_sum  += kld_per_layer.cpu()
        ce_sum   += ce_per_layer.cpu()
        top1_sum += top1.cpu()
        top5_sum += top5.cpu()
        num_batches += 1

    return {
        "kld":  kld_sum  / num_batches,
        "ce":   ce_sum   / num_batches,
        "top1": top1_sum / num_batches,
        "top5": top5_sum / num_batches,
    }
