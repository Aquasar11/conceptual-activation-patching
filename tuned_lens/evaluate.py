import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TunedLensConfig
from lens import TunedLens
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

    for input_ids in tqdm(val_loader, desc="Validating", dynamic_ncols=True):
        input_ids = input_ids.to(device, non_blocking=True)
        targets = input_ids[:, 1:].contiguous()  # (B, S-1)

        log_P_model, H = get_model_outputs(model, input_ids, config.layers, dtype)

        # model_top1 is the same for all layers — compute once
        model_top1 = log_P_model.argmax(dim=-1)  # (B, S-1)

        # Loop over layers: keeps peak memory at (B, S, V) instead of (L, B, S, V)
        for i in range(L):
            logits_l = lens.forward_layer(H[i], unembed_weight, i)  # (B, S-1, V)
            B, S, V = logits_l.shape

            log_P_l = F.log_softmax(logits_l, dim=-1)
            kld_l = F.kl_div(log_P_model, log_P_l, reduction="none", log_target=True).sum(-1).mean()

            top1_l = (logits_l.argmax(-1) == model_top1).float().mean()
            top5_l = (logits_l.topk(5, dim=-1).indices == model_top1.unsqueeze(-1)).any(-1).float().mean()

            ce_l = F.cross_entropy(logits_l.reshape(B * S, V), targets.reshape(B * S))

            kld_sum[i]  += kld_l.cpu()
            ce_sum[i]   += ce_l.cpu()
            top1_sum[i] += top1_l.cpu()
            top5_sum[i] += top5_l.cpu()

        num_batches += 1

    return {
        "kld":  kld_sum  / num_batches,
        "ce":   ce_sum   / num_batches,
        "top1": top1_sum / num_batches,
        "top5": top5_sum / num_batches,
    }
