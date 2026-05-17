import os
import torch
import torch.nn as nn
from typing import List, Optional


class TunedLens(nn.Module):
    """
    Per-layer affine transforms over the model's unembedding head.

    Parameters (one set per trained layer):
        W: (L, D, D)  initialized to identity
        b: (L, D)     initialized to zero

    forward_layer processes one layer at a time to keep peak memory at (B, S, V)
    instead of (L, B, S, V):
        H_flat        = H_l.reshape(B*S, D)
        H_transformed = H_flat @ W[i]                             -> (B*S, D)
        H_normed      = final_norm(H_transformed)  [if pre-norm] -> (B*S, D)
        logits        = H_normed @ U.T                            -> (B*S, V)
        bias          = H_flat @ b[i]                             -> (B*S,)
        output        = (logits + bias).view(B, S, V)

    final_norm must be applied for pre-norm layers (hidden_states[1..num_layers-1]) but
    must be SKIPPED for post-norm layers (hidden_states[num_layers] = last_hidden_state,
    which is already normalized). _apply_norm[i] encodes this per position at init time.

    Note on b: b adds scalar h·b_l to every vocab logit for each token position,
    which is softmax shift-invariant — b does not affect predicted distributions.
    Regularization keeps it at zero. W carries all expressive power.
    """

    def __init__(
        self,
        hidden_dim: int,
        layer_indices: List[int],
        final_norm: Optional[nn.Module] = None,
        postnorm_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_indices = layer_indices
        self.final_norm = final_norm  # frozen model.model.language_model.norm; not a trainable parameter

        postnorm_set = set(postnorm_indices or [])
        # Precomputed per-position flag: True = apply norm, False = hidden state is already post-norm.
        # Stored as a plain Python list so torch.compile specialises on layer_i at trace time
        # (one CUDA graph per layer_i value) and folds the branch statically.
        self._apply_norm: List[bool] = [
            (final_norm is not None) and (idx not in postnorm_set)
            for idx in layer_indices
        ]

        L = len(layer_indices)
        # Single 3D parameters — no stack() needed on every forward pass
        self.W = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).expand(L, -1, -1).clone()
        )  # (L, D, D)
        self.b = nn.Parameter(torch.zeros(L, hidden_dim))  # (L, D)

    def forward_layer(self, H_l: torch.Tensor, unembed_weight: torch.Tensor, layer_i: int) -> torch.Tensor:
        """
        Single-layer forward to keep peak memory at (B, S, V) instead of (L, B, S, V).

        Args:
            H_l:            Hidden states for one layer. Shape: (B, S, D)
            unembed_weight: model.lm_head.weight.        Shape: (V, D)
            layer_i:        Index into self.W, self.b, self._apply_norm (0-based within trained layers).
        Returns:
            logits: Shape (B, S, V)
        """
        B, S, D = H_l.shape
        H_flat = H_l.reshape(B * S, D)                          # (B*S, D)
        H_transformed = H_flat @ self.W[layer_i]                # (B*S, D)
        if self._apply_norm[layer_i]:
            H_transformed = self.final_norm(H_transformed)      # (B*S, D)
        logits = H_transformed @ unembed_weight.T               # (B*S, V)
        bias = H_flat @ self.b[layer_i]                         # (B*S,)
        return (logits + bias.unsqueeze(-1)).view(B, S, -1)     # (B, S, V)

    def save_layers(self, output_dir: str):
        """Save each layer's W and b to its own file: layer_NN.pt"""
        os.makedirs(output_dir, exist_ok=True)
        for i, layer_idx in enumerate(self.layer_indices):
            torch.save(
                {
                    "hidden_dim": self.hidden_dim,
                    "layer_idx": layer_idx,
                    "W": self.W[i].detach().cpu(),
                    "b": self.b[i].detach().cpu(),
                },
                os.path.join(output_dir, f"layer_{layer_idx:02d}.pt"),
            )

    @classmethod
    def load_layers(
        cls,
        output_dir: str,
        layer_indices: List[int],
        device: str = "cpu",
        final_norm: Optional[nn.Module] = None,
        postnorm_indices: Optional[List[int]] = None,
    ) -> "TunedLens":
        """Load per-layer files written by save_layers() and reconstruct a TunedLens.

        final_norm and postnorm_indices must be passed to restore correct norm behaviour;
        omitting them means the lens forward will skip normalisation (wrong for inference).
        """
        first = torch.load(
            os.path.join(output_dir, f"layer_{layer_indices[0]:02d}.pt"),
            map_location=device,
            weights_only=True,
        )
        hidden_dim = first["hidden_dim"]
        lens = cls(hidden_dim, layer_indices, final_norm=final_norm, postnorm_indices=postnorm_indices).to(device)
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Reuse the already-loaded first checkpoint instead of re-reading the file
                ckpt = first if i == 0 else torch.load(
                    os.path.join(output_dir, f"layer_{layer_idx:02d}.pt"),
                    map_location=device,
                    weights_only=True,
                )
                lens.W.data[i].copy_(ckpt["W"])
                lens.b.data[i].copy_(ckpt["b"])
        return lens
