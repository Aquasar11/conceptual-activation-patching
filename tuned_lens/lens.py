import os
import torch
import torch.nn as nn
from typing import List


class TunedLens(nn.Module):
    """
    Per-layer affine transforms over the model's unembedding head.

    Parameterization (for all L layers at once):
        W: (L, D, D)  initialized to identity
        b: (L, D)     initialized to zero

    For a stack of hidden states H: (L, B, S, D):
        H_flat       = H.reshape(L, B*S, D)
        H_transformed = bmm(H_flat, W)          -> (L, B*S, D)
        logits        = H_transformed @ U.T     -> (L, B*S, V)
        bias          = bmm(H_flat, b[:,None])  -> (L, B*S, 1)   [shift-invariant, see note]
        output        = (logits + bias).view(L, B, S, V)

    Note on b: b adds scalar h·b_l to every vocab logit for each token position,
    which is softmax shift-invariant — b does not affect predicted distributions.
    Regularization keeps it at zero. W carries all expressive power.
    """

    def __init__(self, hidden_dim: int, layer_indices: List[int]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_indices = layer_indices
        L = len(layer_indices)

        # Single 3D parameters — no stack() needed on every forward pass
        self.W = nn.Parameter(
            torch.eye(hidden_dim).unsqueeze(0).expand(L, -1, -1).clone()
        )  # (L, D, D)
        self.b = nn.Parameter(torch.zeros(L, hidden_dim))  # (L, D)

    def forward(self, H: torch.Tensor, unembed_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H:              Stacked hidden states for all layers. Shape: (L, B, S, D)
            unembed_weight: model.lm_head.weight.                 Shape: (V, D)
        Returns:
            logits: Shape (L, B, S, V)
        """
        L, B, S, D = H.shape
        H_flat = H.reshape(L, B * S, D)

        H_transformed = torch.bmm(H_flat, self.W)              # (L, B*S, D) — one bmm for all layers
        logits = H_transformed @ unembed_weight.T               # (L, B*S, V) — shared unembed
        bias = torch.bmm(H_flat, self.b.unsqueeze(-1))          # (L, B*S, 1)

        return (logits + bias).view(L, B, S, -1)                # (L, B, S, V)

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
    def load_layers(cls, output_dir: str, layer_indices: List[int], device: str = "cpu") -> "TunedLens":
        """Load per-layer files written by save_layers() and reconstruct a TunedLens."""
        first = torch.load(
            os.path.join(output_dir, f"layer_{layer_indices[0]:02d}.pt"), map_location=device
        )
        hidden_dim = first["hidden_dim"]
        lens = cls(hidden_dim, layer_indices)
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Reuse the already-loaded first checkpoint instead of re-reading the file
                ckpt = first if i == 0 else torch.load(
                    os.path.join(output_dir, f"layer_{layer_idx:02d}.pt"), map_location=device
                )
                lens.W.data[i].copy_(ckpt["W"])
                lens.b.data[i].copy_(ckpt["b"])
        return lens
