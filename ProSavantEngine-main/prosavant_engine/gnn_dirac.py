from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiracGraphConv(nn.Module):
    """
    Capa de convolución de grafo inspirada en Dirac:

      - Usa un embedding latente z por nodo.
      - Calcula correlación coseno entre z_i y z_j para cada arista.
      - Usa eso como atención multiplicativa sobre mensajes x_j → x_i.
    """

    def __init__(self, in_dim: int, out_dim: int, alpha: float = 1.0, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.bias_edge = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def cosine_corr(z_i: torch.Tensor, z_j: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        num = (z_i * z_j).sum(dim=-1)
        den = torch.clamp(z_i.norm(dim=-1) * z_j.norm(dim=-1), min=eps)
        return num / den

    def forward(
        self,
        x: torch.Tensor,          # [num_nodes, in_dim]
        edge_index: torch.Tensor, # [2, num_edges]
        z: torch.Tensor,          # [num_nodes, z_dim]
    ) -> torch.Tensor:
        row, col = edge_index
        # correlación coseno sobre z
        corr = self.cosine_corr(z[row], z[col])                 # [num_edges]
        logits = self.alpha * corr + self.bias_edge            # [num_edges]

        exp_logits = torch.exp(logits - logits.max())
        # denominador: suma sobre vecinos j→i
        denom = torch.zeros_like(x[:, 0]).index_add_(0, row, exp_logits)
        attn = exp_logits / (denom[row] + 1e-9)                # [num_edges]

        # mensajes ponderados
        msgs = attn.unsqueeze(-1) * x[col]                     # [num_edges, in_dim]

        out = torch.zeros_like(x).index_add_(0, row, msgs)     # [num_nodes, in_dim]
        return self.lin(out)                                   # [num_nodes, out_dim]


class GNNDiracRRF(nn.Module):
    """
    Pequeño GNN apilado de DiracGraphConv con activaciones GELU + dropout.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        z_dim: int = 16,
        alpha_attn: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim

        layers = []
        layers.append(DiracGraphConv(in_dim, hidden_dim, alpha=alpha_attn))
        for _ in range(num_layers - 2):
            layers.append(DiracGraphConv(hidden_dim, hidden_dim, alpha=alpha_attn))
        layers.append(DiracGraphConv(hidden_dim, out_dim, alpha=alpha_attn))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # [num_nodes, in_dim]
        edge_index: torch.Tensor, # [2, num_edges]
        z: torch.Tensor,          # [num_nodes, z_dim]
    ) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, z)
            if i < len(self.layers) - 1:
                h = F.gelu(h)
                h = self.dropout(h)
        return h


__all__ = ["DiracGraphConv", "GNNDiracRRF"]
