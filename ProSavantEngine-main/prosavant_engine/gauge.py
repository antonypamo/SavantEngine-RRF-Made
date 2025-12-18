from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SavantRRF_Gauge(nn.Module):
    """
    CNN 1D gauge para los 12 nodos Φ orbitantes.

    Nota importante de forma:
      - Fue entrenada con entradas de forma [batch, 1, 160]
        → después de conv1/2/3 queda [batch, 256, 160]
        → flatten = 256 * 160 = 40960 = in_features de fc1.
      - Si cambias la longitud de la secuencia, tendrás que
        ajustar 40960 a mano o introducir un adaptador.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        # hidden_dim se mantiene por compatibilidad de firma, no se usa directamente.
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.25)

        # ORIGINAL: 256 * 160 = 40960 → entrenado así en tu Colab original.
        # FIX: Change 40960 to 256 to match the actual flattened size from [batch, 256, 1] output
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_dim] (from IcosahedralRRF's x) or [batch, input_dim, seq_len]
        """
        # If input x is 2D (e.g., [batch, features]), reshape to [batch, features, 1]
        if x.dim() == 2:
            # Assuming features = input_dim (e.g., 384)
            x = x.unsqueeze(-1)  # Adds a sequence length dimension: [batch, input_dim, 1]

        # Assert that x is now 3D with the correct input_dim for conv1
        # The input_dim of conv1d is the channels, and x.shape[1] is channels
        assert x.dim() == 3, f"Expected 3D input to SavantRRF_Gauge, got {x.shape}"
        assert x.shape[1] == self.conv1.in_channels,             f"SavantRRF_Gauge: input channels mismatch. Expected {self.conv1.in_channels}, got {x.shape[1]} for input shape {x.shape}"

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


__all__ = ["SavantRRF_Gauge"]
