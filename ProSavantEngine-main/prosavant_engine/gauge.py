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

        # 256 * 160 = 40960 → entrenado así en tu Colab original.
        self.fc1 = nn.Linear(40960, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_dim, seq_len]  (en tu training: input_dim=1, seq_len=160)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


__all__ = ["SavantRRF_Gauge"]
