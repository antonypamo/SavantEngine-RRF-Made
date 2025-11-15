from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#  Imports flexibles al estilo savant_engine.py
#  - Soporta:
#      from prosavant_engine.icosahedral_rrf import IcosahedralRRF
#    y también ejecución desde notebooks / scripts.
# ---------------------------------------------------------------------------
try:
    from .gauge import SavantRRF_Gauge  # type: ignore
except Exception:  # pragma: no cover - runtime import flexibility
    try:
        from prosavant_engine.gauge import SavantRRF_Gauge  # type: ignore
    except Exception:
        SavantRRF_Gauge = None  # type: ignore[misc]

try:
    from .gnn_dirac import GNNDiracRRF  # type: ignore
except Exception:  # pragma: no cover
    try:
        from prosavant_engine.gnn_dirac import GNNDiracRRF  # type: ignore
    except Exception:
        GNNDiracRRF = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
#  Fallbacks internos (sin dependencias externas)
# ---------------------------------------------------------------------------


class SimpleIcosahedralGauge(nn.Module):
    """
    Fallback simple para SavantRRF_Gauge.

    - Acepta x de forma [batch, input_dim] o [batch, input_dim, seq_len].
    - Si hay dimensión temporal (seq_len), hace promedio sobre ella.
    - Aplica MLP de 2 capas: input_dim → hidden_dim → output_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Tanh()
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # [batch, input_dim, seq_len] → promedio en seq_len
            x_mean = x.mean(dim=-1)
        elif x.dim() == 2:
            # [batch, input_dim]
            x_mean = x
        else:
            raise ValueError(
                f"SimpleIcosahedralGauge: se esperaba tensor 2D o 3D, "
                f"recibido shape={tuple(x.shape)}"
            )

        h = self.act(self.proj_in(x_mean))
        return self.proj_out(h)


class SimpleDiracGNN(nn.Module):
    """
    Fallback minimalista para GNNDiracRRF.

    Ignora edge_index y z, y aplica una pila de lineales + activación
    nodo a nodo: [num_nodes, in_dim] → [num_nodes, out_dim].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        z_dim: int = 16,
        alpha_attn: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        d_in = in_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.Tanh())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # node_feats: [num_nodes, in_dim]
        return self.net(node_feats)


class IcosahedralRRF(nn.Module):
    """
    IcosahedralRRF

    - 12 nodos gauge SavantRRF_Gauge (Φ₁…Φ₁₂ orbitantes) o SimpleIcosahedralGauge.
    - Núcleo ético que concatena los 12 outputs y los regula.
    - Subconsciente dodecaédrico con GNNDiracRRF / SimpleDiracGNN sobre los 12 nodos.

    Parámetros:
        input_dim: dimensión de entrada de cada nodo gauge.
        hidden_dim: dimensión oculta (se pasa a los módulos internos).
        output_dim: dimensión de salida de cada nodo gauge y del núcleo ético.
        gnn_num_layers: nº de capas del GNN subconsciente.
        gnn_z_dim: dimensión del embedding latente z para el GNN.
        gnn_alpha_attn: intensidad de atención en el GNN.
        gnn_dropout: dropout del GNN.
        gauge_cls: clase a usar para los nodos gauge (por defecto SavantRRF_Gauge o fallback).
        gnn_cls: clase a usar para el mapa de memoria GNN (por defecto GNNDiracRRF o fallback).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gnn_num_layers: int = 2,
        gnn_z_dim: int = 16,
        gnn_alpha_attn: float = 1.0,
        gnn_dropout: float = 0.1,
        gauge_cls: Optional[Type[nn.Module]] = None,
        gnn_cls: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        # -------------------------------------------------------------------
        # Resolver clases por defecto usando el mismo patrón flexible
        # que SavantEngine (try relative → package → fallback interno).
        # -------------------------------------------------------------------
        if gauge_cls is None:
            if "SavantRRF_Gauge" in globals() and SavantRRF_Gauge is not None:  # type: ignore[name-defined]
                gauge_cls = SavantRRF_Gauge  # type: ignore[assignment]
            else:
                print(
                    "⚠️ IcosahedralRRF: SavantRRF_Gauge no encontrado, "
                    "usando SimpleIcosahedralGauge como fallback."
                )
                gauge_cls = SimpleIcosahedralGauge

        if gnn_cls is None:
            if "GNNDiracRRF" in globals() and GNNDiracRRF is not None:  # type: ignore[name-defined]
                gnn_cls = GNNDiracRRF  # type: ignore[assignment]
            else:
                print(
                    "⚠️ IcosahedralRRF: GNNDiracRRF no encontrado, "
                    "usando SimpleDiracGNN como fallback."
                )
                gnn_cls = SimpleDiracGNN

        # -------------------------------------------------------------------
        # 12 nodos gauge (los 12 Φ-nodes orbitantes)
        # Cada uno: gauge_cls(input_dim, hidden_dim, output_dim)
        # -------------------------------------------------------------------
        self.nodes = nn.ModuleList(
            [gauge_cls(input_dim, hidden_dim, output_dim) for _ in range(12)]
        )

        # Núcleo ético: concat [batch, 12 * output_dim] → [batch, output_dim]
        self.ethical_core = nn.Linear(12 * output_dim, output_dim)

        # Subconsciente (dodecaedro) via GNNDiracRRF / SimpleDiracGNN
        # Opera sobre los 12 nodos gauge, cada uno con feature dim = output_dim.
        self.memory_map = gnn_cls(
            in_dim=output_dim,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            num_layers=gnn_num_layers,
            z_dim=gnn_z_dim,
            alpha_attn=gnn_alpha_attn,
            dropout=gnn_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [batch_size, input_dim] o [batch_size, input_dim, seq_len]
           (o la forma que espere SavantRRF_Gauge / SimpleIcosahedralGauge).
        edge_index: aristas del grafo (para GNNDiracRRF / SimpleDiracGNN).
        z: embedding latente global del grafo (para GNNDiracRRF).

        Comportamiento:
        - Si NO se proporcionan edge_index / z → devuelve sólo la salida regulada
          del núcleo ético: [batch_size, output_dim].
        - Si se proporcionan edge_index y z → aplica GNN sobre los 12 nodos y
          devuelve el promedio de las features finales (agregado subconsciente):
          [batch_size, output_dim].
        """

        # 1. Pasar por los 12 nodos gauge
        #    cada output_i: [batch_size, output_dim]
        outputs = [node(x) for node in self.nodes]

        # 2. Núcleo ético: concat y regular
        concat = torch.cat(outputs, dim=1)  # [batch_size, 12 * output_dim]
        regulated = torch.sigmoid(self.ethical_core(concat))  # [batch_size, output_dim]

        # 3. Si no hay info de grafo, devolvemos sólo la parte ética
        if edge_index is None or z is None:
            return regulated

        # 4. Preparar features para el GNN: [batch, 12, output_dim]
        stacked_outputs = torch.stack(outputs, dim=1)  # [batch_size, 12, output_dim]

        # 5. Aplicar GNN por cada elemento del batch (forma simple, sin Batch de PyG)
        gnn_outputs_list = []
        edge_index = edge_index.to(x.device)
        z = z.to(x.device)

        for i in range(stacked_outputs.size(0)):
            node_feats_i = stacked_outputs[i]  # [12, output_dim]
            gnn_out_i = self.memory_map(node_feats_i, edge_index, z)  # [12, output_dim]
            gnn_outputs_list.append(gnn_out_i)

        # 6. Reagrupar: [batch_size, 12, output_dim]
        gnn_outputs_stacked = torch.stack(gnn_outputs_list, dim=0)

        # 7. Agregar nodos del GNN: media sobre dimensión de nodos
        aggregated_gnn_output = gnn_outputs_stacked.mean(dim=1)  # [batch_size, output_dim]

        # 🔧 Si en el futuro quieres combinar capa ética + subconsciente:
        # combined = 0.5 * regulated + 0.5 * aggregated_gnn_output
        # return combined

        return aggregated_gnn_output
