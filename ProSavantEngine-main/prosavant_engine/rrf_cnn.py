from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import ResNetConfig, ResNetModel
from huggingface_hub import hf_hub_download


DEFAULT_RRF_CNN_REPO = "antonypamo/RRF_CNN"
DEFAULT_RRF_CNN_FILE = "rrf_advanced_cnn_model.pt"


@dataclass
class RRFCNNInfo:
    """Pequeño contenedor con el modelo y su config."""
    model: nn.Module
    config: ResNetConfig
    repo_id: str
    weight_file: str


class RRFCNNBackbone(nn.Module):
    """
    Wrapper ligero sobre un ResNet 2D entrenado para el marco RRF.

    - Carga pesos desde Hugging Face (antonypamo/RRF_CNN).
    - Expone:
        * forward(x, pool=True) → [B, C, H, W] o [B, C] si pool=True.
        * C = 2048 según tu config actual (stage4).
    """

    def __init__(
        self,
        config: ResNetConfig,
        state_dict: Dict[str, Any],
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = ResNetModel(config)
        self.backbone.load_state_dict(state_dict)
        self.do_pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] (imagen o mapa 2D).
        return:
          - si self.do_pool=True → [B, 2048]
          - si self.do_pool=False → [B, 2048, 7, 7] (feature map de stage4)
        """
        outputs = self.backbone(pixel_values=x)
        feat = outputs.last_hidden_state  # [B, C, H, W]

        if self.do_pool:
            # Global Average Pooling sobre H, W
            feat = feat.mean(dim=(2, 3))  # [B, C]

        return feat


def load_rrf_cnn_backbone(
    repo_id: str = DEFAULT_RRF_CNN_REPO,
    weight_file: str = DEFAULT_RRF_CNN_FILE,
    pool: bool = True,
    map_location: str | torch.device = "cpu",
) -> RRFCNNInfo:
    """
    Descarga config + pesos desde Hugging Face y devuelve un backbone listo.

    Uso típico:
        info = load_rrf_cnn_backbone()
        model = info.model
        config = info.config
    """
    # 1) Cargar config desde el propio repo (config.json ya tiene "model_type": "resnet")
    config = ResNetConfig.from_pretrained(repo_id)

    # 2) Descargar el archivo de pesos específico (state_dict)
    weights_path = hf_hub_download(repo_id, weight_file)
    state_dict = torch.load(weights_path, map_location=map_location)

    model = RRFCNNBackbone(config=config, state_dict=state_dict, pool=pool)
    return RRFCNNInfo(model=model, config=config, repo_id=repo_id, weight_file=weight_file)
