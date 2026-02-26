from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from segment_anything import SamPredictor, sam_model_registry


@dataclass(frozen=True)
class SamConfig:
    """
    model_type:
      - vit_b (smallest / easiest)
      - vit_l
      - vit_h (largest)
    checkpoint_path: where the .pth file exists locally (NOT in git).
    device: "cpu" or "cuda"
    """
    model_type: Literal["vit_b", "vit_l", "vit_h"] = "vit_b"
    checkpoint_path: str = "models/sam_vit_b_01ec64.pth"
    device: str = "cpu"


def load_sam_predictor(cfg: SamConfig) -> SamPredictor:
    """
    Loads SAM predictor from a local checkpoint.
    The checkpoint should be downloaded locally (or inside Docker) and NOT committed to GitHub.
    """
    ckpt = Path(cfg.checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {ckpt}\n"
            f"Download it into {ckpt.parent}/ (do NOT commit to git)."
        )

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    sam = sam_model_registry[cfg.model_type](checkpoint=str(ckpt))
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor
