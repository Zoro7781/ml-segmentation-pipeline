from __future__ import annotations

import numpy as np


def ema_smooth_prob_maps(prob_maps: list[np.ndarray], alpha: float = 0.7) -> list[np.ndarray]:
    """
    Exponential moving average smoothing over a sequence of probability maps.

    prob_maps: list of (H, W) float arrays in [0,1]
    alpha: weight for current frame (0..1). Higher = less smoothing.
    """
    if not prob_maps:
        return []

    smoothed: list[np.ndarray] = []
    prev = prob_maps[0].astype(np.float32)
    smoothed.append(prev)

    for t in range(1, len(prob_maps)):
        curr = prob_maps[t].astype(np.float32)
        prev = alpha * curr + (1.0 - alpha) * prev
        smoothed.append(prev)

    return smoothed


def prob_to_mask(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert prob map (H,W) to uint8 binary mask (H,W) with values 0/255."""
    mask = (prob >= threshold).astype(np.uint8) * 255
    return mask
