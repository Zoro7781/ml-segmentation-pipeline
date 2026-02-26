from __future__ import annotations

import numpy as np
from segment_anything import SamPredictor


def segment_with_box(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    box_xyxy: list[float],
) -> np.ndarray:
    """
    Segment a single object using a bounding box prompt.

    image_rgb: (H, W, 3) uint8 RGB
    box_xyxy: [x1, y1, x2, y2] in pixel coords

    Returns: (H, W) float32 probability map in [0,1]
    """
    predictor.set_image(image_rgb)

    box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, logits = predictor.predict(
        box=box[None, :],
        multimask_output=False,
        return_logits=True,
    )

    logit = logits[0].astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob
