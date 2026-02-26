from __future__ import annotations

import io
import zipfile
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from segserve.core.smoothing import ema_smooth_prob_maps, prob_to_mask
from segserve.model.predictor import segment_with_box
from segserve.model.sam_loader import SamConfig, load_sam_predictor

app = FastAPI(title="SAM Segmentation Service", version="0.1.0")

# --- Model load (simple startup load) ---
SAM_CFG = SamConfig(
    model_type="vit_b",
    checkpoint_path="models/sam_vit_b_01ec64.pth",
    device="cpu",
)

try:
    predictor = load_sam_predictor(SAM_CFG)
except FileNotFoundError:
    predictor = None


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    return {"model_loaded": predictor is not None}


def _read_image_rgb(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


@app.post("/v1/segment")
async def segment_image(
    image: Annotated[UploadFile, File(...)],
    x1: Annotated[float, Form(...)],
    y1: Annotated[float, Form(...)],
    x2: Annotated[float, Form(...)],
    y2: Annotated[float, Form(...)],
    threshold: Annotated[float, Form(0.5)],
):
    """
    Segment a single image using a bounding box prompt.
    Returns a PNG mask (0/255).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Missing SAM checkpoint file.")

    img_bytes = await image.read()
    image_rgb = _read_image_rgb(img_bytes)

    prob = segment_with_box(predictor, image_rgb, [x1, y1, x2, y2])
    mask = prob_to_mask(prob, threshold=threshold)

    out = io.BytesIO()
    Image.fromarray(mask).save(out, format="PNG")
    return Response(content=out.getvalue(), media_type="image/png")


@app.post("/v1/segment/video")
async def segment_video(
    video: Annotated[UploadFile, File(...)],
    x1: Annotated[float, Form(...)],
    y1: Annotated[float, Form(...)],
    x2: Annotated[float, Form(...)],
    y2: Annotated[float, Form(...)],
    alpha: Annotated[float, Form(0.7)],
    threshold: Annotated[float, Form(0.5)],
    max_frames: Annotated[int, Form(60)],
):
    """
    Short video segmentation:
    - runs SAM per frame (box prompt)
    - EMA smooths probability maps
    - returns a ZIP of PNG masks: mask_00000.png ...
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Missing SAM checkpoint file.")

    vid_bytes = await video.read()

    tmp_path = "/tmp/input_video"
    with open(tmp_path, "wb") as f:
        f.write(vid_bytes)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not read video file.")

    prob_maps: list[np.ndarray] = []
    count = 0
    while count < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prob = segment_with_box(predictor, frame_rgb, [x1, y1, x2, y2])
        prob_maps.append(prob)
        count += 1

    cap.release()

    if not prob_maps:
        raise HTTPException(status_code=400, detail="No frames decoded from video.")

    smoothed = ema_smooth_prob_maps(prob_maps, alpha=alpha)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for i, prob in enumerate(smoothed):
            mask = prob_to_mask(prob, threshold=threshold)
            out = io.BytesIO()
            Image.fromarray(mask).save(out, format="PNG")
            z.writestr(f"mask_{i:05d}.png", out.getvalue())

    return Response(content=buf.getvalue(), media_type="application/zip")
