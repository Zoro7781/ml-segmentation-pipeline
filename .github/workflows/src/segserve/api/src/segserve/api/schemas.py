from __future__ import annotations

from pydantic import BaseModel, Field


class BoxPrompt(BaseModel):
    x1: float = Field(..., description="Left x pixel coordinate")
    y1: float = Field(..., description="Top y pixel coordinate")
    x2: float = Field(..., description="Right x pixel coordinate")
    y2: float = Field(..., description="Bottom y pixel coordinate")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Mask threshold (0..1)")


class VideoPrompt(BoxPrompt):
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="EMA smoothing alpha (0..1)")
    max_frames: int = Field(60, ge=1, le=300, description="Maximum frames to process")
