# ML Segmentation Pipeline (SAM + short video smoothing)

This repo is a production-style segmentation service:
- SAM segmentation (box prompt)
- short-video temporal smoothing (EMA)
- FastAPI REST API
- Docker-ready

## Quickstart (local)

### 1) Download SAM checkpoint (do NOT commit it)
```bash
bash scripts/download_sam_checkpoint.sh
