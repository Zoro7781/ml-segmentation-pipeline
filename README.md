<p align="center">
  <img src="docs/assets/banner.svg" alt="SON SegServe" width="100%"/>
</p>

<p align="center">
  <img src="docs/assets/badges.svg" alt="badges" width="680"/>
</p>

<br/>

**SON SegServe** is a service that cuts out objects from images automatically. You point at something — with a box, a click, or a word — and it gives you back a precise mask of that object. Fast, runs in Docker, and works over a simple REST API.

---

## How It Works

<p align="center">
  <img src="docs/assets/how_it_works.svg" alt="How it works" width="100%"/>
</p>

1. **Give it an image** — any JPEG or PNG
2. **Tell it what you want** — draw a box, click a point, or type a label
3. **The model figures it out** — SAM2 finds the exact pixels that belong to your object
4. **You get a mask back** — with a confidence score and the exact pixel area

---

## 3 Ways to Describe What You Want

<p align="center">
  <img src="docs/assets/prompt_types.svg" alt="Prompt types" width="100%"/>
</p>

| Prompt | How to use it | Example |
|--------|--------------|---------|
| **Box** | Draw a rectangle around the object | `"box_xyxy": [10, 20, 200, 220]` |
| **Point** | Click on the object | `"point": [120, 88, "fg"]` |
| **Text** | Describe it in words | `"query": "the red car on the left"` |

---

## What the Output Looks Like

<p align="center">
  <img src="docs/assets/output_masks.svg" alt="Output masks" width="100%"/>
</p>

The service can find **multiple objects in one shot**. Each mask comes with:
- A **confidence score** (predicted IoU — how sure the model is)
- The **pixel count** (area in pixels)
- The mask encoded as **RLE** (a compact format, not a giant array)

---

## Under the Hood — The Segmentation Pipeline

<p align="center">
  <img src="docs/assets/segmentation_pipeline.svg" alt="Segmentation pipeline" width="100%"/>
</p>

Here is what happens inside every request:

1. **Pre-process** — your image gets resized and normalized into a tensor
2. **Image Encoder** — a Vision Transformer (ViT) reads the whole image and extracts features
3. **Prompt Encoder** — your box, point, or text gets turned into vectors the model can use
4. **Mask Decoder** — combines the image features and prompt vectors to output a binary mask
5. **Post-process** — compress the mask (RLE), rank by confidence, log metrics, return JSON

> Typical speed on CPU: ~44ms total · ~4ms prep · ~32ms model · ~8ms cleanup

---

## Quickstart

### With Docker (easiest)

```bash
# Build
docker build -f docker/Dockerfile -t son-segserve:dev .

# Run
docker run --rm -p 8080:8080 -e SON_MODEL_ID=sam2_tiny son-segserve:dev

# Check it is running
curl http://localhost:8080/healthz
# {"status": "ok"}
```

### With pip

```bash
pip install -e ".[dev]"

# Ping the service
son-segserve ping --api http://localhost:8080

# Segment something
son-segserve segment \
  --api http://localhost:8080 \
  --image ./photo.jpg \
  --prompt "box:10,10,200,200" \
  --out ./result.json
```

The model downloads automatically on first run to `~/.cache/son-segserve/models/`.
Change the location with `SON_CACHE_DIR=/your/path`.

---

## API Endpoints

| Method | Path | What it does |
|--------|------|-------------|
| `GET` | `/healthz` | Is the service alive? |
| `GET` | `/readyz` | Is the model loaded and ready? |
| `GET` | `/metrics` | Prometheus metrics (latency, throughput) |
| `POST` | `/v1/segment` | Segment an image with a prompt |
| `POST` | `/v1/segment/video` | Segment a short video clip |

### Example request

```bash
curl -X POST http://localhost:8080/v1/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": {"content_type": "image/jpeg", "base64": "..."},
    "prompt": {"type": "box", "box_xyxy": [10, 20, 200, 220]},
    "options": {"max_masks": 3}
  }'
```

### Example response

```json
{
  "model_id": "sam2_tiny",
  "masks": [
    {
      "mask_id": "0",
      "encoding": "rle",
      "predicted_iou": 0.87,
      "area_px": 12345
    }
  ],
  "timing_ms": {
    "preprocess": 4.1,
    "inference": 31.7,
    "postprocess": 7.9,
    "total": 43.7
  }
}
```

---

## Supported Models

| Model | Size | Best for |
|-------|------|---------|
| `sam2_tiny` | ~40 MB | Good balance — the default |
| `sam2_base` | ~80 MB | Better accuracy, a bit slower |
| `mobilesam` | ~9 MB | Fastest, great on CPU or edge devices |

> Models are **never stored in this repo**. They download automatically from release assets and are verified by SHA256 checksum before loading.

---

## Monitoring

The service exposes a `/metrics` endpoint that Prometheus can scrape. A ready-made Grafana dashboard is in `monitoring/grafana/dashboard.json`.

What gets tracked automatically:

- Request count and HTTP status codes
- End-to-end latency (p50 and p95)
- Time spent in the model vs pre and post-processing
- Memory usage
- Number of masks returned per request
- Predicted IoU distribution (useful for catching model drift)

---

## Project Structure

```
son-segserve/
├── src/son_segserve/     ← all the application code
│   ├── api/              ← FastAPI routes and schemas
│   ├── cli/              ← command-line client
│   ├── model/            ← model loading, inference, prompts
│   └── metrics/          ← Prometheus instrumentation
├── docker/               ← Dockerfile + entrypoint
├── configs/              ← service config + model registry (with checksums)
├── data/metadata/        ← dataset cards and model cards (no raw images)
├── scripts/datasets/     ← download scripts for COCO, VOC, OpenImages, DAVIS, YouTube-VOS
├── tests/                ← unit, integration, regression, synthetic
├── monitoring/           ← Grafana dashboard + alert rules
└── docs/                 ← full documentation (served via MkDocs)
```

**One rule:** nothing big goes in git. No model weights. No images. No datasets.
Scripts download them; checksums verify them.

---

## Datasets

| Dataset | Licence | In demo by default? |
|---------|---------|-------------------|
| Open Images v5 | CC BY 4.0 (annotations) | Yes — verify image licence per use |
| COCO 2017 | CC BY 4.0 | Use subsets only |
| DAVIS 2017 | Research use | Evaluation only |
| YouTube-VOS | Non-commercial only | Off by default |
| PASCAL VOC 2012 | Flickr terms apply | Off by default |

---

## Development Setup

```bash
# Set up
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Check everything passes
pre-commit run -a
pytest
```

---

## Contributing

One feature or fix per pull request. Add tests for anything you change. Run `pre-commit run -a` before you push. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

---

## Security

Found a vulnerability? Please **do not open a public issue**.
Email `security@example.org` and we will respond within 72 hours.
See [`SECURITY.md`](SECURITY.md).

---

## Licence

Code is **Apache-2.0** — see [`LICENSE`](LICENSE).
Dataset and model licences are in `data/metadata/` — check them before any commercial use.

---

<p align="center">
  <sub>No weights in git &nbsp;·&nbsp; No datasets committed &nbsp;·&nbsp; Models verified by SHA256 at runtime</sub>
</p>
