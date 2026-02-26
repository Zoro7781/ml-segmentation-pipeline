#!/usr/bin/env bash
set -e

mkdir -p models

echo "Downloading SAM ViT-B checkpoint..."
curl -L -o models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo "Saved to models/sam_vit_b_01ec64.pth"
echo "DO NOT COMMIT THIS FILE TO GIT."
