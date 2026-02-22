#!/usr/bin/env bash

set -euo pipefail

uv run python scripts/train.py \
  --model-config configs/models/whisper-large-v3-lora.yaml \
  --training-config configs/training/whisper-large-v3-lora-elise.yaml \
  --data-config configs/data/elise.yaml \
  --dataset MrDragonFox/Elise
