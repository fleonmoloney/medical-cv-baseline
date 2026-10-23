#!/usr/bin/env bash
# Corre 5 semillas por modelo sobre PneumoniaMNIST y agrega los resultados.
# Uso: bash scripts/run_all.sh [dataset]
set -euo pipefail

DATASET="${1:-pneumoniamnist}"
EPOCHS="${EPOCHS:-30}"

python -m medcv.dataset_info --dataset "$DATASET"

for ARCH in simple_cnn resnet18; do
  for SEED in 0 1 2 3 4; do
    python -m medcv.train --dataset "$DATASET" --arch "$ARCH" --seed "$SEED" --epochs "$EPOCHS"
  done
done

python -m medcv.aggregate --runs-dir runs --out results/summary.md
