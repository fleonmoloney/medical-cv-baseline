"""Describe el dataset descargado: tamaños por partición y balance de clases.

Uso:
    python -m medcv.dataset_info --dataset pneumoniamnist
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import load_medmnist
from .utils import save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="pneumoniamnist")
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results/dataset_info.json")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    splits, meta = load_medmnist(a.dataset, a.size, a.data_root, download=not a.no_download)
    info = {"meta": meta, "splits": {}}
    for name, (x, y) in splits.items():
        y = y.reshape(-1)
        counts = np.bincount(y, minlength=meta["n_classes"])
        info["splits"][name] = {
            "n": int(len(y)),
            "image_shape": list(x.shape[1:]),
            "class_counts": {meta["labels"][str(i)]: int(c) for i, c in enumerate(counts)},
            "class_fraction": {meta["labels"][str(i)]: round(float(c / len(y)), 4) for i, c in enumerate(counts)},
        }
        print(f"{name:>5}: n={len(y):>6}  " + "  ".join(f"{meta['labels'][str(i)]}={c}" for i, c in enumerate(counts)))
    save_json(info, a.out)
    print(f"Guardado {a.out}")


if __name__ == "__main__":
    main()
