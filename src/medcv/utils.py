"""Utilidades de reproducibilidad y registro del entorno."""

from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija todas las semillas y pide algoritmos deterministas.

    `warn_only=True`: si alguna operación no tiene versión determinista, avisa
    en lugar de fallar. En CPU el resultado es reproducible bit a bit.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_info() -> dict:
    """Versiones y plataforma, para guardarlas junto a cada resultado."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import medmnist

        info["medmnist"] = medmnist.__version__
    except Exception:  # pragma: no cover
        pass
    return info


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"No serializable: {type(o)}")
