"""Carga de datos y Dataset de tensores.

Los datos reales se descargan con el paquete `medmnist` (MedMNIST v2/v3).
`ArrayDataset` trabaja con arreglos numpy, de modo que el resto del pipeline
se puede probar con datos sintéticos sin descargar nada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

Arrays = Tuple[np.ndarray, np.ndarray]


class ArrayDataset(Dataset):
    """Imágenes uint8 (N,H,W) o (N,H,W,C) y etiquetas (N,) o (N,1).

    Normalización fija (x-0.5)/0.5: no se calculan estadísticas sobre ningún
    conjunto, así que no hay fuga de información entre particiones.
    Aumento de datos (solo en entrenamiento): traslación aleatoria de hasta
    `pad` píxeles. No se usan volteos: en radiografías el lado importa.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = False, pad: int = 2):
        x = torch.from_numpy(np.ascontiguousarray(images))
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        self.x = x.permute(0, 3, 1, 2).float().div(255.0).sub(0.5).div(0.5)
        self.y = torch.from_numpy(np.asarray(labels).reshape(-1).astype(np.int64))
        self.train = train
        self.pad = pad

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.x[i]
        if self.train and self.pad > 0:
            _, h, w = x.shape
            p = self.pad
            x = F.pad(x.unsqueeze(0), (p, p, p, p), mode="reflect").squeeze(0)
            di = int(torch.randint(0, 2 * p + 1, (1,)))
            dj = int(torch.randint(0, 2 * p + 1, (1,)))
            x = x[:, di : di + h, dj : dj + w]
        return x, self.y[i]


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    # num_workers=0: el aumento usa el RNG global; así el orden es reproducible.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)


def load_medmnist(
    name: str = "pneumoniamnist", size: int = 28, root: str = "data", download: bool = True
) -> Tuple[Dict[str, Arrays], dict]:
    """Descarga (si hace falta) y devuelve {'train','val','test'} -> (imgs, labels) e info."""
    import medmnist
    from medmnist import INFO

    if name not in INFO:
        raise ValueError(f"Dataset desconocido: {name}. Opciones: {sorted(INFO)}")
    info = INFO[name]
    if info["task"] not in ("binary-class", "multi-class"):
        raise ValueError(f"Tarea no soportada en este repo: {info['task']}")
    cls = getattr(medmnist, info["python_class"])
    Path(root).mkdir(parents=True, exist_ok=True)  # medmnist exige que la carpeta exista
    splits = {}
    for split in ("train", "val", "test"):
        d = cls(split=split, download=download, root=root, size=size)
        splits[split] = (d.imgs, d.labels)
    meta = {
        "name": name,
        "task": info["task"],
        "n_channels": info["n_channels"],
        "n_classes": len(info["label"]),
        "labels": info["label"],
        "size": size,
    }
    return splits, meta


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia (calculados solo con train)."""
    y = np.asarray(labels).reshape(-1)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
