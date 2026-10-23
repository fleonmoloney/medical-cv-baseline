"""Entrenamiento reproducible.

Uso:
    python -m medcv.train --dataset pneumoniamnist --arch resnet18 --seed 0

`fit()` recibe arreglos numpy, por lo que se puede probar sin descargar datos.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data import ArrayDataset, class_weights, load_medmnist, make_loader
from .evaluate import auc, compute_metrics, plot_calibration, plot_roc, youden_threshold
from .model import build_model
from .utils import environment_info, get_device, save_json, set_seed

Arrays = Tuple[np.ndarray, np.ndarray]


@dataclass
class Config:
    dataset: str = "pneumoniamnist"
    arch: str = "resnet18"
    size: int = 28
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    pad: int = 2
    class_weighted: bool = True
    patience: int = 8
    n_boot: int = 1000
    device: Optional[str] = None
    out_dir: Optional[str] = None


@torch.no_grad()
def predict_proba(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    """Devuelve (y_true, probs, loss_promedio). probs: (N,K)."""
    model.eval()
    ys, ps, losses, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        losses += float(ce(logits, yb))
        n += len(yb)
        ps.append(torch.softmax(logits, dim=1).cpu().numpy())
        ys.append(yb.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), losses / max(n, 1)


def _collapse(probs: np.ndarray, binary: bool) -> np.ndarray:
    return probs[:, 1] if binary else probs


def fit(cfg: Config, splits: Dict[str, Arrays], meta: dict) -> dict:
    """Entrena, elige umbral en validación y evalúa en test una sola vez."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    n_classes, n_ch = meta["n_classes"], meta["n_channels"]
    binary = meta["task"] == "binary-class"

    ds_tr = ArrayDataset(*splits["train"], train=True, pad=cfg.pad)
    ds_va = ArrayDataset(*splits["val"])
    ds_te = ArrayDataset(*splits["test"])
    dl_tr = make_loader(ds_tr, cfg.batch_size, shuffle=True, seed=cfg.seed)
    dl_va = make_loader(ds_va, 512, shuffle=False)
    dl_te = make_loader(ds_te, 512, shuffle=False)

    model = build_model(cfg.arch, n_ch, n_classes).to(device)
    weight = class_weights(splits["train"][1], n_classes).to(device) if cfg.class_weighted else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best = {"auc": -1.0, "loss": float("inf"), "epoch": -1, "state": None}
    history, bad_epochs, t0 = [], 0, time.time()

    for epoch in range(cfg.epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * len(yb)
            seen += len(yb)
        sched.step()

        yv, pv, vloss = predict_proba(model, dl_va, device)
        vauc = auc(yv, _collapse(pv, binary))
        history.append({"epoch": epoch, "train_loss": run_loss / seen, "val_loss": vloss, "val_auc": vauc})

        improved = vauc > best["auc"] + 1e-6 or (abs(vauc - best["auc"]) <= 1e-6 and vloss < best["loss"])
        if improved:
            best.update(auc=vauc, loss=vloss, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    train_seconds = time.time() - t0
    model.load_state_dict(best["state"])

    # Umbral: solo con validación.
    yv, pv, _ = predict_proba(model, dl_va, device)
    pv_c = _collapse(pv, binary)
    threshold = youden_threshold(yv, pv_c) if binary else None

    # Test: una única evaluación, con el mejor checkpoint de validación.
    yt, pt, _ = predict_proba(model, dl_te, device)
    pt_c = _collapse(pt, binary)

    result = {
        "config": asdict(cfg),
        "dataset_meta": meta,
        "environment": environment_info() | {"device": str(device)},
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_seconds": train_seconds,
        "val": compute_metrics(yv, pv_c, threshold, n_boot=cfg.n_boot, seed=cfg.seed),
        "test": compute_metrics(yt, pt_c, threshold, n_boot=cfg.n_boot, seed=cfg.seed),
        "history": history,
    }

    if cfg.out_dir:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best["state"], "config": asdict(cfg), "meta": meta}, out / "best.pt")
        np.savez(out / "predictions.npz", y_val=yv, p_val=pv, y_test=yt, p_test=pt)
        save_json(result, out / "metrics.json")
        if binary:
            plot_roc(yt, pt_c, out / "roc_test.png")
            plot_calibration(yt, pt_c, out / "calibration_test.png")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    ap.add_argument("--dataset", default=d.dataset)
    ap.add_argument("--arch", default=d.arch, choices=["resnet18", "simple_cnn"])
    ap.add_argument("--size", type=int, default=d.size, choices=[28, 64, 128, 224])
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--n-boot", type=int, default=d.n_boot)
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--no-download", action="store_true", help="Usar archivos ya presentes en --data-root")
    a = ap.parse_args()

    cfg = Config(
        dataset=a.dataset, arch=a.arch, size=a.size, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, weight_decay=a.weight_decay, seed=a.seed, class_weighted=not a.no_class_weights,
        patience=a.patience, n_boot=a.n_boot, device=a.device,
    )
    cfg.out_dir = str(Path(a.runs_dir) / f"{cfg.dataset}_{cfg.arch}_s{cfg.size}_seed{cfg.seed}")
    splits, meta = load_medmnist(cfg.dataset, cfg.size, a.data_root, download=not a.no_download)
    res = fit(cfg, splits, meta)

    t = res["test"]
    print(f"\n[{cfg.dataset} | {cfg.arch} | seed {cfg.seed}] mejor época: {res['best_epoch']}")
    for k in ("auc", "accuracy", "sensitivity", "specificity", "ece"):
        if k in t:
            v, (lo, hi) = t[k]["value"], t[k]["ci95"]
            print(f"  test {k:<11} {v:.3f}  IC95% [{lo:.3f}, {hi:.3f}]")
    print(f"  resultados en {cfg.out_dir}")


if __name__ == "__main__":
    main()
