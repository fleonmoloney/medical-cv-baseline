"""Métricas de evaluación: discriminación, umbral, calibración e intervalos de confianza.

Reglas del repo:
  * El umbral de decisión se elige SOLO con validación (índice de Youden).
  * El conjunto de test se evalúa una única vez, al final.
  * Toda métrica de test se informa con intervalo de confianza bootstrap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve  # noqa: E402


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC binario (y_prob: prob. de clase 1, forma (N,)) o macro one-vs-rest (N,K)."""
    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        p = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        return float(roc_auc_score(y_true, p))
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))


def youden_threshold(y_true: np.ndarray, p1: np.ndarray) -> float:
    """Umbral que maximiza sensibilidad + especificidad - 1."""
    fpr, tpr, thr = roc_curve(y_true, p1)
    return float(thr[np.argmax(tpr - fpr)])


def min_sensitivity_threshold(y_true: np.ndarray, p1: np.ndarray, min_sensitivity: float = 0.95) -> float:
    """Umbral con la mayor especificidad que aún alcanza `min_sensitivity` (criterio alternativo a Youden).

    Igual que Youden, se calcula SOLO con validación. Si ningún umbral llega a la sensibilidad pedida,
    devuelve el de mayor sensibilidad.
    """
    fpr, tpr, thr = roc_curve(y_true, p1)
    ok = np.where(tpr >= min_sensitivity)[0]
    idx = int(ok[0]) if len(ok) else int(np.argmax(tpr))  # fpr no decrece: el primero es el de menor fpr
    return float(thr[idx])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE con confianza de la clase predicha (top-label)."""
    probs = np.stack([1 - y_prob, y_prob], axis=1) if y_prob.ndim == 1 else y_prob
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _binary_counts(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: Optional[float]) -> Dict[str, float]:
    """Métricas puntuales. Binario: usa `threshold`. Multiclase: argmax."""
    if y_prob.ndim == 1:
        pred = (y_prob >= threshold).astype(int)
        c = _binary_counts(y_true, pred)
        return {
            "auc": auc(y_true, y_prob),
            "accuracy": float(accuracy_score(y_true, pred)),
            "sensitivity": _safe_div(c["tp"], c["tp"] + c["fn"]),
            "specificity": _safe_div(c["tn"], c["tn"] + c["fp"]),
            "ppv": _safe_div(c["tp"], c["tp"] + c["fp"]),
            "ece": expected_calibration_error(y_true, y_prob),
        }
    pred = y_prob.argmax(axis=1)
    return {
        "auc": auc(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """IC percentil por bootstrap. Se descartan remuestreos con una sola clase."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            v = fn(y_true[idx], y_prob[idx])
        except ValueError:
            continue
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, dict]:
    """Devuelve {metrica: {'value': v, 'ci95': [lo, hi]}} + conteos de confusión (binario)."""
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob)
    binary = y_prob.ndim == 1
    if binary and threshold is None:
        raise ValueError("En clasificación binaria hay que pasar un umbral (elegido en validación).")
    base = point_metrics(y_true, y_prob, threshold)
    out: Dict[str, dict] = {}
    for name, value in base.items():
        fn = lambda yt, yp, _n=name: point_metrics(yt, yp, threshold)[_n]  # noqa: E731
        lo, hi = bootstrap_ci(y_true, y_prob, fn, n_boot=n_boot, seed=seed)
        out[name] = {"value": value, "ci95": [lo, hi]}
    if binary:
        out["confusion"] = _binary_counts(y_true, (y_prob >= threshold).astype(int))
        out["threshold"] = float(threshold)
        out["prevalence"] = float(y_true.mean())
    out["n"] = int(len(y_true))
    return out


# ----------------------------------------------------------------------------- gráficos


def plot_roc(y_true: np.ndarray, p1: np.ndarray, path: str | Path, title: str = "Curva ROC (test)") -> None:
    fpr, tpr, _ = roc_curve(y_true, p1)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(y_true, p1):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("1 - especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray, p1: np.ndarray, path: str | Path, n_bins: int = 10, title: str = "Calibración (test)"
) -> None:
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p1 > lo) & (p1 <= hi)
        if m.sum() >= 5:
            xs.append(p1[m].mean())
            ys.append(y_true[m].mean())
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Calibración perfecta")
    ax.plot(xs, ys, "o-", lw=2, label="Modelo")
    ax.set_xlabel("Probabilidad predicha (clase 1)")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
