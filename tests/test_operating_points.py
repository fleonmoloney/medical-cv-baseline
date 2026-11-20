"""Pruebas del criterio de sensibilidad mínima y del análisis de puntos de operación (datos sintéticos)."""

import json

import numpy as np

from medcv.evaluate import min_sensitivity_threshold, youden_threshold
from medcv.operating_points import evaluate_run


def test_min_sensitivity_threshold_reaches_requested_sensitivity():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    p = np.r_[rng.beta(2, 5, 200), rng.beta(5, 2, 200)]
    thr = min_sensitivity_threshold(y, p, 0.95)
    sens = ((p >= thr) & (y == 1)).sum() / (y == 1).sum()
    assert sens >= 0.95
    # un umbral algo más alto ya no alcanzaría la sensibilidad pedida (es el de mayor especificidad)
    higher = p[p > thr].min() if (p > thr).any() else thr
    assert ((p >= higher) & (y == 1)).sum() / (y == 1).sum() < 0.95 or higher == thr


def test_min_sensitivity_is_at_most_youden_threshold_when_youden_is_less_sensitive():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    p = np.r_[rng.beta(2, 5, 300), rng.beta(4, 2, 300)]
    ty, ts = youden_threshold(y, p), min_sensitivity_threshold(y, p, 0.99)
    assert ts <= ty


def test_evaluate_run_reads_saved_predictions(tmp_path):
    rng = np.random.default_rng(2)

    def split(n):
        y = rng.integers(0, 2, n)
        p1 = np.clip(0.3 * y + rng.normal(0.35, 0.2, n), 0.01, 0.99)
        return y, np.stack([1 - p1, p1], axis=1)

    yv, pv = split(150)
    yt, pt = split(150)
    np.savez(tmp_path / "predictions.npz", y_val=yv, p_val=pv, y_test=yt, p_test=pt)
    out = evaluate_run(tmp_path, 0.9)
    assert set(out) == {"youden", "sens_min"}
    assert out["sens_min"]["val"]["sensitivity"] >= 0.9
    assert 0.0 <= out["youden"]["test"]["specificity"] <= 1.0
    json.dumps({k: v["threshold"] for k, v in out.items()})  # serializable
