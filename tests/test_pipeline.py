"""Pruebas con datos sintéticos: no descargan nada y corren en segundos en CPU."""

import numpy as np
import pytest

from medcv.data import ArrayDataset, class_weights
from medcv.evaluate import (
    auc,
    bootstrap_ci,
    compute_metrics,
    expected_calibration_error,
    youden_threshold,
)
from medcv.explain import GradCAM
from medcv.model import build_model, last_conv_layer
from medcv.train import Config, fit

META_BIN = {
    "name": "synthetic",
    "task": "binary-class",
    "n_channels": 1,
    "n_classes": 2,
    "labels": {"0": "neg", "1": "pos"},
    "size": 28,
}


def make_split(n, seed, shift=1.5):
    """Dos clases separables: la clase 1 tiene un parche brillante."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = rng.normal(100, 20, (n, 28, 28)).clip(0, 255)
    for i in np.where(y == 1)[0]:
        x[i, 8:16, 8:16] += 60 * shift
    return x.clip(0, 255).astype(np.uint8), y.astype(np.int64)


def synthetic_splits():
    return {"train": make_split(240, 1), "val": make_split(80, 2), "test": make_split(80, 3)}


# ------------------------------------------------------------------ métricas


def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_youden_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(y, p)
    assert 0.3 < t <= 0.7


def test_ece_zero_for_perfectly_confident_correct():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_compute_metrics_requires_threshold_for_binary():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.3, 0.7]))


def test_bootstrap_is_reproducible_and_contains_value():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 300), 0, 1)
    a = bootstrap_ci(y, p, auc, n_boot=200, seed=7)
    b = bootstrap_ci(y, p, auc, n_boot=200, seed=7)
    assert a == b
    assert a[0] <= auc(y, p) <= a[1]


def test_class_weights_inverse_frequency():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y, 2)
    assert w[1] > w[0]


# ------------------------------------------------------------------ datos y modelos


def test_dataset_shapes_and_augmentation_keeps_size():
    x, y = make_split(10, 0)
    ds = ArrayDataset(x, y, train=True, pad=2)
    xi, yi = ds[0]
    assert xi.shape == (1, 28, 28)
    assert xi.min() >= -1.0 and xi.max() <= 1.0


@pytest.mark.parametrize("arch", ["simple_cnn", "resnet18"])
def test_models_forward(arch):
    import torch

    m = build_model(arch, 1, 2)
    out = m(torch.zeros(3, 1, 28, 28))
    assert out.shape == (3, 2)


@pytest.mark.parametrize("arch", ["simple_cnn", "resnet18"])
def test_gradcam_output_in_unit_range(arch):
    import torch

    m = build_model(arch, 1, 2)
    cam = GradCAM(m, last_conv_layer(m, arch))
    heat = cam(torch.randn(2, 1, 28, 28))
    assert heat.shape == (2, 28, 28)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0 + 1e-6


# ------------------------------------------------------------------ pipeline completo


def _cfg(tmp_path=None, seed=0):
    return Config(
        dataset="synthetic", arch="simple_cnn", epochs=4, batch_size=32, seed=seed,
        n_boot=50, device="cpu", out_dir=str(tmp_path) if tmp_path else None,
    )


def test_fit_learns_separable_task_and_writes_artifacts(tmp_path):
    res = fit(_cfg(tmp_path), synthetic_splits(), META_BIN)
    assert res["test"]["auc"]["value"] > 0.9
    for f in ("best.pt", "metrics.json", "predictions.npz", "roc_test.png", "calibration_test.png"):
        assert (tmp_path / f).exists(), f
    lo, hi = res["test"]["auc"]["ci95"]
    assert lo <= res["test"]["auc"]["value"] <= hi


def test_fit_is_reproducible_with_same_seed():
    a = fit(_cfg(seed=3), synthetic_splits(), META_BIN)
    b = fit(_cfg(seed=3), synthetic_splits(), META_BIN)
    assert a["test"]["auc"]["value"] == b["test"]["auc"]["value"]
    assert a["history"] == b["history"]


def test_threshold_comes_from_validation_only():
    res = fit(_cfg(seed=1), synthetic_splits(), META_BIN)
    assert res["val"]["threshold"] == res["test"]["threshold"]


# ------------------------------------------------------------------ carga con el paquete medmnist


def test_load_medmnist_reads_local_npz_without_network(tmp_path):
    """Simula el archivo .npz de MedMNIST para probar la ruta de carga real."""
    from medcv.data import load_medmnist

    xtr, ytr = make_split(40, 1)
    xva, yva = make_split(12, 2)
    xte, yte = make_split(12, 3)
    np.savez(
        tmp_path / "pneumoniamnist.npz",
        train_images=xtr, train_labels=ytr[:, None],
        val_images=xva, val_labels=yva[:, None],
        test_images=xte, test_labels=yte[:, None],
    )
    splits, meta = load_medmnist("pneumoniamnist", 28, str(tmp_path), download=False)
    assert meta["task"] == "binary-class" and meta["n_channels"] == 1
    assert splits["train"][0].shape == (40, 28, 28)
    res = fit(_cfg(), splits, meta)
    assert 0.0 <= res["test"]["auc"]["value"] <= 1.0


def test_gradcam_hook_survives_no_grad_inference():
    """Regresión: la predicción en lote bajo no_grad no debe romper el hook."""
    import torch

    m = build_model("simple_cnn", 1, 2).eval()
    GradCAM(m, last_conv_layer(m, "simple_cnn"))
    with torch.no_grad():
        m(torch.randn(2, 1, 28, 28))
