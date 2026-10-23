"""Grad-CAM para inspeccionar en qué zonas se apoya el modelo.

Uso:
    python -m medcv.explain --run runs/pneumoniamnist_resnet18_s28_seed0

Advertencia: un mapa de calor NO demuestra que el modelo "razone" como un
clínico. Sirve para detectar atajos evidentes (bordes, marcas, texto), no para validar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .data import ArrayDataset, load_medmnist  # noqa: E402
from .model import build_model, last_conv_layer  # noqa: E402


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.act = None
        self.grad = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inp, out):
        self.act = out.detach()
        if out.requires_grad:  # en inferencia con no_grad no hay gradiente que capturar
            out.register_hook(lambda g: setattr(self, "grad", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | None = None) -> torch.Tensor:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        idx = logits.argmax(1) if class_idx is None else class_idx
        logits.gather(1, idx[:, None]).sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mn = cam.amin(dim=(1, 2), keepdim=True)
        mx = cam.amax(dim=(1, 2), keepdim=True)
        return (cam - mn) / (mx - mn + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Directorio de una corrida (contiene best.pt)")
    ap.add_argument("--n", type=int, default=6, help="Ejemplos correctos e incorrectos a mostrar")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    run = Path(a.run)
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cfg, meta = ckpt["config"], ckpt["meta"]
    model = build_model(cfg["arch"], meta["n_channels"], meta["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # BatchNorm en modo evaluación también para las predicciones
    cam = GradCAM(model, last_conv_layer(model, cfg["arch"]))

    splits, _ = load_medmnist(cfg["dataset"], cfg["size"], a.data_root, download=not a.no_download)
    ds = ArrayDataset(*splits["test"])
    with torch.no_grad():
        probs = torch.softmax(model(ds.x), dim=1)
    pred = probs.argmax(1)
    wrong = (pred != ds.y).nonzero().flatten().tolist()
    right = (pred == ds.y).nonzero().flatten().tolist()
    rng = np.random.default_rng(0)
    pick_r = rng.choice(right, size=min(a.n, len(right)), replace=False).tolist() if right else []
    pick_w = rng.choice(wrong, size=min(a.n, len(wrong)), replace=False).tolist() if wrong else []
    picks = [(i, "correcto") for i in pick_r] + [(i, "error") for i in pick_w]

    names = meta["labels"]
    fig, axes = plt.subplots(2, len(picks), figsize=(1.9 * len(picks), 4.2), squeeze=False)
    for col, (i, tag) in enumerate(picks):
        x = ds.x[i : i + 1].clone()
        heat = cam(x)[0].numpy()
        img = (x[0, 0].numpy() * 0.5 + 0.5) if x.shape[1] == 1 else np.transpose(x[0].numpy() * 0.5 + 0.5, (1, 2, 0))
        for row in (0, 1):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray" if x.shape[1] == 1 else None)
            if row == 1:
                ax.imshow(heat, cmap="jet", alpha=0.45)
            ax.axis("off")
        t, p = names[str(int(ds.y[i]))], names[str(int(pred[i]))]
        axes[0, col].set_title(f"{tag}\nreal: {t}\npred: {p}", fontsize=7)
    fig.suptitle("Grad-CAM (fila inferior). Solo inspección cualitativa, no validación clínica.", fontsize=8)
    fig.tight_layout()
    out = run / "gradcam.png"
    fig.savefig(out, dpi=150)
    print(f"Guardado {out}")


if __name__ == "__main__":
    main()
