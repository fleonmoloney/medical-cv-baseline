"""Compara dos puntos de operación sobre las predicciones ya guardadas (no reentrena nada).

Criterios, ambos con el umbral elegido SOLO en validación:
  * youden:   maximiza sensibilidad + especificidad - 1 (criterio original del repo).
  * sens_min: mayor especificidad con sensibilidad de validación >= --min-sensitivity.

Este análisis se agregó DESPUÉS de ver los resultados de test con el criterio Youden: se presenta como
análisis adicional del compromiso sensibilidad/especificidad, no como el resultado principal.
Mostrar también la sensibilidad y la especificidad en validación permite ver si el umbral elegido allí
se traslada a test (una brecha grande sugiere diferencia de distribución entre particiones).

Uso:
    python -m medcv.operating_points --runs-dir runs --out results/operating_points.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluate import min_sensitivity_threshold, point_metrics, youden_threshold


def _p1(p: np.ndarray) -> np.ndarray:
    return p[:, 1] if p.ndim == 2 else p


def evaluate_run(run_dir: Path, min_sensitivity: float) -> dict:
    """Devuelve {criterio: {umbral, val:{...}, test:{...}}} para una corrida binaria."""
    d = np.load(run_dir / "predictions.npz")
    yv, pv = d["y_val"].reshape(-1), _p1(d["p_val"])
    yt, pt = d["y_test"].reshape(-1), _p1(d["p_test"])
    thresholds = {
        "youden": youden_threshold(yv, pv),
        "sens_min": min_sensitivity_threshold(yv, pv, min_sensitivity),
    }
    return {
        name: {
            "threshold": thr,
            "val": point_metrics(yv, pv, thr),
            "test": point_metrics(yt, pt, thr),
        }
        for name, thr in thresholds.items()
    }


def _ms(vals: list[float]) -> str:
    sd = np.std(vals, ddof=1) if len(vals) > 1 else float("nan")
    return f"{np.mean(vals):.3f} ± {sd:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results/operating_points.md")
    ap.add_argument("--min-sensitivity", type=float, default=0.95)
    a = ap.parse_args()

    groups = defaultdict(list)
    for f in sorted(Path(a.runs_dir).glob("*/metrics.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["dataset_meta"]["task"] != "binary-class" or not (f.parent / "predictions.npz").exists():
            continue
        c = r["config"]
        groups[(c["dataset"], c["arch"], c["size"])].append(evaluate_run(f.parent, a.min_sensitivity))
    if not groups:
        raise SystemExit(f"No hay corridas binarias con predictions.npz en {a.runs_dir}/")

    head = "| Modelo | Criterio | Semillas | Sens. val | Espec. val | Sens. test | Espec. test | Exactitud test | VPP test |"
    lines = [head, "|" + "---|" * 9]
    labels = {"youden": "Youden", "sens_min": f"Sens. val ≥ {a.min_sensitivity:g}"}
    for (ds, arch, size), runs in sorted(groups.items()):
        for crit in ("youden", "sens_min"):
            g = lambda split, m: _ms([r[crit][split][m] for r in runs])  # noqa: E731
            lines.append(
                f"| {arch} ({size} px) | {labels[crit]} | {len(runs)} | {g('val','sensitivity')} | {g('val','specificity')} "
                f"| {g('test','sensitivity')} | {g('test','specificity')} | {g('test','accuracy')} | {g('test','ppv')} |"
            )
    text = (
        "\n".join(lines)
        + "\n\n_Media ± desvío entre semillas. Umbral elegido solo en validación. Análisis agregado después de ver "
        "el resultado de test con Youden: es un análisis adicional, no el resultado principal._\n"
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
