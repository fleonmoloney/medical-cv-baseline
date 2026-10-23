"""Agrega varias semillas en una tabla media ± desvío (markdown).

Uso:
    python -m medcv.aggregate --runs-dir runs --out results/summary.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["auc", "accuracy", "sensitivity", "specificity", "ece"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results/summary.md")
    a = ap.parse_args()

    groups = defaultdict(list)
    for f in sorted(Path(a.runs_dir).glob("*/metrics.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        c = r["config"]
        groups[(c["dataset"], c["arch"], c["size"])].append(r)
    if not groups:
        raise SystemExit(f"No hay corridas en {a.runs_dir}/*/metrics.json")

    lines = [
        "| Dataset | Modelo | Tamaño | Semillas | " + " | ".join(m.upper() for m in METRICS) + " |",
        "|---|---|---|---|" + "---|" * len(METRICS),
    ]
    for (ds, arch, size), runs in sorted(groups.items()):
        cells = []
        for m in METRICS:
            vals = [r["test"][m]["value"] for r in runs if m in r["test"]]
            if not vals:
                cells.append("–")
            else:
                sd = np.std(vals, ddof=1) if len(vals) > 1 else float("nan")
                cells.append(f"{np.mean(vals):.3f} ± {sd:.3f}")
        lines.append(f"| {ds} | {arch} | {size} | {len(runs)} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n\n_Media ± desvío estándar entre semillas, sobre el conjunto de test._\n"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
