# medcv — línea base reproducible de visión por computadora sobre imágenes médicas

Clasificación de radiografías de tórax (neumonía pediátrica) con datos públicos de
[MedMNIST](https://medmnist.com), con foco en lo que suele faltar en los repositorios de
demostración: **reproducibilidad, evaluación honesta y límites declarados**.

> **English summary.** A small, fully reproducible computer-vision baseline on a public medical
> dataset (PneumoniaMNIST). Fixed seeds, threshold chosen on validation only, test set evaluated
> once, bootstrap confidence intervals, calibration (ECE), Grad-CAM inspection, multi-seed
> aggregation and unit tests that run without downloading any data. **Research/portfolio code,
> not a medical device.**

## Estado de los resultados

**Este repositorio todavía no incluye resultados medidos.** El código y las pruebas están
verificados con datos sintéticos, pero las métricas reales se generan ejecutando los pasos de
abajo. Cuando lo hagas, pegá aquí el contenido de `results/summary.md` y no cambies ningún
número a mano.

| Dataset | Modelo | Semillas | AUC | Sensibilidad | Especificidad | ECE |
|---|---|---|---|---|---|---|
| _pendiente de ejecución_ | | | | | | |

## Reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # pruebas con datos sintéticos, sin descargas
python -m medcv.dataset_info             # tamaños y balance de clases (descarga el dataset)

# una corrida
python -m medcv.train --dataset pneumoniamnist --arch resnet18 --seed 0

# 5 semillas por modelo + tabla agregada
bash scripts/run_all.sh
python -m medcv.aggregate                # escribe results/summary.md

# inspección cualitativa
python -m medcv.explain --run runs/pneumoniamnist_resnet18_s28_seed0
```

Cada corrida guarda en `runs/<dataset>_<modelo>_s<tamaño>_seed<n>/`: `metrics.json` (configuración,
versiones del entorno, métricas con IC95 %, historial), `predictions.npz`, `best.pt`, curva ROC y
curva de calibración.

**Tiempos de referencia** (medidos en una CPU de 2 núcleos con datos sintéticos del mismo tamaño que
PneumoniaMNIST, una época): `simple_cnn` ≈ 4 s, `resnet18` ≈ 76 s. Con 30 épocas, `resnet18` toma
unos 40 minutos por semilla en esa CPU; en una GPU (por ejemplo Colab) son minutos.

## Decisiones de diseño

| Decisión | Motivo |
|---|---|
| Umbral de decisión elegido **solo en validación** (índice de Youden) | Elegirlo con test infla la sensibilidad y la especificidad. |
| Test evaluado **una sola vez**, con el mejor checkpoint de validación | Evita el sobreajuste al conjunto de test por iteración. |
| IC95 % por **bootstrap** en cada métrica de test | Un número sin incertidumbre sobre ~600 imágenes engaña. |
| **ECE** y curva de calibración | En medicina importa que la probabilidad signifique algo, no solo el AUC. |
| Varias **semillas** y reporte media ± desvío | Una semilla afortunada no es un resultado. |
| Entrenamiento **desde cero**, sin pesos preentrenados | Sin descargas ocultas ni datos de origen incontrolado. |
| Normalización fija `(x-0.5)/0.5` y pesos de clase calculados solo con train | Sin fuga de información entre particiones. |
| Sin volteos horizontales | En radiografías el lado tiene significado anatómico. |
| `set_seed` + algoritmos deterministas | En CPU, misma semilla ⇒ mismos resultados (hay una prueba que lo verifica). |
| Entorno (versiones, dispositivo) guardado en cada `metrics.json` | Trazabilidad de qué produjo cada número. |

## Límites y uso responsable

- **No es un dispositivo médico ni sirve para decisiones clínicas.** Es código de investigación y de portafolio.
- MedMNIST reduce las imágenes (28×28 por defecto). Se pierde detalle que un radiólogo usaría; los
  resultados no son comparables con trabajos que usan resolución completa.
- Según la documentación de MedMNIST, las particiones provienen de las del dataset original
  (Kermany et al., 2018). Puede haber **diferencia de distribución entre validación y test**: si ves un
  salto de rendimiento, reportalo en lugar de ocultarlo. `python -m medcv.dataset_info` muestra el balance
  de clases por partición.
- Es un único dataset pediátrico de una sola institución: no hay validación externa, así que no se
  puede afirmar generalización a otras poblaciones, equipos o centros.
- **Grad-CAM no valida el razonamiento del modelo.** Sirve para detectar atajos evidentes (bordes, marcas
  en la imagen), no para demostrar que el modelo "mira donde mira un clínico".
- No se evalúa equidad por subgrupos porque el dataset no incluye metadatos demográficos.

Más detalle en [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Estructura

```
src/medcv/
  data.py          carga con medmnist y Dataset de tensores (sin fuga entre particiones)
  model.py         simple_cnn y resnet18 adaptada a imágenes pequeñas
  train.py         entrenamiento, selección de umbral en validación, evaluación única en test
  evaluate.py      AUC, sensibilidad, especificidad, ECE, bootstrap, gráficos
  explain.py       Grad-CAM
  aggregate.py     tabla media ± desvío entre semillas
  dataset_info.py  tamaños y balance de clases por partición
tests/             16 pruebas con datos sintéticos (sin red)
scripts/run_all.sh 5 semillas × 2 modelos
```

## Citas y licencias

- Yang, J. et al. *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image
  classification.* Scientific Data 10, 41 (2023).
- Kermany, D. S. et al. *Identifying medical diagnoses and treatable diseases by image-based deep
  learning.* Cell 172, 1122–1131 (2018).
- Código bajo licencia MIT (ver `LICENSE`). Los datos de MedMNIST tienen su propia licencia (consultá
  la ficha de cada subconjunto en medmnist.com) y no se redistribuyen en este repositorio.

## Autor

Fernando León Moloney — Ingeniero en Sistemas, especialización en Ingeniería Biomédica (UdelaR).
