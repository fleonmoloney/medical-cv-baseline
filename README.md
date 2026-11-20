# medcv — línea base reproducible de visión por computadora sobre imágenes médicas

Clasificación de radiografías de tórax (neumonía pediátrica) con datos públicos de
[MedMNIST](https://medmnist.com), con foco en lo que suele faltar en los repositorios de
demostración: **reproducibilidad, evaluación honesta y límites declarados**.

> **English summary.** A small, fully reproducible computer-vision baseline on a public medical
> dataset (PneumoniaMNIST). Fixed seeds, threshold chosen on validation only, test set evaluated
> once, bootstrap confidence intervals, calibration (ECE), Grad-CAM inspection, multi-seed
> aggregation and unit tests that run without downloading any data. **Research/portfolio code,
> not a medical device.**

## Resultados

Ejecutados el 19/11/2026 en Colab con `scripts/run_all.sh` (5 semillas por modelo, PneumoniaMNIST 28×28).
Tabla generada por `python -m medcv.aggregate` (`results/summary.md`), sin cambios a mano. Umbral de decisión
elegido solo en validación con el índice de Youden; test evaluado una vez.

| Dataset | Modelo | Tamaño | Semillas | AUC | ACCURACY | SENSITIVITY | SPECIFICITY | ECE |
|---|---|---|---|---|---|---|---|---|
| pneumoniamnist | resnet18 | 28 | 5 | 0.982 ± 0.001 | 0.907 ± 0.016 | 0.996 ± 0.003 | 0.757 ± 0.042 | 0.083 ± 0.039 |
| pneumoniamnist | simple_cnn | 28 | 5 | 0.976 ± 0.002 | 0.893 ± 0.018 | 0.994 ± 0.007 | 0.725 ± 0.056 | 0.045 ± 0.013 |

_Media ± desvío estándar entre semillas, sobre el conjunto de test._

**Lectura.** La sensibilidad es alta (≈ 0.99) pero la especificidad en test es moderada (0.73–0.76): alrededor
de 1 de cada 4 radiografías normales se marca como neumonía. resnet18 no queda mejor calibrado que
`simple_cnn` (ECE 0.083 frente a 0.045).

### Segundo punto de operación (análisis adicional)

Con el resultado anterior a la vista se agregó un segundo criterio de umbral, también elegido solo en
validación: la mayor especificidad con sensibilidad de validación ≥ 0.95. **Se agregó después de ver el
resultado de test con Youden**, por lo que Youden sigue siendo el resultado principal.
Generado con `python -m medcv.operating_points` (`results/operating_points.md`), sin reentrenar.

| Modelo | Criterio | Semillas | Sens. val | Espec. val | Sens. test | Espec. test | Exactitud test | VPP test |
|---|---|---|---|---|---|---|---|---|
| resnet18 (28 px) | Youden | 5 | 0.978 ± 0.010 | 0.990 ± 0.011 | 0.996 ± 0.003 | 0.757 ± 0.042 | 0.907 ± 0.016 | 0.873 ± 0.019 |
| resnet18 (28 px) | Sens. val ≥ 0.95 | 5 | 0.962 ± 0.006 | 0.999 ± 0.003 | 0.986 ± 0.006 | 0.825 ± 0.056 | 0.926 ± 0.018 | 0.904 ± 0.028 |
| simple_cnn (28 px) | Youden | 5 | 0.976 ± 0.005 | 0.990 ± 0.008 | 0.994 ± 0.007 | 0.725 ± 0.056 | 0.893 ± 0.018 | 0.858 ± 0.025 |
| simple_cnn (28 px) | Sens. val ≥ 0.95 | 5 | 0.964 ± 0.004 | 0.999 ± 0.003 | 0.987 ± 0.003 | 0.785 ± 0.021 | 0.911 ± 0.007 | 0.885 ± 0.010 |

La especificidad en validación es ≈ 0.99–1.0 con ambos criterios y baja a 0.73–0.83 en test: la validación es
demasiado fácil para elegir un umbral que se traslade a test. El segundo criterio mejora la especificidad en
test en unos 0.06–0.07, del orden del desvío entre semillas.

### Particiones

`python -m medcv.dataset_info` (`results/dataset_info.json`): las radiografías normales son el 25.8 % en
entrenamiento (1214 de 4708), 25.8 % en validación (135 de 524) y **37.5 % en test (234 de 624)**. Las particiones no
tienen la misma distribución. La prevalencia por sí sola no explica la baja de especificidad (se calcula solo entre
normales): sugiere además que las radiografías normales de test difieren de las de entrenamiento y validación.
Es una hipótesis aún sin investigar; tampoco se verificó si hay imágenes de un mismo paciente repartidas entre particiones.

### Grad-CAM

En una inspección cualitativa de 12 ejemplos de test (resnet18, semilla 0; 6 correctos y 6 errores), el foco cae
mayormente en la zona central de la imagen y no en los campos pulmonares; en algunos casos va a un borde o esquina.
No prueba un atajo, pero es lo que conviene investigar con más ejemplos y semillas.

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
python -m medcv.operating_points          # compara Youden y sensibilidad mínima (usa predictions.npz)

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
  En la ejecución del 19/11/2026 se observó: normales 25.8 % en entrenamiento y validación frente a 37.5 % en test.
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
  operating_points.py  compara umbrales (Youden y sensibilidad mínima) sin reentrenar
tests/             19 pruebas con datos sintéticos (sin red)
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
