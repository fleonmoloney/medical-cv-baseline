# Ficha del modelo

## Resumen

| | |
|---|---|
| Tarea | Clasificación binaria de radiografías de tórax: normal vs. neumonía (pediátrica) |
| Datos | PneumoniaMNIST (MedMNIST), imágenes en escala de grises, 28×28 por defecto |
| Modelos | `simple_cnn` (3 bloques convolucionales) y `resnet18` adaptada a imágenes pequeñas, entrenados desde cero |
| Salida | Probabilidad de la clase "pneumonia" y decisión con umbral fijado en validación |
| Estado | Línea base de investigación y portafolio |

## Uso previsto y no previsto

- **Previsto:** demostrar un flujo reproducible de entrenamiento y evaluación; servir de base para
  comparar métodos en el mismo benchmark.
- **No previsto:** diagnóstico, triaje o cualquier decisión clínica. No hay validación externa,
  ni estudio con lectores humanos, ni aprobación regulatoria.

## Datos

- Fuente: Kermany et al. (2018), redistribuido y reducido por MedMNIST (Yang et al., 2023).
- Las particiones se toman tal como las entrega `medmnist`. Los tamaños y el balance de clases
  reales se obtienen con `python -m medcv.dataset_info` (se guardan en `results/dataset_info.json`).
- No hay metadatos demográficos en el subconjunto, por lo que no se puede evaluar desempeño por edad,
  sexo ni equipo de captura.

## Entrenamiento

- Pérdida: entropía cruzada con pesos de clase calculados en train. Optimizador: AdamW con
  coseno. Aumento: traslación aleatoria de hasta 2 píxeles; sin volteos.
- Selección de modelo: mejor AUC de validación, con parada temprana.
- Semillas fijas y algoritmos deterministas; el entorno se guarda en cada `metrics.json`.

## Evaluación

- Umbral: índice de Youden calculado **solo en validación**.
- Test: una sola evaluación con el mejor checkpoint.
- Métricas: AUC, exactitud, sensibilidad, especificidad, valor predictivo positivo, ECE; cada una con
  IC95 % por bootstrap (1000 remuestreos por defecto).
- Resultados: ver `results/summary.md` después de ejecutar `scripts/run_all.sh`.

## Riesgos y limitaciones conocidos

1. **Resolución reducida:** pierde detalle diagnóstico; los números no se pueden comparar con
   trabajos a resolución completa.
2. **Posible cambio de distribución** entre validación y test, según la documentación del dataset.
   Un umbral elegido en validación puede no ser óptimo en test; se informa la diferencia.
3. **Una institución, una población pediátrica:** sin evidencia de generalización a otros centros,
   equipos o edades.
4. **Atajos posibles:** los modelos pueden apoyarse en artefactos de la imagen y no en la patología.
   Grad-CAM ayuda a detectar casos evidentes, pero no los descarta.
5. **Calibración:** una probabilidad bien calibrada en este dataset no garantiza calibración en otro.
6. **Sin análisis de subgrupos** por falta de metadatos.

## Cómo reportar resultados con este repositorio

Informar siempre: media ± desvío entre semillas, IC95 % de test, diferencia validación–test, ECE,
y las limitaciones anteriores. No modificar los números a mano ni reportar una única semilla.
