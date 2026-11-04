# Proyecto Integrador de Visión por Computadora — 2026
## Detección, Segmentación y Estimación de Pose utilizando YOLO

Repositorio correspondiente al Trabajo Integrador Final de la materia **Visión por Computadora** (Licenciatura en Ciencia de Datos, Universidad del Gran Rosario - UGR).

El proyecto implementa una solución integral de Computer Vision que entrena un modelo YOLO sobre **clases personalizadas no pertenecientes a COCO** (`mate` y `termo`), y procesa videos en tiempo real combinando simultáneamente **tres arquitecturas de visión por computadora**:
1. **Detección de Clases Personalizadas** (`mate`, `termo`) con Bounding Boxes, clase y confianza.
2. **Segmentación Semántica de Objetos COCO** (excluyendo personas y las clases personalizadas para evitar falsos solapamientos).
3. **Estimación de Pose Humana (Pose Estimation)** con keypoints, esqueleto corporal y tracking visual de personas detectadas.

---

## 👥 Integrantes del Equipo
* **Bruno Pace**
* **Franco Bianciotto**
* **Franco Calcia**
* **Tomas Avecilla**
* **Juana Zorzolo**

---

## 📁 Estructura del Repositorio

```text
CV-mate-detection/
├── README.md
├── FALTA_HACER.md                    # Tareas pendientes para la entrega final
├── requirements.txt
├── pyproject.toml                    # Configuración de ruff (PEP 8 + docstrings Google)
├── docs/
│   └── enunciado_2026.pdf            # Enunciado oficial de la cátedra
├── dataset/                          # Imágenes originales (JPG/JPEG)
├── data/
│   ├── raw_annotations/              # CSV exportados por cada integrante (MakeSense)
│   └── unified/
│       ├── annotations.csv           # Todas las cajas unificadas
│       └── labels/                   # Un .txt YOLO por imagen + classes.txt
├── src/
│   ├── unify_annotations.py          # raw_annotations/ -> unified/
│   └── validate_dataset.py           # Validación del dataset y requisito de 50 imágenes/clase
├── notebooks/
│   ├── 01_Entrenamiento_YOLO26.ipynb # Split, transfer learning y evaluación en test
│   └── 02_Procesamiento_Videos_YOLO26.ipynb # Detección + segmentación + pose sobre video
├── models/
│   └── best.pt                       # Pesos entrenados (los copia el notebook 01)
├── resultados/                       # Métricas y gráficos finales (los copia el notebook 01)
└── videos/
    ├── input/                        # Videos originales
    └── output/                       # Videos procesados
```

Las carpetas `dataset_yolo26_mate_termo/` (split train/val/test) y `entrenamientos/` (corridas de Ultralytics) las genera el notebook 01 y no se versionan.

---

## 🏷️ Clases Personalizadas y Homogeneización de Etiquetas

Se seleccionaron 2 clases que **no pertenecen a las 80 clases de COCO**:
* **Clase `0`**: `mate`
* **Clase `1`**: `termo`

### Problema resuelto en la unificación de datos
Durante la etapa de etiquetado colaborativo, las herramientas de exportación asignaron órdenes distintos de clases entre integrantes (en particular, las anotaciones de Bruno tenían los IDs invertidos: `0` para termo y `1` para mate).

El script [`src/unify_annotations.py`](src/unify_annotations.py) resolvió este problema de forma automática y reproducible:
* Homogeneizó todas las etiquetas al estándar oficial `{0: 'mate', 1: 'termo'}`.
* Eliminó anotaciones duplicadas sobre una misma imagen (`IMG_7177.jpg`).
* Validó que todas las coordenadas estuvieran normalizadas en `[0.0, 1.0]`.
* Limpió copias redundantes de descarga (`IMG_4041(1).jpeg`, etc.).

Para volver a generar o auditar las etiquetas:
```bash
python3 src/unify_annotations.py
python3 src/validate_dataset.py
```

---

## ⚙️ Instalación y Requisitos

Se recomienda utilizar un entorno virtual con **Python 3.10+**:

```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Flujo de Trabajo y Ejecución

### 1. Notebook 01: Entrenamiento (`notebooks/01_Entrenamiento_YOLO26.ipynb`)
* **Objetivo:** Aplicar *Transfer Learning* partiendo de los pesos preentrenados en COCO de YOLO26 nano (`yolo26n.pt`, Ultralytics >= 8.4) sobre el dataset de mate y termo.
* **Entorno recomendado:** Google Colab con acelerador GPU (T4 o superior).
* **Parámetros configurados:**
  * `EPOCHS = 120` (con early stopping, `PATIENCE = 15`)
  * `IMGSZ = 640`
  * `BATCH = 16`
  * Split de datos: 80% train / 10% val / 10% test.
* **Salidas generadas:**
  * `entrenamientos/mate_termo_yolo26/weights/best.pt` (pesos del mejor modelo).
  * Curvas de aprendizaje (`results.png`), curvas PR y F1 (`BoxPR_curve.png`, `BoxF1_curve.png`).
  * Matriz de confusión con `conf=0.35` en `entrenamientos/test_mate_termo_conf35/`.
  * Reporte cuantitativo en test con P, R, F1 y mAP (`metricas_test.json` y `metricas_por_clase.csv`).

### 2. Notebook 02: Procesamiento de Videos (`notebooks/02_Procesamiento_Videos_YOLO26.ipynb`)
* **Objetivo:** Ingerir videos de entrada (ubicados en `videos/input/`) de hasta 2 minutos y procesarlos integrando simultáneamente (modelos `best.pt`, `yolo26n-seg.pt` y `yolo26n-pose.pt`):
  * **Segmentación:** Segmenta objetos COCO (tazas, botellas, mesas, libros, etc.) que superen `CONF_SEG=0.45`, excluyendo la clase `person` y suprimiendo máscaras que solapen con las cajas de mate/termo.
  * **Pose:** Estima keypoints y dibuja el esqueleto de todas las personas detectadas con tracking persistente.
  * **Detección Custom:** Dibuja bounding boxes diferenciadas por color para `mate` y `termo`, indicando clase y score de confianza.
* **Salida generada:**
  * Videos finales codificados en H.264 (`.mp4`) en `videos/output/`.
  * Gráfico comparativo de fotogramas (original vs procesado).
  * Reporte de cobertura y conteo de detecciones (`resumen_procesamiento.csv`).

### Estilo de código
El código sigue PEP 8 (líneas de 79 caracteres) y documenta cada función con docstrings estilo Google. La configuración de `ruff` está en `pyproject.toml` y cubre `src/` y los notebooks:
```bash
ruff check src/ notebooks/
```

---

## 📋 Entregables del Proyecto

Según las consignas del enunciado, la entrega final debe incluir:
1. **Código fuente:** Notebooks completos y scripts auxiliares.
2. **Dataset:** Imágenes y etiquetas unificadas en formato YOLO, junto al archivo `data.yaml`.
3. **Pesos entrenados:** Archivo `best.pt`.
4. **Videos:** Videos originales y videos finales procesados con los 3 modelos integrados.
5. **Documentación final:** Resumen de dataset, proceso de entrenamiento, métricas obtenidas, ejemplos visuales y conclusiones.

> [!NOTE]
> Para consultar la lista detallada de tareas pendientes antes de la entrega formal, consultar el archivo [`FALTA_HACER.md`](FALTA_HACER.md).
