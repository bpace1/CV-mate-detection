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
├── README.md                                   # Documentación principal del proyecto
├── requirements.txt                            # Dependencias y librerías requeridas
├── FALTA_HACER.md                              # Guía de tareas pendientes para la entrega final
├── Proyecto Integrador Visión por Computadora - 2026.pdf # Enunciado oficial de la cátedra
│
├── data/
│   ├── raw_annotations/                        # Anotaciones originales de cada integrante
│   │   ├── anotaciones_bruno.csv / .zip
│   │   ├── labels_Franco_B.csv / .zip
│   │   └── labels-juana.csv / .zip
│   └── unified/                                # Dataset y etiquetas unificadas y corregidas
│       ├── annotations.csv                     # CSV unificado con todas las etiquetas
│       ├── labels_yolo.zip                     # ZIP con todos los .txt en formato YOLO
│       ├── labels/                             # Archivos .txt individuales y classes.txt
│       └── data.yaml                           # Archivo de configuración YOLO
│
├── dataset/                                    # Imágenes del dataset (formato JPG/JPEG)
│
├── notebooks/
│   ├── 01_Entrenamiento_YOLO26.ipynb           # Notebook de entrenamiento (Transfer Learning y Test)
│   └── 02_Procesamiento_Videos_YOLO26.ipynb    # Notebook de inferencia multi-modelo sobre videos
│
├── src/
│   ├── __init__.py
│   ├── utils.py                                # Utilidades de visualización, video y colores
│   ├── unify_annotations.py                    # Script de homogeneización de anotaciones
│   └── validate_dataset.py                     # Script de validación estricta y auditoría del dataset
│
├── videos/
│   ├── input/                                  # Videos originales de prueba (subir aquí)
│   └── output/                                 # Videos procesados finales con overlay multi-modelo
│
└── models/
    └── .gitkeep                                # Directorio de pesos entrenados (best.pt)
```

---

## 🏷️ Clases Personalizadas y Homogeneización de Etiquetas

Se seleccionaron 2 clases que **no pertenecen a las 80 clases de COCO**:
* **Clase `0`**: `mate`
* **Clase `1`**: `termo`

### Problema resuelto en la unificación de datos
Durante la etapa de etiquetado colaborativo, las herramientas de exportación asignaron órdenes distintos de clases entre integrantes (en particular, las anotaciones de Bruno tenían los IDs invertidos: `0` para termo y `1` para mate).

El script [`src/unify_annotations.py`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/src/unify_annotations.py) resolvió este problema de forma automática y reproducible:
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
* **Objetivo:** Aplicar *Transfer Learning* partiendo de pesos preentrenados (Ultralytics YOLO11 / YOLOv8) sobre el dataset de mate y termo.
* **Entorno recomendado:** Google Colab con acelerador GPU (T4 o superior).
* **Parámetros configurados:**
  * `EPOCHS = 60` (con early stopping por paciencia)
  * `IMGSZ = 640`
  * `BATCH = 16`
  * Split de datos: 80% train / 10% val / 10% test.
* **Salidas generadas:**
  * `entrenamientos/mate_termo_yolo26/weights/best.pt` (pesos del mejor modelo).
  * Curvas de aprendizaje (`results.png`), matriz de confusión (`confusion_matrix.png`), curvas PR y F1.
  * Reporte cuantitativo en test (`metricas_test.json` y `metricas_por_clase.csv`).

### 2. Notebook 02: Procesamiento de Videos (`notebooks/02_Procesamiento_Videos_YOLO26.ipynb`)
* **Objetivo:** Ingerir videos de entrada (ubicados en `videos/input/`) de hasta 2 minutos y procesarlos integrando simultáneamente:
  * **Segmentación:** Segmenta objetos COCO (tazas, botellas, mesas, libros, etc.) que superen `CONF_SEG=0.45`, excluyendo la clase `person` y suprimiendo máscaras que solapen con las cajas de mate/termo.
  * **Pose:** Estima keypoints y dibuja el esqueleto de todas las personas detectadas con tracking persistente.
  * **Detección Custom:** Dibuja bounding boxes diferenciadas por color para `mate` y `termo`, indicando clase y score de confianza.
* **Salida generada:**
  * Videos finales codificados en H.264 (`.mp4`) en `videos/output/`.
  * Gráfico comparativo de fotogramas (original vs procesado).
  * Reporte de cobertura y conteo de detecciones (`resumen_procesamiento.csv`).

---

## 📋 Entregables del Proyecto

Según las consignas del enunciado, la entrega final debe incluir:
1. **Código fuente:** Notebooks completos y scripts auxiliares.
2. **Dataset:** Imágenes y etiquetas unificadas en formato YOLO, junto al archivo `data.yaml`.
3. **Pesos entrenados:** Archivo `best.pt`.
4. **Videos:** Videos originales y videos finales procesados con los 3 modelos integrados.
5. **Documentación final:** Resumen de dataset, proceso de entrenamiento, métricas obtenidas, ejemplos visuales y conclusiones.

> [!NOTE]
> Para consultar la lista detallada de tareas pendientes antes de la entrega formal, consultar el archivo [`FALTA_HACER.md`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/FALTA_HACER.md).
