# Detección, segmentación y pose con YOLO26

Trabajo integrador final de Visión por Computadora 2026, Licenciatura en Ciencia de Datos, Universidad del Gran Rosario (UGR). Grupo 5.

Entrenamos un detector YOLO26 para dos clases que no están en COCO, `mate` y `termo`. Después procesamos videos con tres modelos sobre cada fotograma:

1. El detector propio dibuja una caja por cada mate o termo, con la clase y la confianza.
2. `yolo26n-seg.pt` pinta máscaras sobre los objetos COCO. Omite las personas y las máscaras que caen sobre un mate o un termo.
3. `yolo26n-pose.pt` dibuja los keypoints y el esqueleto de cada persona, con un ID de seguimiento.

## Integrantes

- Bruno Pace
- Franco Bianciotto
- Franco Calcia
- Tomas Avecilla
- Juana Zorzolo

## Resultados

El modelo final es YOLO26n entrenado sin mosaic, elegido entre cuatro configuraciones por su mAP50-95 en validación. Con confianza 0.25, en test (14 imágenes, 17 objetos):

| Precisión | Recall | F1 | mAP50 | mAP50-95 |
|-----------|--------|----|-------|----------|
| 0.875 | 0.879 | 0.874 | 0.905 | 0.593 |

Los cuatro videos procesados están en `videos/output/`. El análisis del entrenamiento está en la sección 10 del notebook 01 y el de los videos en la sección 7 del notebook 02.

## Estructura del repositorio

```text
CV-mate-detection/
├── README.md
├── FALTA_HACER.md                    # Pendientes antes de entregar
├── data.yaml                         # Configuración del dataset para Ultralytics
├── requirements.txt                  # Dependencias para correr en local
├── requirements-training.txt         # Dependencias que instala el notebook 01
├── requirements-video.txt            # Dependencias que instala el notebook 02
├── pyproject.toml                    # Configuración de ruff
├── docs/
│   ├── enunciado_2026.pdf            # Enunciado de la cátedra
│   ├── entrega_y_rubrica.md          # Consigna de entrega y rúbrica
│   ├── dataset_y_entrenamiento.md    # Correcciones, split y comparación de modelos
│   └── videos.md                     # Videos usados y revisión de resultados
├── dataset/                          # 90 fotos originales
├── data/
│   ├── raw_annotations/              # CSV de MakeSense de cada integrante
│   ├── unified/                      # Etiquetas YOLO, una por imagen
│   ├── source_groups.csv             # Sesión a la que pertenece cada foto
│   └── split_manifest.csv            # Split train/val/test con hashes
├── src/
│   ├── unify_annotations.py          # CSV de MakeSense -> etiquetas YOLO
│   ├── validate_dataset.py           # Valida etiquetas y mínimo de imágenes por clase
│   ├── prepare_dataset.py            # Arma el dataset dividido a partir del manifiesto
│   └── training_utils.py             # Split por sesiones y métricas, usado por el notebook 01
├── notebooks/
│   ├── 01_Entrenamiento_YOLO26.ipynb        # Dataset, entrenamiento y evaluación
│   └── 02_Procesamiento_Videos_YOLO26.ipynb # Detección, segmentación y pose sobre video
├── models/
│   ├── best.pt                       # Pesos del detector final
│   └── inferencia.json               # Confianza elegida y hash de los pesos
├── resultados/
│   ├── comparacion_mejoras/          # Métricas, curvas y matriz del modelo final
│   └── historico_original/           # Primera corrida, como referencia
└── videos/
    ├── input/                        # Videos originales
    └── output/                       # Videos procesados y resumen
```

## Cómo correr los notebooks

### En Google Colab

Es la forma recomendada, porque el entrenamiento necesita GPU.

1. Abrir el notebook en Colab: **Archivo > Abrir notebook > GitHub**, pegar `https://github.com/bpace1/CV-mate-detection` y elegir el notebook.
2. Elegir **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4)**.
3. Ejecutar todo con **Entorno de ejecución > Ejecutar todas**.

La primera celda de código clona el repositorio en `/content/CV-mate-detection` y la segunda instala las dependencias. No hace falta subir archivos ni montar Drive. Los archivos de `/content` se borran al cerrar la sesión, así que al terminar hay que descargar el ZIP que arma la última celda.

**Notebook 01** (entrenamiento): valida las etiquetas, arma el split, entrena los cuatro candidatos, evalúa el elegido en test y exporta `models/best.pt` y `models/inferencia.json`. Al final arma `entregable_entrenamiento.zip`. Entrena cuatro modelos de hasta 80 épocas cada uno, así que en una T4 tarda bastante. Sin GPU se detiene antes de entrenar.

**Notebook 02** (videos): procesa los videos de `videos/input/` con el detector propio, la segmentación y la pose, y escribe los resultados en `videos/output/`. Al final arma `entregable_videos.zip`. Usa los pesos que ya están en `models/`, así que se puede correr sin pasar por el notebook 01. Con una T4, los cuatro videos tardan unos 4 minutos.

Para procesar otros videos, subirlos a `/content/CV-mate-detection/videos/input/` después de la primera celda. Cada video puede durar hasta 2 minutos.

### En una máquina propia

Hace falta Python 3.10 o posterior. Primero se instala PyTorch: con GPU NVIDIA, siguiendo las instrucciones de [pytorch.org](https://pytorch.org/get-started/locally/); sin GPU, la versión para CPU:

```bash
git clone https://github.com/bpace1/CV-mate-detection.git
cd CV-mate-detection
python3 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
jupyter notebook notebooks/
```

El notebook 02 corre en CPU, más lento. El notebook 01 necesita una GPU con CUDA para entrenar. `yolo26n-seg.pt`, `yolo26n-pose.pt` y los pesos base se descargan solos la primera vez.

## Dataset y etiquetas

| ID | Clase | Imágenes | Cajas |
|----|-------|----------|-------|
| 0 | `mate` | 62 | 65 |
| 1 | `termo` | 50 | 51 |

Cada integrante anotó un grupo de fotos en MakeSense y exportó un CSV. `src/unify_annotations.py` asigna el ID según el nombre de la etiqueta, así que el orden de clases de cada export no importa. Para regenerar y validar las etiquetas:

```bash
python3 src/unify_annotations.py
python3 src/validate_dataset.py
```

El split es 63/13/14 imágenes y mantiene juntas las fotos de una misma sesión. Está fijado en `data/split_manifest.csv`, y `python3 src/prepare_dataset.py` arma la carpeta `dataset_yolo26_mate_termo/` que usa `data.yaml`. Las correcciones de etiquetas y el criterio del split están en [docs/dataset_y_entrenamiento.md](docs/dataset_y_entrenamiento.md).

## Estilo de código

El código sigue PEP 8 con líneas de hasta 79 caracteres, y cada función tiene un docstring con formato Google. `pyproject.toml` configura ruff para `src/` y los notebooks:

```bash
ruff check src/ notebooks/
```

## Entregables

| Pide el enunciado | Dónde está |
|-------------------|------------|
| Código fuente y notebooks ejecutados | `notebooks/` y `src/` |
| Dataset | `dataset/`, `data/unified/labels/` y `data/split_manifest.csv` |
| `data.yaml` | `data.yaml` |
| Pesos del modelo final | `models/best.pt` |
| Video original | `videos/input/` |
| Video procesado | `videos/output/` |
| Métricas y gráficos | `resultados/comparacion_mejoras/` |
