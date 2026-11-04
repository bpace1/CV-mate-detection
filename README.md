# Detección, segmentación y pose con YOLO26

Trabajo integrador final de Visión por Computadora 2026, Licenciatura en Ciencia de Datos, Universidad del Gran Rosario (UGR).

Entrenamos un detector YOLO26 para dos clases que no están en COCO, `mate` y `termo`. Después procesamos videos con tres modelos sobre cada fotograma:

1. El detector propio dibuja una caja por cada mate o termo, con la clase y la confianza.
2. `yolo26n-seg.pt` pinta máscaras sobre los objetos COCO. Omite las personas y las máscaras que caen sobre una caja de mate o termo.
3. `yolo26n-pose.pt` dibuja los keypoints y el esqueleto de cada persona, con un ID de seguimiento.

## Integrantes

- Bruno Pace
- Franco Bianciotto
- Franco Calcia
- Tomas Avecilla
- Juana Zorzolo

## Estructura del repositorio

```text
CV-mate-detection/
├── README.md
├── FALTA_HACER.md                    # Tareas pendientes para la entrega
├── requirements.txt
├── pyproject.toml                    # Configuración de ruff
├── docs/
│   └── enunciado_2026.pdf            # Enunciado de la cátedra
├── dataset/                          # 90 fotos originales
├── data/
│   ├── raw_annotations/              # CSV de MakeSense de cada integrante
│   └── unified/
│       ├── annotations.csv           # Todas las cajas en un solo CSV
│       └── labels/                   # Un .txt YOLO por imagen y classes.txt
├── src/
│   ├── unify_annotations.py          # Genera data/unified/ a partir de raw_annotations/
│   └── validate_dataset.py           # Valida etiquetas y el mínimo de 50 imágenes por clase
├── notebooks/
│   ├── 01_Entrenamiento_YOLO26.ipynb # Split, entrenamiento y evaluación en test
│   └── 02_Procesamiento_Videos_YOLO26.ipynb # Detección, segmentación y pose sobre video
├── models/
│   └── best.pt                       # Pesos entrenados
├── resultados/                       # Métricas y gráficos de la evaluación en test
└── videos/
    ├── input/                        # Videos originales
    └── output/                       # Videos procesados
```

El notebook 01 genera `dataset_yolo26_mate_termo/` (el split train/val/test) y `entrenamientos/` (las corridas de Ultralytics). Ninguna de las dos se versiona.

## Clases y etiquetas

| ID | Clase | Imágenes | Cajas |
|----|-------|----------|-------|
| 0 | `mate` | 60 | 61 |
| 1 | `termo` | 50 | 51 |

Cada integrante anotó un grupo de fotos en MakeSense y exportó un CSV. En el export de Bruno los IDs estaban invertidos (0 era termo y 1 era mate). `src/unify_annotations.py` asigna el ID según el nombre de la etiqueta, así que el orden de cada export no importa. El script también descarta las cajas repetidas dentro de una imagen; había una en `IMG_7177.jpg`.

Para regenerar y validar las etiquetas:

```bash
python3 src/unify_annotations.py
python3 src/validate_dataset.py
```

`unify_annotations.py` borra y vuelve a escribir `data/unified/labels/`. Cualquier anotación nueva tiene que estar en un CSV de `data/raw_annotations/` y en la lista `SOURCES` del script.

## Instalación

Hace falta Python 3.10 o posterior.

```bash
python3 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` instala PyTorch con CUDA. En una máquina sin GPU NVIDIA conviene instalar antes la versión para CPU:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## Ejecución

### Notebook 01: entrenamiento

Entrena `yolo26n.pt`, preentrenado en COCO, sobre nuestras dos clases. Conviene correrlo con GPU (Colab con T4 alcanza).

| Parámetro | Valor |
|-----------|-------|
| Épocas | hasta 120, con early stopping a las 15 sin mejora |
| Tamaño de imagen | 640 |
| Batch | 16 |
| Split | 80 % train, 10 % val, 10 % test |

La última celda copia `best.pt` a `models/` y las métricas y gráficos a `resultados/`:

- `metricas_test.json` y `metricas_por_clase.csv`: precisión, recall, F1, mAP50 y mAP50-95 en test.
- `BoxPR_curve.png` y `BoxF1_curve.png`: curvas precisión-recall y F1 según la confianza.
- `confusion_matrix.png`: matriz de confusión con confianza 0.35. Con el umbral por defecto de Ultralytics (0.001) la columna de fondo se llena de predicciones casi sin confianza y la matriz no se puede leer.
- `results.png` y `results.csv`: pérdidas y métricas de validación por época.

Para correrlo en Colab, agregar una celda al principio con:

```python
!git clone https://github.com/bpace1/CV-mate-detection.git /content/CV-mate-detection
%cd /content/CV-mate-detection/notebooks
```

Esa celda no va en el notebook que se entrega.

### Notebook 02: videos

Lee los videos de `videos/input/` (hasta 2 minutos cada uno) y escribe en `videos/output/` un MP4 H.264 por video con las tres capas. También genera `resumen_procesamiento.csv`, con la cantidad de detecciones, máscaras y personas por video. Necesita `models/best.pt`.

Los umbrales están en la celda de modelos: `CONF_CUSTOM=0.35` para el detector propio, `CONF_SEG=0.45` para la segmentación y `CONF_POSE=0.35` para la pose. Si una máscara COCO cubre una caja de mate o termo en un 50 % o más (`MASK_OVERLAP_SUPPRESS`), no se dibuja.

## Estilo de código

El código sigue PEP 8 con líneas de hasta 79 caracteres, y cada función tiene un docstring con formato Google. `pyproject.toml` configura ruff para `src/` y los notebooks:

```bash
ruff check src/ notebooks/
```

## Entregables

Según el enunciado, la entrega incluye:

1. Los dos notebooks ejecutados y los scripts de `src/`.
2. El dataset: `dataset/`, `data/unified/labels/` y el `data.yaml` que genera el notebook 01.
3. Los pesos entrenados, `models/best.pt`.
4. Los videos originales y los procesados.
5. La documentación del dataset, el entrenamiento, las métricas y las conclusiones.

Lo que falta está en [FALTA_HACER.md](FALTA_HACER.md).
