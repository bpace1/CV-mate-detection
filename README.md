# Detección, segmentación y pose con YOLO26

Trabajo integrador final de Visión por Computadora 2026, Licenciatura en Ciencia de Datos, Universidad del Gran Rosario (UGR).

Entrenamos un detector YOLO26 para dos clases que no están en COCO, `mate` y `termo`. Después procesamos videos con tres modelos sobre cada fotograma:

1. El detector propio dibuja una caja por cada mate o termo, con la clase y la confianza.
2. `yolo26n-seg.pt` pinta máscaras sobre los objetos COCO. Omite las personas y las máscaras que caen sobre una caja de mate o termo.
3. `yolo26n-pose.pt` dibuja los keypoints y el esqueleto de cada persona, con un ID de seguimiento.

## Comparación revisada del entrenamiento

El notebook 01 compara cuatro configuraciones de YOLO26 con etiquetas corregidas, sesiones agrupadas y un split persistente en `data/split_manifest.csv`. Selecciona modelo y confianza exclusivamente con validación. El notebook 02 lee esa confianza desde `models/inferencia.json`.

Los resultados nuevos se guardan en `resultados/comparacion_mejoras/`; los pesos y resultados originales están en `resultados/historico_original/`. Las métricas antiguas citadas más abajo describen la corrida anterior, con otro split y otras etiquetas, y no son directamente comparables.

Para el entrenamiento revisado, instalar PyTorch compatible con el driver y después `pip install -r requirements-training.txt`. En Colab se conserva el torch/CUDA ya instalado. El notebook descarga los pesos preentrenados al necesitarlos.

Correcciones, criterio de anotación y límites: [docs/mejoras_entrenamiento.md](docs/mejoras_entrenamiento.md).

## Estado de la entrega

Los dos notebooks están ejecutados y tienen conclusiones. Se incluyen los pesos seleccionados, la configuración de inferencia, métricas de la comparación y cuatro videos originales/procesados. El detector elegido es YOLO26n sin mosaic, con confianza 0.25; en test obtuvo mAP50 0.9045 y mAP50–95 0.5933. El test contiene 14 imágenes y 17 objetos, por lo que no demuestra robustez general.

El video04 muestra keypoints, esqueleto y seguimiento de una persona. Persisten falsos positivos de termo sobre la persona y máscaras COCO sobre el mate cuando falla su detección propia. Estas limitaciones están documentadas en los notebooks y en [docs/revision_videos.md](docs/revision_videos.md).

El enunciado pide documentación del proyecto, dataset, entrenamiento, resultados, ejemplos y conclusiones; no fija un formato separado de informe PDF. La documentación está distribuida entre este README, los notebooks y `docs/`. Si la cátedra solicita un informe o presentación aparte, falta preparar ese formato.

### Reconstruir el dataset sin entrenar

El archivo versionado [data.yaml](data.yaml) usa rutas relativas. Después de clonar y desde la raíz:

```bash
python src/prepare_dataset.py
```

El script usa `data/split_manifest.csv`, verifica hashes y conserva exactamente las 63/13/14 imágenes de train/val/test. Copia imágenes y etiquetas a `dataset_yolo26_mate_termo/` y crea allí otro YAML portable; no cambia el split ni entrena. Esa carpeta y los ZIP se regeneran y no se suben duplicados a Git. Las imágenes originales y etiquetas sí están versionadas.

El YAML dentro del ZIP original de Colab conserva la ruta de esa sesión. Para reproducir en otra máquina, usar el YAML del repositorio y el script anterior.

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
| 0 | `mate` | 62 | 65 |
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

Compara YOLO26n y YOLO26s, preentrenados en COCO, sobre nuestras dos clases. Conviene correrlo con GPU (Colab con T4 alcanza).

| Parámetro | Valor |
|-----------|-------|
| Épocas | hasta 80 por candidato, early stopping de 20 |
| Tamaño de imagen | 640 |
| Batch | 16 en Colab; 4 en local |
| Split | 63 train, 13 val, 14 test; sesiones agrupadas |

La exportación copia los pesos elegidos por validación a `models/best.pt` y su umbral a `models/inferencia.json`. Las métricas y gráficos revisados van a `resultados/comparacion_mejoras/`. Los archivos antiguos en la raíz de `resultados/` y en `historico_original/` corresponden a la corrida previa.

Para Colab, ver [docs/ejecutar_en_colab.md](docs/ejecutar_en_colab.md): abrir el notebook actualizado, habilitar GPU y cargar el ZIP completo en su primera celda. No es necesario clonar GitHub ni instalar PyTorch manualmente en Colab.

### Notebook 02: videos

Lee los videos de `videos/input/` (hasta 2 minutos cada uno) y escribe en `videos/output/` un MP4 H.264 por video con las tres capas. También genera `resumen_procesamiento.csv`, con la cantidad de detecciones, máscaras y personas por video. Necesita `models/best.pt`.

El detector propio lee la confianza de `models/inferencia.json` (debe acompañar a los pesos; se verifica su hash). Los otros umbrales son `CONF_SEG=0.45` para la segmentación y `CONF_POSE=0.35` para la pose. Si una máscara COCO cubre una caja de mate o termo en un 50 % o más (`MASK_OVERLAP_SUPPRESS`), no se dibuja.

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
