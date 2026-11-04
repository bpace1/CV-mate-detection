#!/usr/bin/env python3
"""Unifica las anotaciones de los integrantes en un único dataset YOLO.

Lee los CSV exportados por cada integrante (formato MakeSense: caja en
píxeles con esquina superior izquierda, ancho y alto) y genera:

1. ``data/unified/annotations.csv``: todas las cajas con coordenadas en
   píxeles y normalizadas.
2. ``data/unified/labels/*.txt``: una etiqueta YOLO por imagen y
   ``classes.txt``.

El ``data.yaml`` y el split train/val/test los genera el notebook 01.

Las clases se homogeneizan por nombre (``mate`` -> 0, ``termo`` -> 1), lo que
corrige los IDs invertidos del export de Bruno, y se descartan cajas
duplicadas dentro de una misma imagen.

Uso:
    python3 src/unify_annotations.py
"""

import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_ANNOTATIONS_DIR = DATA_DIR / 'raw_annotations'
UNIFIED_DIR = DATA_DIR / 'unified'
LABELS_OUT_DIR = UNIFIED_DIR / 'labels'

CLASS_MAP = {'mate': 0, 'termo': 1}
MIN_IMAGES_PER_CLASS = 50

# Toda anotación debe vivir en uno de estos CSV: el script borra y regenera
# ``labels/``, así que lo que no esté acá se pierde al volver a ejecutarlo.
SOURCES = [
    ('Bruno', RAW_ANNOTATIONS_DIR / 'anotaciones_bruno.csv'),
    ('Franco_B', RAW_ANNOTATIONS_DIR / 'labels_Franco_B.csv'),
    ('Juana', RAW_ANNOTATIONS_DIR / 'labels-juana.csv'),
    ('Franco_C', RAW_ANNOTATIONS_DIR / 'labels_Franco_C.csv'),
]

CSV_FIELDS = [
    'image_name', 'label_name', 'class_id',
    'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
    'x_center', 'y_center', 'norm_width', 'norm_height',
    'image_width', 'image_height', 'author',
]

def clamp(value: float, min_value: float = 0.0,
          max_value: float = 1.0) -> float:
    """Restringe un valor al intervalo ``[min_value, max_value]``.

    Args:
        value: Valor a restringir.
        min_value: Límite inferior.
        max_value: Límite superior.

    Returns:
        ``value`` recortado al intervalo.
    """
    return max(min_value, min(max_value, value))


def format_yolo_coords(x_center: float, y_center: float, width: float,
                       height: float) -> tuple[float, float, float, float]:
    """Redondea una caja YOLO a 6 decimales y la mantiene dentro de la imagen.

    Si el redondeo deja algún borde fuera de ``[0, 1]``, se desplaza el centro
    lo mínimo necesario, sin modificar el tamaño de la caja.

    Args:
        x_center: Centro horizontal normalizado.
        y_center: Centro vertical normalizado.
        width: Ancho normalizado.
        height: Alto normalizado.

    Returns:
        ``(x_center, y_center, width, height)`` redondeados y válidos.
    """
    xc, yc = round(x_center, 6), round(y_center, 6)
    w, h = round(width, 6), round(height, 6)
    if xc + w / 2 > 1.0:
        xc = round(1.0 - w / 2 - 1e-6, 6)
    if xc - w / 2 < 0.0:
        xc = round(w / 2 + 1e-6, 6)
    if yc + h / 2 > 1.0:
        yc = round(1.0 - h / 2 - 1e-6, 6)
    if yc - h / 2 < 0.0:
        yc = round(h / 2 + 1e-6, 6)
    return xc, yc, w, h


def parse_row(row: dict[str, str], author: str) -> dict[str, Any] | None:
    """Convierte una fila del CSV de MakeSense en una anotación normalizada.

    Args:
        row: Fila leída con ``csv.DictReader``.
        author: Integrante que hizo la anotación.

    Returns:
        Anotación con coordenadas en píxeles y normalizadas, o ``None``
        si la etiqueta no es ``mate`` ni ``termo``.
    """
    label = row['label_name'].strip().lower()
    if label not in CLASS_MAP:
        return None
    bx, by = float(row['bbox_x']), float(row['bbox_y'])
    bw, bh = float(row['bbox_width']), float(row['bbox_height'])
    iw, ih = float(row['image_width']), float(row['image_height'])
    return {
        'author': author,
        'label_name': label,
        'class_id': CLASS_MAP[label],
        'bbox_x': bx,
        'bbox_y': by,
        'bbox_width': bw,
        'bbox_height': bh,
        'image_name': row['image_name'].strip(),
        'image_width': int(iw),
        'image_height': int(ih),
        'x_center': clamp((bx + bw / 2.0) / iw),
        'y_center': clamp((by + bh / 2.0) / ih),
        'norm_width': clamp(bw / iw),
        'norm_height': clamp(bh / ih),
    }


def load_annotations(
        sources: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """Lee y desduplica las anotaciones de todos los integrantes.

    Dos cajas de una misma imagen se consideran duplicadas si tienen la misma
    clase y coinciden al redondear centro y tamaño a 2 decimales (1 % de la
    imagen).

    Args:
        sources: Pares ``(autor, ruta_csv)``.

    Returns:
        Anotaciones válidas, en el orden de lectura.
    """
    annotations = []
    seen_boxes = defaultdict(set)
    for author, csv_path in sources:
        if not csv_path.exists():
            print(f'[ALERTA] No se encontró el archivo: {csv_path}')
            continue
        loaded = 0
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                item = parse_row(row, author)
                if item is None:
                    print(f"[{author}] Etiqueta desconocida "
                          f"'{row['label_name']}' en {row['image_name']}, "
                          'se omite.')
                    continue
                signature = (
                    round(item['x_center'], 2), round(item['y_center'], 2),
                    round(item['norm_width'], 2),
                    round(item['norm_height'], 2), item['class_id'],
                )
                if signature in seen_boxes[item['image_name']]:
                    print(f"[DUPLICADO] {item['image_name']} ({author}): "
                          f"caja de {item['label_name']} repetida, se omite.")
                    continue
                seen_boxes[item['image_name']].add(signature)
                annotations.append(item)
                loaded += 1
        print(f'-> {author}: {loaded} anotaciones cargadas.')
    return annotations


def write_unified_csv(annotations: list[dict[str, Any]], path: Path) -> None:
    """Guarda todas las anotaciones en un único CSV.

    Args:
        annotations: Anotaciones devueltas por
            :func:`load_annotations`.
        path: Ruta del CSV de salida.
    """
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(annotations)
    print(f'[OK] CSV unificado: {path} ({len(annotations)} instancias)')


def write_yolo_labels(annotations: list[dict[str, Any]],
                      labels_dir: Path) -> int:
    """Regenera los ``.txt`` YOLO (uno por imagen) y ``classes.txt``.

    Borra antes todos los ``.txt`` existentes en ``labels_dir``.

    Args:
        annotations: Anotaciones devueltas por
            :func:`load_annotations`.
        labels_dir: Carpeta de salida.

    Returns:
        Cantidad de imágenes con etiqueta.
    """
    for txt_file in labels_dir.glob('*.txt'):
        txt_file.unlink()
    (labels_dir / 'classes.txt').write_text('mate\ntermo\n', encoding='utf-8')

    by_image = defaultdict(list)
    for item in annotations:
        by_image[item['image_name']].append(item)
    for image_name, items in by_image.items():
        lines = []
        for item in items:
            xc, yc, w, h = format_yolo_coords(
                item['x_center'], item['y_center'],
                item['norm_width'], item['norm_height'])
            lines.append(f"{item['class_id']} {xc:.6f} {yc:.6f} "
                         f'{w:.6f} {h:.6f}\n')
        txt_path = labels_dir / f'{Path(image_name).stem}.txt'
        txt_path.write_text(''.join(lines), encoding='utf-8')
    print(f'[OK] {len(by_image)} etiquetas YOLO en: {labels_dir}')
    return len(by_image)


def print_summary(annotations: list[dict[str, Any]]) -> None:
    """Muestra cajas e imágenes por clase y avisa si falta alguna.

    Args:
        annotations: Anotaciones devueltas por
            :func:`load_annotations`.
    """
    boxes = Counter(item['label_name'] for item in annotations)
    images = defaultdict(set)
    for item in annotations:
        images[item['label_name']].add(item['image_name'])
    total_images = len({item['image_name'] for item in annotations})

    print('\n' + '=' * 60)
    print('RESUMEN DE DATOS UNIFICADOS')
    print('=' * 60)
    print(f'Imágenes únicas con anotaciones: {total_images}')
    print(f'Instancias totales: {len(annotations)}')
    for name, class_id in CLASS_MAP.items():
        n_images = len(images[name])
        print(f"  * '{name}' (ID {class_id}): {boxes[name]} cajas, "
              f'{n_images} imágenes')
        if n_images < MIN_IMAGES_PER_CLASS:
            print(f'    [ATENCIÓN] faltan {MIN_IMAGES_PER_CLASS - n_images} '
                  'imágenes para el mínimo del enunciado.')
    print('=' * 60)


def unify() -> None:
    """Ejecuta la unificación completa y escribe todas las salidas."""
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compatibilidad: algunos CSV se dejaron en la raíz del repositorio.
    for _, path in SOURCES:
        alt_path = PROJECT_ROOT / path.name
        if not path.exists() and alt_path.exists():
            shutil.copy2(alt_path, path)

    print('=' * 60)
    print('UNIFICACIÓN DE ANOTACIONES')
    print('=' * 60)
    annotations = load_annotations(SOURCES)

    write_unified_csv(annotations, UNIFIED_DIR / 'annotations.csv')
    write_yolo_labels(annotations, LABELS_OUT_DIR)
    print_summary(annotations)


if __name__ == '__main__':
    unify()
