#!/usr/bin/env python3
"""Valida el dataset unificado antes de entrenar.

Comprueba:

1. Duplicados binarios entre imágenes y copias de descarga (``(1)``, ``(2)``).
2. Correspondencia entre imágenes de ``dataset/`` y etiquetas de
   ``data/unified/labels/``.
3. Formato YOLO de cada línea: 5 columnas, clase válida y coordenadas
   normalizadas.
4. Requisito del enunciado: al menos 50 imágenes por clase.

Uso:
    python3 src/validate_dataset.py

El proceso termina con código 1 si el dataset no cumple los requisitos.
"""

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / 'dataset'
LABELS_DIR = PROJECT_ROOT / 'data' / 'unified' / 'labels'
CLASSES = {0: 'mate', 1: 'termo'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MIN_IMAGES_PER_CLASS = 50


def find_duplicates(image_files: list[Path]) -> tuple[list[tuple[str, str]],
                                                       list[str]]:
    """Busca imágenes idénticas byte a byte y copias de descarga.

    Args:
        image_files: Imágenes a revisar.

    Returns:
        Una tupla ``(duplicados, copias)``. ``duplicados`` tiene pares
        ``(archivo, original)`` con el mismo SHA-256 y ``copias`` los nombres
        con sufijo ``(1)`` o ``(2)``.
    """
    seen_hashes = {}
    duplicates = []
    suffix_copies = []
    for img_path in image_files:
        if '(1)' in img_path.name or '(2)' in img_path.name:
            suffix_copies.append(img_path.name)
        digest = hashlib.sha256(img_path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            duplicates.append((img_path.name, seen_hashes[digest]))
        else:
            seen_hashes[digest] = img_path.name
    return duplicates, suffix_copies


def check_label_line(line: str) -> tuple[int | None, str | None]:
    """Valida una línea de etiqueta YOLO.

    Args:
        line: Línea sin espacios al inicio ni al final.

    Returns:
        ``(class_id, None)`` si la línea es válida, o ``(None, motivo)`` si
        no lo es.
    """
    parts = line.split()
    if len(parts) != 5:
        return None, f'se esperaban 5 valores, hay {len(parts)}'
    try:
        class_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
    except ValueError:
        return None, 'valores no numéricos'
    if class_id not in CLASSES:
        return None, f'clase desconocida {class_id}'
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
            and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None, 'coordenadas fuera de rango [0, 1]'
    return class_id, None


def run_validation() -> bool:
    """Ejecuta todas las comprobaciones e imprime un reporte.

    Returns:
        ``True`` si las etiquetas son válidas y cada clase tiene al menos
        ``MIN_IMAGES_PER_CLASS`` imágenes.
    """
    print('=' * 65)
    print('VALIDACIÓN DEL DATASET')
    print('=' * 65)

    for folder in (IMAGES_DIR, LABELS_DIR):
        if not folder.exists():
            print(f'[ERROR] No existe el directorio: {folder}')
            return False

    image_files = sorted(p for p in IMAGES_DIR.iterdir()
                         if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f'Imágenes en {IMAGES_DIR.name}/: {len(image_files)}')

    duplicates, suffix_copies = find_duplicates(image_files)
    if duplicates:
        print(f'\n[ALERTA] {len(duplicates)} imágenes duplicadas:')
        for dup, orig in duplicates:
            print(f'  - {dup} es idéntica a {orig}')

    label_files = sorted(p for p in LABELS_DIR.glob('*.txt')
                         if p.name != 'classes.txt')
    label_stems = {p.stem for p in label_files}
    image_stems = {p.stem for p in image_files}
    missing_labels = [p.name for p in image_files
                      if p.stem not in label_stems
                      and p.name not in suffix_copies]
    orphan_labels = sorted(label_stems - image_stems)

    print(f'\nImágenes anotadas: {len(label_stems)}')
    if missing_labels:
        print(f'[INFO] Imágenes sin anotar ({len(missing_labels)}): '
              f'{missing_labels}')
    if orphan_labels:
        print(f'[ERROR] Etiquetas sin imagen ({len(orphan_labels)}): '
              f'{orphan_labels}')

    box_counts = Counter()
    image_counts = defaultdict(set)
    syntax_errors = []
    for label_path in label_files:
        lines = [line.strip() for line in
                 label_path.read_text(encoding='utf-8').splitlines()
                 if line.strip()]
        for lineno, line in enumerate(lines, start=1):
            class_id, error = check_label_line(line)
            if error:
                syntax_errors.append(f'{label_path.name}:{lineno} - {error}')
                continue
            box_counts[CLASSES[class_id]] += 1
            image_counts[CLASSES[class_id]].add(label_path.stem)

    if syntax_errors:
        print(f'\n[ERROR] {len(syntax_errors)} errores en etiquetas:')
        for error in syntax_errors[:10]:
            print(f'  - {error}')
    else:
        print('\n[OK] Sintaxis y rangos YOLO válidos.')

    print('\n' + '=' * 65)
    print(f'REQUISITO: mínimo {MIN_IMAGES_PER_CLASS} imágenes por clase')
    print('=' * 65)
    meets_minimum = True
    for name in CLASSES.values():
        n_images = len(image_counts[name])
        ok = n_images >= MIN_IMAGES_PER_CLASS
        meets_minimum = meets_minimum and ok
        status = ('CUMPLE' if ok else
                  f'INCOMPLETO (faltan {MIN_IMAGES_PER_CLASS - n_images})')
        print(f'  * {name.upper()}: {n_images} imágenes '
              f'({box_counts[name]} cajas) -> [{status}]')
    print('=' * 65)
    return meets_minimum and not syntax_errors and not orphan_labels


if __name__ == '__main__':
    sys.exit(0 if run_validation() else 1)
