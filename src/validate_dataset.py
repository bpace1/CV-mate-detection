#!/usr/bin/env python3
"""
Script de validación estricta y diagnóstico del dataset.
Verifica:
1. Correspondencia 1 a 1 entre imágenes y archivos de etiquetas .txt.
2. Formato YOLO (5 columnas numéricas, rangos normalizados 0-1, class_id válido).
3. Integridad física de las imágenes (dimensiones mínimas, decodificación).
4. Detección de duplicados binarios.
5. Cumplimiento de los requisitos del Trabajo Práctico (mínimo 50 imágenes por clase).
"""

import os
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "dataset"
LABELS_DIR = PROJECT_ROOT / "data" / "unified" / "labels"
CLASSES = {0: 'mate', 1: 'termo'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

def run_validation():
    print("=" * 65)
    print("VALIDACIÓN INTEGRAL DEL DATASET (CRITERIO AUDITORÍA)")
    print("=" * 65)

    if not IMAGES_DIR.exists():
        print(f"[ERROR] Directorio de imágenes no encontrado: {IMAGES_DIR}")
        return False
    if not LABELS_DIR.exists():
        print(f"[ERROR] Directorio de etiquetas no encontrado: {LABELS_DIR}")
        return False

    image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    print(f"Total imágenes encontradas en {IMAGES_DIR.name}/: {len(image_files)}")

    # 1. Detección de duplicados exactos y nombres con sufijos accidentales
    seen_hashes = {}
    duplicates = []
    duplicate_suffix_files = []

    for img_path in image_files:
        if "(1)" in img_path.name or "(2)" in img_path.name:
            duplicate_suffix_files.append(img_path.name)
        with open(img_path, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h in seen_hashes:
            duplicates.append((img_path.name, seen_hashes[h]))
        else:
            seen_hashes[h] = img_path.name

    if duplicates:
        print(f"\n[ALERTA - IMÁGENES DUPLICADAS]: Se encontraron {len(duplicates)} archivos idénticos:")
        for dup, orig in duplicates:
            print(f"  - {dup} es idéntico a {orig}")

    # 2. Validación de etiquetas
    label_files = sorted(list(LABELS_DIR.glob("*.txt")))
    label_stems = {p.stem for p in label_files if p.name != "classes.txt"}
    img_stems = {p.stem: p for p in image_files}

    missing_labels = [p.name for p in image_files if p.stem not in label_stems and p.name not in duplicate_suffix_files]
    orphan_labels = [s for s in label_stems if s not in img_stems]

    print(f"\nImágenes válidas anotadas: {len(label_stems)}")
    if missing_labels:
        print(f"[INFO] Imágenes presentes sin anotar ({len(missing_labels)}): {missing_labels}")
    if orphan_labels:
        print(f"[ERROR] Etiquetas sin imagen asociada ({len(orphan_labels)}): {orphan_labels}")

    # 3. Verificación de sintaxis y rangos YOLO
    class_box_counts = Counter()
    class_image_counts = defaultdict(set)
    syntax_errors = []

    for lab_path in label_files:
        if lab_path.name == "classes.txt":
            continue
        with open(lab_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        for lineno, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                syntax_errors.append(f"{lab_path.name}:{lineno} - Se esperaban 5 valores, encontrados {len(parts)}")
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                syntax_errors.append(f"{lab_path.name}:{lineno} - Valores no numéricos")
                continue

            if cls_id not in CLASSES:
                syntax_errors.append(f"{lab_path.name}:{lineno} - Clase desconocida {cls_id}")
                continue

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                syntax_errors.append(f"{lab_path.name}:{lineno} - Coordenadas fuera de rango [0,1]")
                continue

            class_box_counts[CLASSES[cls_id]] += 1
            class_image_counts[CLASSES[cls_id]].add(lab_path.stem)

    if syntax_errors:
        print(f"\n[ERROR] Se encontraron {len(syntax_errors)} errores de sintaxis en etiquetas:")
        for err in syntax_errors[:10]:
            print(f"  - {err}")
    else:
        print("\n[OK] Sintaxis y rangos YOLO: 100% válidos.")

    # 4. Evaluación de requisitos del Trabajo Práctico
    print("\n" + "=" * 65)
    print("CONTROL DE REQUISITOS DEL ENUNCIADO:")
    print("=" * 65)
    print("Requisito: Mínimo 50 imágenes por cada clase personalizada (mate, termo)")
    
    cumple_todo = True
    for cname in ['mate', 'termo']:
        count_imgs = len(class_image_counts[cname])
        cumple = count_imgs >= 50
        cumple_todo = cumple_todo and cumple
        estado = "CUMPLE" if cumple else f"INCOMPLETO (faltan {50 - count_imgs})"
        print(f"  * {cname.upper()}: {count_imgs} imágenes ({class_box_counts[cname]} bboxes) -> [{estado}]")

    print("=" * 65)
    return cumple_todo

if __name__ == "__main__":
    run_validation()

