#!/usr/bin/env python3
"""
Script de unificación y validación de anotaciones para el Proyecto Integrador de Visión por Computadora.
Unifica las anotaciones de Bruno, Franco B y Juana resolviendo:
1. Homogeneización de IDs de clase: {0: 'mate', 1: 'termo'}.
   (Bruno tenía invertidos los IDs en su export YOLO: 0=termo, 1=mate).
2. Desduplicación de anotaciones repetidas (ej. IMG_7177.jpg).
3. Normalización y verificación matemática de bounding boxes YOLO.
4. Generación de CSV unificado, carpeta de labels TXT, ZIP unificado y data.yaml preliminar.
"""

import os
import csv
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_ANNOTATIONS_DIR = DATA_DIR / "raw_annotations"
UNIFIED_DIR = DATA_DIR / "unified"
LABELS_OUT_DIR = UNIFIED_DIR / "labels"

CLASS_MAP = {
    'mate': 0,
    'termo': 1
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, val))

def format_yolo_coords(xc, yc, nw, nh):
    xc_r = round(xc, 6)
    yc_r = round(yc, 6)
    w_r = round(nw, 6)
    h_r = round(nh, 6)
    if xc_r + w_r / 2 > 1.0: xc_r = round(1.0 - w_r / 2 - 1e-6, 6)
    if xc_r - w_r / 2 < 0.0: xc_r = round(w_r / 2 + 1e-6, 6)
    if yc_r + h_r / 2 > 1.0: yc_r = round(1.0 - h_r / 2 - 1e-6, 6)
    if yc_r - h_r / 2 < 0.0: yc_r = round(h_r / 2 + 1e-6, 6)
    return xc_r, yc_r, w_r, h_r

def unify():
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Fuentes de datos
    sources = [
        ("Bruno", RAW_ANNOTATIONS_DIR / "anotaciones_bruno.csv"),
        ("Franco_B", RAW_ANNOTATIONS_DIR / "labels_Franco_B.csv"),
        ("Juana", RAW_ANNOTATIONS_DIR / "labels-juana.csv")
    ]
    
    # Comprobar si los archivos están en raw_annotations o en raíz
    for name, path in sources:
        if not path.exists():
            alt_path = PROJECT_ROOT / path.name
            if alt_path.exists():
                shutil.copy2(alt_path, path)
    
    unified_rows = []
    # Estructura por imagen: image_name -> list of annotations
    images_annotations = defaultdict(list)
    images_metadata = {}
    
    seen_boxes_per_image = defaultdict(set)
    duplicate_count = 0

    print("=" * 60)
    print("INICIANDO UNIFICACIÓN DE ANOTACIONES")
    print("=" * 60)

    for author, csv_path in sources:
        if not csv_path.exists():
            print(f"[ALERTA] No se encontró el archivo: {csv_path}")
            continue
            
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                label = row['label_name'].strip().lower()
                img_name = row['image_name'].strip()
                
                if label not in CLASS_MAP:
                    print(f"[{author}] Etiqueta desconocida '{label}' en {img_name}, omitiendo.")
                    continue
                
                cls_id = CLASS_MAP[label]
                bx = float(row['bbox_x'])
                by = float(row['bbox_y'])
                bw = float(row['bbox_width'])
                bh = float(row['bbox_height'])
                iw = float(row['image_width'])
                ih = float(row['image_height'])
                
                # Normalización YOLO
                xc = clamp((bx + bw / 2.0) / iw)
                yc = clamp((by + bh / 2.0) / ih)
                nw = clamp(bw / iw)
                nh = clamp(bh / ih)
                
                # Detección de duplicados muy cercanos (tolerancia 1%)
                box_signature = (round(xc, 2), round(yc, 2), round(nw, 2), round(nh, 2), cls_id)
                if box_signature in seen_boxes_per_image[img_name]:
                    print(f"[DUPLICADO DETECTADO] Imagen {img_name} ({author}) - Caja duplicada con {label}. Se omite.")
                    duplicate_count += 1
                    continue
                
                seen_boxes_per_image[img_name].add(box_signature)
                
                annot_item = {
                    'author': author,
                    'label_name': label,
                    'class_id': cls_id,
                    'bbox_x': bx,
                    'bbox_y': by,
                    'bbox_width': bw,
                    'bbox_height': bh,
                    'image_name': img_name,
                    'image_width': int(iw),
                    'image_height': int(ih),
                    'x_center': xc,
                    'y_center': yc,
                    'norm_width': nw,
                    'norm_height': nh
                }
                
                unified_rows.append(annot_item)
                images_annotations[img_name].append(annot_item)
                images_metadata[img_name] = (int(iw), int(ih))
                count += 1
                
        print(f"-> {author}: {count} anotaciones cargadas exitosamente.")

    # 1. Guardar CSV unificado
    unified_csv_path = UNIFIED_DIR / "annotations.csv"
    fieldnames = [
        'image_name', 'label_name', 'class_id',
        'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
        'x_center', 'y_center', 'norm_width', 'norm_height',
        'image_width', 'image_height', 'author'
    ]
    with open(unified_csv_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in unified_rows:
            writer.writerow(r)
    print(f"\n[OK] CSV unificado generado en: {unified_csv_path}")
    print(f"     Total de instancias: {len(unified_rows)}")

    # 2. Guardar TXT por imagen en formato YOLO
    # Limpiar carpeta labels previa si existiera
    for txt_file in LABELS_OUT_DIR.glob("*.txt"):
        txt_file.unlink()

    # Guardar classes.txt
    classes_file = LABELS_OUT_DIR / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write("mate\ntermo\n")
    print(f"[OK] Archivo classes.txt creado con clases: mate (0), termo (1)")

    for img_name, annots in images_annotations.items():
        stem = Path(img_name).stem
        txt_path = LABELS_OUT_DIR / f"{stem}.txt"
        with open(txt_path, mode='w', encoding='utf-8') as f:
            for a in annots:
                xc_r, yc_r, w_r, h_r = format_yolo_coords(a['x_center'], a['y_center'], a['norm_width'], a['norm_height'])
                f.write(f"{a['class_id']} {xc_r:.6f} {yc_r:.6f} {w_r:.6f} {h_r:.6f}\n")

    print(f"[OK] {len(images_annotations)} archivos .txt de etiquetas YOLO creados en: {LABELS_OUT_DIR}")

    # 3. Generar ZIP unificado
    unified_zip_path = UNIFIED_DIR / "labels_yolo.zip"
    with zipfile.ZipFile(unified_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for txt_file in sorted(LABELS_OUT_DIR.glob("*.txt")):
            z.write(txt_file, txt_file.name)
    print(f"[OK] Archivo ZIP unificado creado en: {unified_zip_path}")

    # 4. Generar data.yaml preliminar
    data_yaml_path = UNIFIED_DIR / "data.yaml"
    data_yaml_content = f"""# YOLO Dataset Configuration - Mate and Termo Detection
# Generado automáticamente por unify_annotations.py

path: {DATA_DIR.resolve()}  # Raíz del dataset
train: images/train
val: images/val
test: images/test

names:
  0: mate
  1: termo
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    print(f"[OK] Archivo data.yaml generado en: {data_yaml_path}")

    # Estadísticas finales
    class_counter = Counter(r['label_name'] for r in unified_rows)
    imgs_per_class = defaultdict(set)
    for r in unified_rows:
        imgs_per_class[r['label_name']].add(r['image_name'])

    print("\n" + "=" * 60)
    print("RESUMEN DE DATOS UNIFICADOS:")
    print("=" * 60)
    print(f"Total imágenes únicas con anotaciones: {len(images_annotations)}")
    print(f"Instancias totales de objetos: {len(unified_rows)}")
    for cname in ['mate', 'termo']:
        print(f"  * Clase '{cname}' (ID {CLASS_MAP[cname]}):")
        print(f"      - Instancias (boxes): {class_counter[cname]}")
        print(f"      - Imágenes únicas: {len(imgs_per_class[cname])}")
    print("=" * 60)

    if len(imgs_per_class['termo']) < 50:
        faltan = 50 - len(imgs_per_class['termo'])
        print(f"\n[ATENCIÓN]: La clase 'termo' tiene {len(imgs_per_class['termo'])} imágenes únicas.")
        print(f"Se requieren al menos 50 según el enunciado del trabajo práctico (faltan {faltan}).")

if __name__ == "__main__":
    unify()

