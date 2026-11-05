"""Splits persistentes y métricas operativas para los experimentos YOLO26."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def assign_groups(records: list[dict], source_file: Path) -> None:
    """Asigna las sesiones revisadas; conserva prefijos de videos restantes.

    Args:
        records: Registros validados por el notebook.
        source_file: CSV con image_name, group y reason.
    """
    with source_file.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row['image_name']: row['group'] for row in rows}
    if len(mapping) != len(rows):
        raise ValueError('Hay imágenes repetidas en source_groups.csv.')
    for record in records:
        name = record['image'].name
        record['group'] = mapping.get(name, name.split('__')[0])


def persistent_split(records: list[dict], manifest: Path,
                     seed: int = 42) -> dict[str, list[dict]]:
    """Carga o fija un split 70/15/15 aproximado, independiente del modelo.

    Busca equilibrio de tamaño y presencia de clases sin separar sesiones.
    Solo usa etiquetas y grupos, nunca predicciones ni métricas del modelo.
    Un manifiesto existente se conserva; cambios de datos requieren revisarlo.

    Args:
        records: Registros con image, label, group y class_ids.
        manifest: CSV persistente que fija los tres subconjuntos.
        seed: Semilla de las propuestas aleatorias.

    Returns:
        Registros por split.
    """
    names = ('train', 'val', 'test')
    inputs = {}
    for record in records:
        inputs[record['image'].name] = {
            'image_name': record['image'].name,
            'group': record['group'],
            'image_sha256': hashlib.sha256(
                record['image'].read_bytes()).hexdigest(),
            'label_sha256': hashlib.sha256(
                record['label'].read_bytes()).hexdigest(),
        }
    if manifest.exists():
        with manifest.open(encoding='utf-8', newline='') as handle:
            saved = list(csv.DictReader(handle))
        if len(saved) != len(inputs):
            raise ValueError('Cambió el dataset: revisar split_manifest.csv.')
        assignments = {}
        for row in saved:
            key = row['image_name']
            if key in assignments or key not in inputs:
                raise ValueError('Manifiesto duplicado o con imágenes ajenas.')
            if any(row[k] != value for k, value in inputs[key].items()):
                raise ValueError(f'Cambió {key}: revisar el manifiesto.')
            if row['split'] not in names:
                raise ValueError('Split desconocido en el manifiesto.')
            assignments[key] = row['split']
    else:
        grouped = defaultdict(list)
        for record in records:
            grouped[record['group']].append(record)
        group_names = sorted(grouped)
        rng = np.random.default_rng(seed)
        targets = np.array([0.70, 0.15, 0.15])
        total_classes = np.array([
            sum(cls in rec['class_ids'] for rec in records)
            for cls in (0, 1)
        ])
        best_score, best_parts = float('inf'), None
        for _ in range(3000):
            shuffled = list(rng.permutation(group_names))
            sizes = np.array([len(grouped[g]) for g in shuffled])
            cumulative = np.cumsum(sizes)
            cut1 = int(np.argmin(abs(cumulative - len(records) * .70))) + 1
            cut2 = int(np.argmin(abs(cumulative - len(records) * .85))) + 1
            if not (0 < cut1 < cut2 < len(shuffled)):
                continue
            parts = [shuffled[:cut1], shuffled[cut1:cut2], shuffled[cut2:]]
            subsets = [[rec for g in part for rec in grouped[g]]
                       for part in parts]
            counts = np.array([
                [sum(cls in r['class_ids'] for r in part) for cls in (0, 1)]
                for part in subsets
            ])
            # Evita holdouts con solo uno o dos ejemplos por clase.
            if np.any(counts < np.array([[10], [4], [4]])):
                continue
            fractions = np.array([len(p) for p in subsets]) / len(records)
            score = float(np.abs(fractions - targets).sum())
            score += float(np.abs(
                counts / total_classes - targets[:, None]).mean())
            if score < best_score:
                best_score, best_parts = score, parts
        if best_parts is None:
            raise ValueError('No hay split viable: agregar fuentes diversas.')
        group_split = {g: name for name, part in zip(names, best_parts,
                                                   strict=True)
                       for g in part}
        assignments = {r['image'].name: group_split[r['group']]
                       for r in records}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'image_name', 'group', 'image_sha256', 'label_sha256', 'split',
            ])
            writer.writeheader()
            writer.writerows({**row, 'split': assignments[key]}
                             for key, row in sorted(inputs.items()))
    splits = {name: [r for r in records
                     if assignments[r['image'].name] == name]
              for name in names}
    group_sets = [{r['group'] for r in splits[name]} for name in names]
    if any(group_sets[i] & group_sets[j]
           for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Una sesión aparece en dos splits.')
    return splits


def collect_predictions(model, subset: list[dict], imgsz: int,
                        device) -> list[dict]:
    """Predice una sola vez para evaluar distintos umbrales de confianza.

    Args:
        model: Detector Ultralytics ya cargado.
        subset: Registros del split a procesar.
        imgsz: Resolución de inferencia.
        device: Dispositivo Ultralytics.

    Returns:
        Predicciones y etiquetas con cajas xyxy normalizadas por imagen.
    """
    output = []
    for record in subset:
        prediction = model.predict(
            str(record['image']), imgsz=imgsz, device=device, conf=0.001,
            verbose=False,
        )[0]
        labels = []
        for line in record['label'].read_text().splitlines():
            if line.strip():
                cls, x, y, w, h = map(float, line.split())
                labels.append([cls, x - w / 2, y - h / 2,
                               x + w / 2, y + h / 2])
        boxes = prediction.boxes
        detections = np.column_stack((
            boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(),
            boxes.xyxyn.cpu().numpy(),
        ))
        output.append({'image': record['image'].name,
                       'labels': np.array(labels).reshape(-1, 5),
                       'detections': detections})
    return output


def operational_metrics(predictions: list[dict], confidence: float,
                        iou_threshold: float = .5) -> dict:
    """Cuenta detecciones con matching 1:1 por IoU y umbral fijo.

    El matching ignora la clase para que una confusión entre mate y termo
    aparezca fuera de la diagonal. Fila=predicho, columna=real; fondo=2.

    Args:
        predictions: Predicciones de collect_predictions.
        confidence: Umbral de confianza.
        iou_threshold: IoU mínimo para asociar una caja a un objeto.

    Returns:
        Matriz, P/R/F1 por clase y sus medias macro, y errores por imagen.
    """
    matrix = np.zeros((3, 3), dtype=int)
    errors = []
    for item in predictions:
        detections = item['detections']
        detections = detections[detections[:, 1] >= confidence]
        labels = item['labels']
        matches = []
        for li, label in enumerate(labels):
            for di, detection in enumerate(detections):
                a, b = label[1:], detection[2:]
                wh = np.maximum(0, np.minimum(a[2:], b[2:])
                                - np.maximum(a[:2], b[:2]))
                intersection = float(wh.prod())
                union = float(np.prod(a[2:] - a[:2])
                              + np.prod(b[2:] - b[:2]) - intersection)
                overlap = intersection / union if union > 0 else 0
                if overlap >= iou_threshold:
                    matches.append((overlap, li, di))
        used_labels, used_detections = set(), set()
        before = matrix.copy()
        for _, li, di in sorted(matches, reverse=True):
            if li in used_labels or di in used_detections:
                continue
            used_labels.add(li)
            used_detections.add(di)
            matrix[int(detections[di, 0]), int(labels[li, 0])] += 1
        for li, label in enumerate(labels):
            if li not in used_labels:
                matrix[2, int(label[0])] += 1
        for di, detection in enumerate(detections):
            if di not in used_detections:
                matrix[int(detection[0]), 2] += 1
        delta = matrix - before
        mistakes = int(delta.sum() - np.trace(delta[:2, :2]))
        if mistakes:
            errors.append({'image': item['image'], 'errors': mistakes,
                           'matrix': delta.tolist()})
    tp = np.diag(matrix)[:2]
    precision = np.divide(tp, matrix[:2].sum(axis=1),
                          out=np.zeros(2), where=matrix[:2].sum(axis=1) > 0)
    recall = np.divide(tp, matrix[:, :2].sum(axis=0),
                       out=np.zeros(2), where=matrix[:, :2].sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(2), where=precision + recall > 0)
    return {'confidence': confidence, 'iou': iou_threshold,
            'matrix': matrix.tolist(), 'precision': precision.tolist(),
            'recall': recall.tolist(), 'f1': f1.tolist(),
            'macro_precision': float(precision.mean()),
            'macro_recall': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'errors': errors}


def calibrate_confidence(predictions: list[dict], output: Path) -> float:
    """Elige confianza por F1 macro en validación y registra el barrido.

    Args:
        predictions: Predicciones exclusivamente de validación.
        output: CSV destino del barrido de umbrales.

    Returns:
        Umbral seleccionado (menor confianza en caso de empate).
    """
    rows = []
    for value in np.round(np.arange(.05, .81, .05), 2):
        metrics = operational_metrics(predictions, float(value))
        rows.append({key: metrics[key] for key in (
            'confidence', 'macro_precision', 'macro_recall', 'macro_f1',
        )})
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return max(rows, key=lambda row: row['macro_f1'])['confidence']


def save_predictions(predictions: list[dict], path: Path) -> None:
    """Guarda las cajas para que la evaluación sea auditable.

    Args:
        predictions: Predicciones de collect_predictions.
        path: JSON de salida.
    """
    serializable = [{key: value.tolist() if isinstance(value, np.ndarray)
                     else value for key, value in item.items()}
                    for item in predictions]
    path.write_text(json.dumps(serializable, indent=2), encoding='utf-8')
