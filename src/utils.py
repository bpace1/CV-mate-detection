"""
Módulo de utilidades para procesamiento de imágenes, video y visualización
en el Proyecto Integrador de Visión por Computadora (YOLO Mate & Termo Detection).
"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

# Configuración de Clases Personalizadas
CLASSES_CUSTOM = {0: 'mate', 1: 'termo'}
COLORS_CUSTOM = {
    'mate': (30, 230, 30),     # Verde
    'termo': (40, 160, 255)    # Azul / Naranja suave
}

# Paleta de colores para segmentación de clases COCO
COLORS_SEG = [
    (205, 130, 40),
    (180, 60, 180),
    (40, 170, 170),
    (110, 210, 60),
    (200, 80, 80),
    (60, 120, 220),
    (220, 190, 40)
]

# Esqueleto COCO para visualización de Keypoints de Pose
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Cabeza y rostro
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Brazos y hombros
    (5, 11), (6, 12), (11, 12),               # Tronco
    (11, 13), (13, 15), (12, 14), (14, 16)    # Piernas
]

def get_recommended_model_names() -> Dict[str, str]:
    """
    Retorna los nombres oficiales de los modelos recomendados según Ultralytics.
    Mapea el nombre simbólico 'YOLO26' del enunciado a la arquitectura real disponible (YOLO11 o YOLOv8).
    """
    return {
        'detect_base': 'yolo11n.pt',
        'segment': 'yolo11n-seg.pt',
        'pose': 'yolo11n-pose.pt',
        'fallback_detect_base': 'yolov8n.pt',
        'fallback_segment': 'yolov8n-seg.pt',
        'fallback_pose': 'yolov8n-pose.pt',
    }

def detect_environment(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Detecta de forma transparente si la ejecución ocurre en Google Colab o en entorno local,
    resolviendo las rutas correspondientes del repositorio.
    """
    import sys
    import os

    in_colab = 'google.colab' in sys.modules or os.path.exists('/content')
    
    if in_colab:
        drive_path = Path('/content/drive/MyDrive/Proyecto_Integrador_YOLO26')
        colab_repo_path = Path('/content/CV-mate-detection')
        if drive_path.exists():
            root = drive_path
        elif colab_repo_path.exists():
            root = colab_repo_path
        else:
            root = Path.cwd()
    else:
        root = base_dir or Path.cwd()
        if (root / 'notebooks').exists():
            pass
        elif root.name == 'notebooks':
            root = root.parent

    paths = {
        'in_colab': in_colab,
        'root': root,
        'dataset_images': root / 'dataset',
        'unified_data': root / 'data' / 'unified',
        'unified_labels': root / 'data' / 'unified' / 'labels',
        'unified_csv': root / 'data' / 'unified' / 'annotations.csv',
        'data_yaml': root / 'data' / 'unified' / 'data.yaml',
        'runs_dir': root / 'entrenamientos',
        'best_model': root / 'entrenamientos' / 'mate_termo_yolo26' / 'weights' / 'best.pt',
        'alt_best_model': root / 'models' / 'best.pt',
        'videos_in': root / 'videos' / 'input',
        'videos_out': root / 'videos' / 'output'
    }
    return paths

def bbox_yolo_to_xyxy(x_center: float, y_center: float, width: float, height: float,
                      img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Convierte coordenadas relativas YOLO [0, 1] a píxeles absolutos (x1, y1, x2, y2).
    """
    w = width * img_width
    h = height * img_height
    x1 = int(round((x_center * img_width) - (w / 2.0)))
    y1 = int(round((y_center * img_height) - (h / 2.0)))
    x2 = int(round(x1 + w))
    y2 = int(round(y1 + h))
    return max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

