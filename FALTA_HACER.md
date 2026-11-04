# Lista de Tareas Pendientes para la Entrega Final
## Proyecto Integrador: Detección, Segmentación y Pose con YOLO (2026)

Lo que falta para cumplir el enunciado (`docs/enunciado_2026.pdf`).

---

## ~~1. Dataset: Completar las 2 imágenes faltantes de la clase `termo`~~ ✅ COMPLETADO

### Estado final (actualizado 2026-11-04):
* **Clase `mate` (ID 0):** **60 imágenes únicas** (62 bounding boxes) → **CUMPLE** ✅
* **Clase `termo` (ID 1):** **50 imágenes únicas** (52 bounding boxes) → **CUMPLE** ✅

> Se anotaron `IMG_4040.jpeg` y `termo-47.jpeg` con bounding boxes verificados visualmente para mate y termo.
> Labels guardados en `data/unified/labels/`, y CSV (112 registros) actualizados. Las anotaciones están en `data/raw_annotations/labels_Franco_C.csv`.

---

## 2. Videos de Prueba: Grabar o recopilar 2 a 3 videos originales

### Requisito del enunciado:
> *"El sistema desarrollado deberá recibir como entrada: Un video de hasta 2 minutos de duración, o un link de algún video público. Se recomienda preferentemente: 2 o 3 videos cortos de aproximadamente 20 segundos cada uno."*

### Criterio de contenido recomendado para los videos:
Los videos deben contener:
* Al menos una persona (para que el modelo de **Pose Estimation** dibuje keypoints y esqueleto corporal).
* Al menos un mate y un termo en escena (para que el modelo de **Detección Custom** dibuje las bounding boxes de las clases entrenadas).
* Otros objetos del hogar u oficina como botellas, tazas, mesas, sillas, laptops o mochilas (para que el modelo de **Segmentación COCO** aplique máscaras semánticas sobre clases no humanas).

### Acción a realizar:
1. Grabar o descargar 2 o 3 videos en formato `.mp4` de ~20 segundos cada uno.
2. Guardarlos en la carpeta: `videos/input/`
   *(Ejemplos de nombres: `video_mate_01.mp4`, `video_mate_02.mp4`)*.

---

## 3. Entrenamiento: Re-ejecutar en GPU con el dataset completo (60 mate + 50 termo)

### Requisito del enunciado:
> *"Entrenamiento deberá realizarse utilizando: Modelos YOLO, Transfer Learning, Dataset propio generado por el grupo... Entregable: Archivo de best.pt (los pesos entrenados)"*

### Nota:
Las corridas previas de Bruno usaron **YOLO11n**, no YOLO26: el notebook solo tomaba `yolo26n.pt` si el archivo ya estaba en disco. Última corrida (2026-11-04, 120 épocas, cortó en la 73): test P 0.948, R 0.804, F1 0.87, mAP50 0.800, mAP50-95 0.606.
* El notebook ahora fuerza `yolo26n.pt` (Ultralytics >= 8.4), reporta F1 y genera la matriz de confusión con `conf=0.35`. **Hay que re-entrenar.**
* `models/best.pt` es de una corrida anterior (60 épocas); reemplazarlo por el nuevo.
* El test tiene solo 9 imágenes (11 cajas): las métricas varían mucho con un solo error. Mencionarlo en las conclusiones.

### Acción a realizar:
1. Hacer `git pull`.
2. Abrir `notebooks/01_Entrenamiento_YOLO26.ipynb` en una máquina con GPU (Colab T4 o la notebook de Bruno).
3. Ejecutar todas las celdas en orden.
4. Commitear `models/best.pt` y `resultados/`: la última celda los copia sola.

---

## 4. Análisis y Redacción Académica de Resultados (Notebook 01)

### Requisito del enunciado:
> *"Deberán documentar: Arquitectura utilizada, Cantidad de épocas, Batch size, Tamaño de imagen, Métricas obtenidas, Resultados del entrenamiento."*

### Acción a realizar:
Una vez completado el re-entrenamiento en el Notebook 01, redactar en la **Sección 10 (Interpretación y conclusiones)** los valores concretos obtenidos:
1. **Métricas cuantitativas obtenidas en Test:** Precisión, Recall, mAP@0.50 y mAP@0.50:0.95 por clase.
2. **Análisis de la Matriz de Confusión.**
3. **Comentarios de Ingeniería:** Época de convergencia, limitaciones observadas.

---

## 5. Inferencia en Video: Procesamiento Multi-Modelo (Notebook 02)

### Requisito del enunciado:
> *"El video procesado deberá integrar simultáneamente: a) Segmentación (COCO sin personas ni clases nuevas), b) Estimación de Pose (keypoints y esqueleto corporal), c) Detección de Clases Personalizadas (bounding boxes, etiqueta y confianza)."*

### Acción a realizar:
1. Abrir `notebooks/02_Procesamiento_Videos_YOLO26.ipynb`.
2. Asegurar que `models/best.pt` esté disponible y que los videos originales estén en `videos/input/`.
3. Ejecutar el procesamiento completo.
4. Verificar la salida en `videos/output/`.

---

## 6. Empaquetado y Control de Entregables Finales

### Lista de verificación de entrega según el PDF:
* [x] **Dataset:** Imágenes en `dataset/`, etiquetas en `data/unified/labels/` y archivo `data.yaml`. (60 mate + 50 termo ✅)
* [x] **Código Fuente:** Repositorio limpio con los 2 notebooks y scripts auxiliares.
* [ ] **Videos de prueba:** Colocar en `videos/input/` 2-3 videos de ~20 seg.
* [ ] **Re-entrenamiento:** Correr Notebook 01 con YOLO26 y el dataset completo.
* [ ] **Pesos y métricas:** `models/best.pt` y `resultados/` commiteados.
* [ ] **Videos procesados:** Videos procesados en `videos/output/`.
* [ ] **Documentación:** Completar conclusiones con métricas reales del re-entrenamiento.
* [ ] **Fecha de entrega:** Verificar envío antes del último día hábil de la ante-última semana de cursada.
