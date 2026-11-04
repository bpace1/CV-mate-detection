# Lista de Tareas Pendientes para la Entrega Final
## Proyecto Integrador: Detección, Segmentación y Pose con YOLO (2026)

Este documento detalla **con precisión técnica y académica todo lo que falta hacer** para completar el 100% de los requisitos exigidos en el enunciado oficial de la cátedra (`Proyecto Integrador Visión por Computadora - 2026.pdf`), además de la mera ejecución de los notebooks.

---

## 1. Dataset: Completar las 2 imágenes faltantes de la clase `termo`

### Estado actual de la auditoría:
* **Clase `mate` (ID 0):** **58 imágenes únicas** (59 bounding boxes) $\rightarrow$ **CUMPLE** (supera el mínimo de 50).
* **Clase `termo` (ID 1):** **48 imágenes únicas** (49 bounding boxes) $\rightarrow$ **INCOMPLETO** (faltan 2 imágenes únicas).

### Requisito del enunciado:
> *"El dataset deberá contener al menos 50 imágenes por cada clase personalizada."*

### Acción a realizar:
1. En la carpeta `dataset/` se identificaron 2 imágenes de alta calidad que quedaron sin anotar:
   * [`IMG_4040.jpeg`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/dataset/IMG_4040.jpeg) (perteneciente a la serie de Bruno).
   * [`termo-47.jpeg`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/dataset/termo-47.jpeg) (imagen aislada de un termo).
2. **Tarea pendiente:**
   * Abrir [MakeSense.ai](https://www.makesense.ai/) (o CVAT / Roboflow).
   * Etiquetar la caja de termo en esas dos imágenes (o en dos fotografías nuevas de termos).
   * Generar los archivos `IMG_4040.txt` y `termo-47.txt` en formato YOLO (`1 <x_center> <y_center> <width> <height>`).
   * Guardarlos en `data/unified/labels/` (y agregar sus registros a `data/unified/annotations.csv`).
   * Ejecutar nuevamente el script de validación para confirmar el cumplimiento estricto:
     ```bash
     python3 src/validate_dataset.py
     ```

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
2. Guardarlos en la carpeta:
   [`videos/input/`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/videos/input/)
   *(Ejemplos de nombres: `video_mate_01.mp4`, `video_mate_02.mp4`)*.

---

## 3. Entrenamiento: Ejecución en GPU y obtención de `best.pt`

### Requisito del enunciado:
> *"Entrenamiento deberá realizarse utilizando: Modelos YOLO, Transfer Learning, Dataset propio generado por el grupo... Entregable: Archivo de best.pt (los pesos entrenados)"*

### Acción a realizar:
1. Abrir [`notebooks/01_Entrenamiento_YOLO26.ipynb`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/notebooks/01_Entrenamiento_YOLO26.ipynb) en **Google Colab** (con acelerador GPU habilitado: *Entorno de ejecución $\rightarrow$ Cambiar tipo de entorno $\rightarrow$ T4 GPU*).
2. Ejecutar secuencialmente todas las celdas:
   * Verificación del dataset y split reproducible (80% train / 10% val / 10% test).
   * Entrenamiento con `yolo11n.pt` / `yolov8n.pt` durante 60 épocas.
   * Evaluación cuantitativa final sobre el conjunto de test.
3. Descargar el archivo resultante `best.pt` desde la carpeta de entrenamiento y colocar una copia en:
   [`models/best.pt`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/models/best.pt)

---

## 4. Análisis y Redacción Académica de Resultados (Notebook 01)

### Requisito del enunciado:
> *"Deberán documentar: Arquitectura utilizada, Cantidad de épocas, Batch size, Tamaño de imagen, Métricas obtenidas, Resultados del entrenamiento."*

### Acción a realizar:
Una vez completado el entrenamiento en el Notebook 01, redactar en la **Sección 10 (Interpretación y conclusiones)** los valores concretos obtenidos:
1. **Métricas cuantitativas obtenidas en Test:**
   * Precisión global y por clase (`mate` vs `termo`).
   * Recall global y por clase.
   * $\text{mAP}@0.50$ y $\text{mAP}@0.50:0.95$.
2. **Análisis de la Matriz de Confusión:**
   * ¿Hubo falsos positivos de fondo confundidos con mates?
   * ¿Hubo confusión entre mates y termos en ángulos o distancias complejas?
3. **Comentarios de Ingeniería:**
   * Época en la que se alcanzó el mejor punto de convergencia (early stopping).
   * Limitaciones observadas (ej. reflejos en termos de acero inoxidable, mates de distintos materiales, oclusiones parciales).

---

## 5. Inferencia en Video: Procesamiento Multi-Modelo (Notebook 02)

### Requisito del enunciado:
> *"El video procesado deberá integrar simultáneamente: a) Segmentación (COCO sin personas ni clases nuevas), b) Estimación de Pose (keypoints y esqueleto corporal), c) Detección de Clases Personalizadas (bounding boxes, etiqueta y confianza)."*

### Acción a realizar:
1. Abrir [`notebooks/02_Procesamiento_Videos_YOLO26.ipynb`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/notebooks/02_Procesamiento_Videos_YOLO26.ipynb).
2. Asegurar que `models/best.pt` esté disponible y que los videos originales estén en `videos/input/`.
3. Ejecutar el procesamiento de video completo:
   * El script procesará fotograma a fotograma combinando:
     * El detector entrenado (`best.pt`).
     * El segmentador COCO (`yolo11n-seg.pt`).
     * El modelo de pose (`yolo11n-pose.pt` con ByteTrack persistente).
   * Se aplicará la supresión de solapamiento para que los termos y mates no sean tapados por máscaras COCO erróneas.
4. Verificar la salida en [`videos/output/`](file:///home/bd/Documents/UGR%20-%20Lic.%20Cs%20de%20Datos/CV/CV-mate-detection/videos/output/):
   * Comprobar que los videos finales se reproduzcan fluidamente con códec H.264.
   * Revisar el cuadro de resumen de detecciones (`resumen_procesamiento.csv`).

---

## 6. Empaquetado y Control de Entregables Finales

### Lista de verificación de entrega según el PDF:
* [ ] **Código Fuente:** Repositorio limpio con los 2 notebooks y scripts auxiliares.
* [ ] **Dataset:** Imágenes en `dataset/`, etiquetas en `data/unified/labels/` y archivo `data.yaml`.
* [ ] **Pesos entrenados:** Archivo `best.pt`.
* [ ] **Videos:** Videos originales (`videos/input/`) y videos procesados (`videos/output/`).
* [ ] **Documentación:** README completo con análisis de resultados y conclusiones.
* [ ] **Fecha de entrega:** Verificar envío antes del último día hábil de la ante-última semana de cursada.

