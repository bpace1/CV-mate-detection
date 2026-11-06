# Entrega y rúbrica

Copia de la consigna de entrega del campus. El enunciado completo está en `enunciado_2026.pdf`.

Fecha de entrega: viernes 6 de noviembre de 2026. Grupo 5.

## Materiales a entregar

Hay que entregar todo lo necesario para reproducir y verificar el trabajo:

- El código fuente completo. Puede estar en uno o más notebooks, por ejemplo uno de entrenamiento y otro de inferencia.
- Los notebooks ejecutados, con todas las celdas corridas, para que se vea el proceso completo y los resultados.
- El dataset de entrenamiento. Si se usó Roboflow, alcanza con el fragmento que identifica el dataset.
- El archivo `data.yaml`.
- El archivo `best.pt` con los pesos del modelo final.
- El video original usado como entrada.
- El video final procesado.
- Una visualización correcta de la segmentación, la estimación de pose y las bounding boxes.

El docente tiene que poder abrir todos los archivos. Si se usa Google Drive, se sube todo a una carpeta y se comparte el enlace con permiso de solo lectura. Antes de enviar, hay que comprobar el acceso desde una ventana de incógnito o con otra cuenta.

## Rúbrica (36 puntos)

| Criterio | Puntos | Qué pide el nivel máximo |
|----------|--------|--------------------------|
| Dataset y etiquetado | 4 | Dataset propio con 2 o 3 clases válidas, al menos 50 imágenes por clase, variedad de ejemplos, etiquetado consistente y formato YOLO. |
| Entrenamiento de clases personalizadas | 5 | Transfer learning con YOLO26. Se documentan el modelo, las épocas, el batch, el tamaño de imagen y los demás parámetros. El proceso es reproducible. |
| Evaluación y análisis del modelo | 4 | Métricas de entrenamiento y validación, gráficos, e interpretación del desempeño con fortalezas, errores y limitaciones. |
| Detección de clases personalizadas | 4 | Las clases nuevas se detectan en el video con caja, nombre y confianza, de forma consistente y clara. |
| Segmentación de objetos | 4 | YOLO26 Segmentation sobre objetos COCO, con el umbral definido y sin personas ni clases nuevas. Máscaras claras. |
| Estimación de pose | 4 | YOLO26 Pose sobre las personas del video, con keypoints y esqueleto visibles. |
| Integración y procesamiento final | 6 | Detección, segmentación y pose sobre el mismo video, funcionando juntas y con un resultado claro. |
| Código, documentación y entregables | 4 | Código ordenado y ejecutable, notebooks ejecutados con resultados visibles, `data.yaml`, `best.pt`, dataset, video original y video procesado. La documentación permite reproducir el trabajo. |

Los niveles intermedios de cada criterio están en la página de la tarea en el campus.
