# Pendientes para la entrega

Lo que falta para cumplir el enunciado (`docs/enunciado_2026.pdf`). Actualizado el 4 de noviembre de 2026.

## Hecho

- Dataset. Hay 60 imágenes con mate (61 cajas) y 50 con termo (51 cajas), así que se cumple el mínimo de 50 por clase. Las dos últimas fotos anotadas, `IMG_4040.jpeg` y `termo-47.jpeg`, están en `data/raw_annotations/labels_Franco_C.csv`.
- Código. Los dos notebooks usan YOLO26, y el código pasa `ruff check src/ notebooks/`.
- Re-entrenamiento con YOLO26n. Se corrió en Colab con una T4. Cortó en la época 52 y la mejor fue la 37. Los pesos están en `models/best.pt`, las métricas en `resultados/` y el notebook 01 se subió ejecutado.

## 1. Videos de prueba

El enunciado pide como entrada "un video de hasta 2 minutos de duración, o un link de algún video público", y recomienda "2 o 3 videos cortos de aproximadamente 20 segundos cada uno".

Cada video tiene que mostrar:

- al menos una persona, para que se vea la pose;
- un mate y un termo, para que se vean las cajas del detector propio;
- otros objetos de COCO, como una taza, una botella, una silla o una laptop, para que se vean las máscaras.

Hay que grabar o descargar 2 o 3 videos `.mp4` de unos 20 segundos y guardarlos en `videos/input/`, por ejemplo `video_mate_01.mp4`. Si son descargados, hay que anotar la URL y la licencia.

## 2. Conclusiones del entrenamiento

El enunciado pide documentar la arquitectura, la cantidad de épocas, el batch, el tamaño de imagen, las métricas y los resultados. Va en la sección 10 del notebook 01.

Resultados en test con YOLO26n (9 imágenes, 11 cajas):

| Clase | Precisión | Recall | F1 | mAP50 | mAP50-95 |
|-------|-----------|--------|----|-------|----------|
| mate | 0.808 | 0.606 | 0.692 | 0.734 | 0.539 |
| termo | 0.708 | 0.750 | 0.728 | 0.683 | 0.527 |
| total | 0.758 | 0.678 | 0.716 | 0.708 | 0.533 |

Puntos para el análisis:

- La corrida anterior usó YOLO11n por un error del notebook. En test dio F1 0.87 y mAP50 0.80, mejor que YOLO26n. En validación pasó al revés: YOLO26n llegó a mAP50 0.86 y mAP50-95 0.66. Con 11 cajas en test, cada error mueve el recall de una clase entre 14 y 25 puntos, así que con estos datos no se puede decir cuál de los dos modelos es mejor.
- En la matriz de confusión (`resultados/confusion_matrix.png`, con confianza 0.35) hay 2 mates clasificados como termo, 1 mate y 1 termo no detectados, y 1 falso positivo de mate.
- Desde la época 40, más o menos, la pérdida de validación sube mientras la de entrenamiento sigue bajando. Es sobreajuste, esperable con 72 imágenes de entrenamiento.
- La mejora con más impacto sería sumar fotos, sobre todo con mate y termo juntos, que es donde el modelo los confunde.

## 3. Procesamiento de videos

El enunciado pide que el video procesado integre segmentación de COCO sin personas ni clases nuevas, pose con keypoints y esqueleto, y detección de las clases propias con caja, etiqueta y confianza.

1. Poner los videos en `videos/input/`.
2. Correr `notebooks/02_Procesamiento_Videos_YOLO26.ipynb`. Usa `models/best.pt`.
3. Revisar los videos completos en `videos/output/`. Si aparece una máscara de "cup" o "bottle" encima de un mate o un termo, subir `CONF_CUSTOM` o bajar `MASK_OVERLAP_SUPPRESS`.
4. Escribir las conclusiones de la sección 7 del notebook 02.

## Lista de entrega

- [x] Dataset: `dataset/`, `data/unified/labels/` y el `data.yaml` que genera el notebook 01
- [x] Código: los dos notebooks y `src/`
- [x] Re-entrenamiento con YOLO26
- [x] Pesos y métricas: `models/best.pt` y `resultados/`
- [ ] Videos originales en `videos/input/`
- [ ] Videos procesados en `videos/output/`
- [ ] Conclusiones en los dos notebooks
- [ ] Entrega antes del último día hábil de la anteúltima semana de cursada
