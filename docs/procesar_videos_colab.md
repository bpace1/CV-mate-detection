# Procesar los videos en Colab

1. Abrir `02_Procesamiento_Videos_YOLO26.ipynb` actualizado en Colab.
2. Habilitar GPU desde Entorno de ejecución > Cambiar tipo de entorno de ejecución.
3. Ejecutar la primera celda y cargar `CV_mate_termo_Videos_Colab.zip` sin descomprimir.
4. Ejecutar las demás celdas en orden. Comprobar que el entorno muestra GPU, experimento `n_sin_mosaic` y confianza `0.25`.
5. Revisar los tres videos procesados completos. En particular, comprobar que las máscaras no tapen mates y termos y que la pose corresponda a personas visibles.
6. Descargar `/content/CV-mate-detection/entregable_videos.zip` desde el panel Archivos. Guardar también el notebook ejecutado desde Archivo > Descargar > Descargar .ipynb.
7. Copiar `videos/output/` del ZIP al proyecto local para revisar la entrega.

El paquete de entrada contiene los pesos nuevos y su configuración, los tres videos, este notebook y las dependencias. No requiere el dataset ni ejecutar otra vez el notebook 01. Si se usa la misma sesión de Colab, se puede conservar la carpeta existente; la celda carga el paquete cuando faltan archivos.

En VS Code con kernel Colab, subir primero el ZIP con Upload to Colab; al terminar, descargar el ZIP de resultados desde Colab > Contents > Download.

## Revisión inicial de los videos

- `video01.mp4`: 278 fotogramas, 30 FPS, 576x1024, aproximadamente 9.27 segundos.
- `video02.mp4`: 1301 fotogramas, aproximadamente 59.95 FPS, 464x832, aproximadamente 21.70 segundos.
- `video03.mp4`: 633 fotogramas, aproximadamente 59.97 FPS, 464x832, aproximadamente 10.56 segundos.

Todos están por debajo de 120 segundos. En las muestras al 20 %, 50 % y 80 % aparecen mate/termo y manos, pero no una persona de cuerpo entero o medio cuerpo. Sirven para probar el detector propio y otros objetos de COCO; para demostrar pose conviene agregar otro video con una persona claramente visible y luego volver a procesar el conjunto.

No se ejecutó inferencia local de los tres modelos. Se recibió la ejecución completa de cuatro videos en Colab y se revisaron muestras visuales; ver docs/revision_videos.md. Los totales del resumen cuentan apariciones por fotograma, no objetos o personas únicos.
