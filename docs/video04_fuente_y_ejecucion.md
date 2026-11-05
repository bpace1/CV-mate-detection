# Video 04: fuente y ejecución

- Fuente: https://www.youtube.com/watch?v=I5JOt0B6iS8
- Título: La forma definitiva de preparar mate: ¡Tips infalibles!
- Canal: Sudamerican Mate.
- Tramo seleccionado por el usuario: 00:00 a 00:25.
- Incorporado el 5 de noviembre de 2026 como `videos/input/video04.mp4`.
- Verificado: 25 segundos exactos, 625 fotogramas a 25 FPS, 1280x720 y decodificación completa. Se revisaron muestras en los segundos 1, 12 y 24.
- Procedencia y hash: `videos/input/video04_fuente.json`. El extractor no informa una licencia específica; no se presupone Creative Commons.

En las muestras aparecen una persona con cabeza y brazos visibles, mates, termo y otros elementos sobre una mesa. El clip complementa los tres anteriores para probar pose. Ya fue procesado en Colab y se recibió la salida completa; la revisión está en docs/revision_videos.md.

## Colab

Abrir la copia actualizada del notebook 02 que acompaña al paquete `CV_mate_termo_Videos_Colab_v4.zip`, habilitar GPU y ejecutar la primera celda. Seleccionar ese ZIP sin descomprimir. Luego ejecutar el resto de las celdas en orden.

El paquete contiene los cuatro videos de entrada, los pesos seleccionados y su configuración. No requiere volver a entrenar. La copia para ejecutar tiene las salidas anteriores limpias; el notebook ejecutado local se conserva como evidencia de la corrida de los tres videos originales.

Al finalizar, descargar `entregable_videos.zip` y el notebook ejecutado. Esta vez el resumen debe incluir `video04.mp4`; comprobar visualmente el esqueleto y la segmentación. La incorporación del video no cambia el filtro de máscaras del notebook.
