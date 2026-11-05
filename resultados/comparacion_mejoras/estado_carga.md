# Resultados completos

Revisado el 5 de noviembre de 2026. Se recibieron `entregable_entrenamiento.zip` y el notebook 01 ejecutado en Colab. Sus 13 celdas de código tienen número de ejecución y no contienen errores guardados.

Se incorporaron los 22 archivos originales de resultados a esta carpeta. Los JSON reconstruidos durante la descarga parcial fueron reemplazados por los originales del ZIP, cuya equivalencia se comprobó antes de reemplazarlos. No quedan archivos pendientes de la exportación.

## Comprobaciones

- Integridad CRC del ZIP y coincidencia de los archivos extraídos.
- Hash de los pesos coincidente con `models/inferencia.json` y con los pesos del ZIP.
- Manifiesto local idéntico al exportado y hash coincidente con la configuración.
- Predicciones guardadas de validación y test correspondientes a sus imágenes y etiquetas locales.
- Matriz y P/R/F1 de test recalculados desde las predicciones: coinciden con el reporte.
- Barrido de confianza recalculado en validación: selecciona 0.25.
- Historial de entrenamiento con 80 épocas.

Modelo: YOLO26n sin mosaic. Test: 14 fotos, 17 objetos, mAP50 0.904536, mAP50-95 0.593311 y F1 macro 0.873684. La matriz tiene 9 mates y 6 termos correctos, 1 mate confundido con termo, 1 termo omitido y 1 falso positivo de termo.

No se volvió a entrenar ni ejecutar inferencia local. Ya se recibieron cuatro videos originales/procesados y el notebook 02 ejecutado. La revisión visual y las limitaciones están en `docs/revision_videos.md`.

La raíz de `resultados/` y `historico_original/` conservan la corrida anterior. Los resultados de esta corrida están en `comparacion_mejoras/`. Las copias de descarga duplicadas y la captura se conservan como material auxiliar; el PNG original de curvas es `results.png`.
