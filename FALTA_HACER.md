# Estado de entrega y pendientes

Actualizado el 5 de noviembre de 2026 contra `docs/enunciado_2026.pdf`.

## Entregables disponibles

- [x] Dos clases propias: mate y termo.
- [x] Dataset de 90 fotos: mate en 62 imágenes (65 cajas) y termo en 50 imágenes (51 cajas), con formato YOLO y CSV originales de anotación.
- [x] Correcciones de etiquetas, sesiones identificadas y split fijo 63/13/14 con hashes verificables.
- [x] `data.yaml` portable en la raíz y `src/prepare_dataset.py` para reconstruir el dataset sin reentrenar.
- [x] Entrenamiento por transfer learning con YOLO26; comparación de cuatro configuraciones y selección usando validación.
- [x] Pesos seleccionados en `models/best.pt` y confianza 0.25 en `models/inferencia.json`, unidos por SHA-256.
- [x] Métricas, curvas, matriz, predicciones, calibración e historial de 80 épocas en `resultados/comparacion_mejoras/`.
- [x] Ambos notebooks ejecutados sin errores guardados y con conclusiones.
- [x] Cuatro videos originales y procesados, todos por debajo de dos minutos.
- [x] Integración ejecutada de detección, segmentación COCO y pose; video04 aporta evidencia visual de esqueleto y seguimiento.
- [x] Fuente y tramo 00:00–00:25 del video público registrados en `videos/input/video04_fuente.json` y `docs/video04_fuente_y_ejecucion.md`.
- [x] Documentación del proyecto, datos, entrenamiento, resultados, ejemplos y conclusiones en README, notebooks y docs.

## Pendientes reales antes de cerrar la entrega

1. **Segmentación sobre clases propias.** El enunciado pide excluirlas. El filtro implementado depende de las cajas propias; en muestras de video02 y video04 una máscara COCO cubre el mate cuando falta o es parcial su detección. La exclusión no es consistente. Hay que mejorar y volver a comprobar ese comportamiento para afirmar cumplimiento visual completo; documentarlo no lo corrige.
2. **Falsos positivos y generalización.** Hay detecciones de termo sobre la persona en video04 y mates omitidos. El enunciado no fija un mAP mínimo ni exige detección perfecta, pero estas limitaciones deben quedar explícitas al presentar el trabajo. El test de 14 fotos/17 objetos es pequeño.
3. **Revisión final humana.** Mirar completos los cuatro videos procesados, además de las muestras ya revisadas, y comprobar que la selección final muestre con claridad las tres tareas.
4. **Formato de presentación.** El enunciado no exige expresamente un informe PDF separado. Si la cátedra lo pidió por otro medio, o el grupo quiere una memoria unificada, falta preparar ese informe/presentación; la información técnica ya está documentada.
5. **Envío a la cátedra.** Confirmar el canal y la fecha efectiva, y realizar la entrega. El enunciado refiere al último día hábil de la anteúltima semana de cursada, sin una fecha calendario.

## Referencia de resultados vigentes

Seleccionado: YOLO26n sin mosaic, mAP50–95 de validación 0.8236. Test: precisión macro 0.8750, recall macro 0.8786, F1 macro 0.8737, mAP50 0.9045 y mAP50–95 0.5933. La corrida previa usaba otras etiquetas y otro test; sus métricas no son una comparación controlada.

Los videos suman 2837 fotogramas y 248.7 segundos de procesamiento, aproximadamente 11.4 FPS efectivos incluyendo lectura, inferencia, dibujo y conversión. Es procesamiento offline, sin demostración de tiempo real a los FPS de entrada.

## Organización del repositorio

Los originales del dataset, etiquetas, pesos, notebooks ejecutados, métricas y videos finales se versionan. Los ZIP de exportación, el dataset duplicado generado, las copias de descarga y el respaldo local redundante de los primeros tres videos quedan locales. La corrida original de entrenamiento se conserva como referencia histórica.
