# Revisión de los cuatro videos procesados

Revisado el 5 de noviembre de 2026. La corrida fue ejecutada en Colab por el usuario. Se incorporaron los seis archivos originales de `entregable_videos.zip` a `videos/output/`; la corrida anterior de tres videos se conservó en `resultados/historico_videos_3/`.

## Integridad y configuración

El ZIP pasa su comprobación CRC; los seis archivos locales coinciden con los originales. Los hashes de los cuatro videos de entrada y los pesos coinciden con `configuracion_procesamiento.json`. Se decodificaron completos los cuatro videos de salida. El cuarto mantiene 625 fotogramas a 25 FPS: 25 segundos.

Detector: YOLO26n `n_sin_mosaic`, confianza 0.25. Segmentación: `yolo26n-seg.pt`, confianza 0.45. Pose: `yolo26n-pose.pt`, confianza 0.35. No se volvió a ejecutar inferencia local ni se cambiaron umbrales.

## Revisión visual de video04

Se inspeccionaron muestras en los segundos 1, 4, 8, 12, 16, 20, 22, 24 y 24.8. Es una revisión por muestras; no equivale a una inspección visual continua de cada fotograma.

- Se observa un esqueleto reconocible de la parte superior del cuerpo y seguimiento de la persona con ID. Este video aporta la demostración de pose que faltaba en los planos cerrados de los otros videos.
- El termo negro de la izquierda se detecta en las muestras; el mate rojo se detecta en varios cuadros y se pierde en otros.
- Hay falsos positivos importantes: en las muestras de 4 y 8 segundos aparecen cajas de termo sobre la cabeza o el torso. Una confianza alta no garantiza que la clasificación sea correcta fuera del dataset de evaluación.
- En las muestras de 1, 20 y 22 segundos aparece una máscara COCO de cup sobre el mate rojo cuando el detector propio no lo reconoce. El filtro actual depende de las cajas propias del mismo fotograma, por lo que no resuelve ese caso.

El nuevo video permite demostrar pose, pero no garantiza que todas las salidas visuales sean correctas. El enunciado no establece un umbral mínimo de mAP ni exige cero errores de detección; las confusiones y superposiciones deben describirse como limitaciones. La exclusión de máscaras sobre las clases propias sigue siendo imperfecta en escenas nuevas.

## Resumen de la corrida

Los conteos siguientes son apariciones por fotograma, no objetos o personas únicos, ni una estimación de precisión.

| Video | Fotogramas | Mate | Termo | Máscaras COCO | Poses |
|---|---:|---:|---:|---:|---:|
| video01.mp4 | 278 | 282 | 98 | 389 | 113 |
| video02.mp4 | 1301 | 702 | 913 | 1184 | 427 |
| video03.mp4 | 633 | 751 | 398 | 1578 | 1779 |
| video04.mp4 | 625 | 261 | 868 | 272 | 628 |

## Documentación recibida y conclusiones

Se recibió el notebook 02 de la corrida de cuatro videos: sus 10 celdas de código tienen número de ejecución y no contienen errores guardados. El resumen coincide con `videos/output/resumen_procesamiento.csv`, y usa el experimento `n_sin_mosaic` con confianza 0.25.

Se completaron las conclusiones de ambos notebooks usando los resultados guardados y las observaciones visuales documentadas. Se conservaron sin cambios el código, los números de ejecución y los outputs. No se volvió a entrenar ni se ejecutó inferencia local.

La fuente y el recorte de video04 están registrados en `docs/video04_fuente_y_ejecucion.md` y `videos/input/video04_fuente.json`. La integración de modelos tiene evidencia de ejecución y de pose; las confusiones y la supresión imperfecta de máscaras se mantienen explícitas como limitaciones.
