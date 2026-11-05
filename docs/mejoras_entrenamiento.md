# Revisión del dataset y comparación de modelos

Revisión visual de las 90 imágenes el 5 de noviembre de 2026.

## Anotaciones corregidas

- `IMG_4037.jpeg`: era clase termo, pero el objeto es un mate. Se conserva la caja y se corrige la clase en el CSV original de Bruno.
- `termo-42.jpeg`: se agrega el mate visible que faltaba, en el CSV de Franco B.
- `IMG_7160.jpg` y `IMG_7165.jpg`: se agrega el segundo mate visible al fondo. El primero está desenfocado; la caja se limita al recipiente visible.
- Criterio: recipiente visible y soporte, sin extender la caja para abarcar toda la bombilla. No se inventa la parte tapada por la mano.

Las cajas nuevas se revisaron visualmente. La revisión no garantiza ausencia de otros errores, especialmente en objetos pequeños, borrosos o parcialmente ocultos. En la inspección ampliada de `01c37ef5-ac2f-4867-8b92-8fc876377a82.jpg` se identifica el termo Lumilagro gris al fondo y se agrega su caja visible (truncado en el borde superior). Esta imagen también tenía una anotación incompleta.

## Fuentes y partición

`data/source_groups.csv` registra las sesiones inferidas visualmente, no metadatos confirmados de captura. Se agrupan las sesiones de comedor, mesada, cocina y cortina, así como algunas parejas claramente similares. Pueden quedar relaciones entre fotos que no se reconocieron visualmente.

El split nuevo es aproximadamente 70/15/15 por imágenes, sin separar grupos. Se busca balance de clases usando solo etiquetas; nunca métricas del modelo. `data/split_manifest.csv` fija la asignación y hashes de imágenes/etiquetas. La comparación reutiliza exactamente ese manifiesto.

Los resultados nuevos no son directamente comparables con el test anterior: cambian etiquetas y partición. `resultados/historico_original/` conserva los resultados, notebook y pesos originales. El modelo antiguo vio parte de los nuevos holdouts; no se usa como referencia de generalización. Se entrena una nueva referencia desde pesos COCO.

## Experimentos

Cuatro candidatos: YOLO26n con mosaic 1.0, 0.0 y 0.5, y YOLO26s con mosaic 0.5. Comparten split, semilla, resolución, batch y presupuesto de épocas. Mosaic se desactiva a partir de la época 20 en las corridas de 80 épocas; no hay aumento en val/test. La selección usa mAP50-95 de validación y el umbral usa F1 macro de validación. Test se evalúa después de fijar ambos.

P/R/F1 operativos se calculan con confianza fija e IoU 0.5, matching uno a uno. mAP se calcula por separado con conf=0.001. Se guarda el detalle de las cajas y errores; F1 macro es la media de F1 por clase, no el F1 de P/R promediados. Una corrida por candidato no mide variabilidad entre semillas.

No se agregaron fotos nuevas ni negativos sintéticos. Para seguir mejorando, reunir escenas nuevas de uso real y negativos revisados manualmente. Al incorporar imágenes o cambiar etiquetas, revisar y versionar el manifiesto antes de rehacer el split.

## Estado de validación antes de Colab

Las etiquetas pasan la validación del proyecto: 62 imágenes con mate (65 cajas) y 50 con termo (51 cajas), sobre 90 fotos. El split persistente tiene 63/13/14 imágenes y no comparte las sesiones identificadas. Se verificó su recarga sin cambios, el matching de confusiones, duplicados, omisiones y negativos, así como el barrido de confianza con un caso controlado. Ruff y la validación de estructura/sintaxis de ambos notebooks pasan.

La instalación local de CUDA se detuvo a pedido del usuario para ejecutar en Colab. Esta comparación todavía no se entrenó: no hay métricas nuevas ni pesos nuevos. Los outputs antiguos se quitaron del notebook revisado para evitar presentarlos como resultados de la nueva configuración. La prueba pendiente es ejecutar el notebook completo en Colab con GPU y revisar los resultados guardados.

## Resultado posterior de Colab

La ejecución final ya fue recibida y verificada. Se eligió YOLO26n sin mosaic (mAP50-95 de validación 0.8236), con confianza 0.25. Test: mAP50 0.9045, mAP50-95 0.5933 y F1 macro 0.8737. Los archivos originales están en resultados/comparacion_mejoras/. Los cuatro videos y ambos notebooks ejecutados también fueron recibidos; las conclusiones describen la pose lograda y los errores que persisten. El apartado anterior registra el estado previo a ejecutar en Colab.
