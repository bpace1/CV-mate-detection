# Dataset y entrenamiento

Detalle de cómo armamos el dataset, cómo lo dividimos y cómo elegimos el modelo. El código está en `notebooks/01_Entrenamiento_YOLO26.ipynb` y los resultados en `resultados/comparacion_mejoras/`.

## Dataset

Hay 90 fotos en `dataset/`, sacadas por los integrantes del grupo. Cada uno anotó una parte en [MakeSense](https://www.makesense.ai/) y exportó un CSV, que está en `data/raw_annotations/`. `src/unify_annotations.py` junta los CSV y escribe una etiqueta YOLO por imagen en `data/unified/labels/`.

| Clase | ID | Imágenes | Cajas |
|-------|----|----------|-------|
| `mate` | 0 | 62 | 65 |
| `termo` | 1 | 50 | 51 |

Criterio de anotación: la caja cubre el recipiente visible y su base, sin estirarla para abarcar toda la bombilla. Si una mano tapa parte del objeto, la caja llega hasta donde se ve.

### Correcciones

Antes del segundo entrenamiento revisamos las 90 fotos con sus cajas. Corregimos:

- `IMG_4037.jpeg`: estaba etiquetado como termo y es un mate. Se corrigió la clase en el CSV de Bruno.
- `termo-42.jpeg`: faltaba la caja del mate. Se agregó en el CSV de Franco B.
- `IMG_7160.jpg` y `IMG_7165.jpg`: faltaba un segundo mate, desenfocado al fondo.
- `01c37ef5-ac2f-4867-8b92-8fc876377a82.jpg`: faltaba un termo gris al fondo, cortado por el borde superior.

La revisión fue visual. Puede quedar algún error en objetos chicos, borrosos o tapados.

## Partición por sesiones

Muchas fotos se sacaron en la misma sesión: mismo lugar, misma luz, mismos objetos. Si dos fotos de una sesión caen una en train y otra en test, el test mide cuánto se parecen y no si el modelo generaliza. Por eso cada sesión va entera a un solo split.

`data/source_groups.csv` asigna cada foto a una sesión. Las sesiones las identificamos mirando las fotos (comedor, mesada, cocina, cortina y algunos pares muy parecidos), no con metadatos de la cámara. El split apunta a 70/15/15 en cantidad de imágenes y busca un balance de clases parecido en los tres conjuntos, usando solo las etiquetas.

Resultado: 63 imágenes de entrenamiento, 13 de validación y 14 de test. `data/split_manifest.csv` guarda el split con el hash SHA-256 de cada imagen y etiqueta. El notebook 01 lo reutiliza y se detiene si algún archivo cambió.

Para reconstruir el dataset dividido sin entrenar:

```bash
python src/prepare_dataset.py
```

El script copia las imágenes y etiquetas a `dataset_yolo26_mate_termo/` según el manifiesto. `data.yaml`, en la raíz, apunta a esa carpeta con rutas relativas.

## Comparación de configuraciones

Entrenamos cuatro candidatos desde pesos COCO, con el mismo split, semilla 42, imágenes de 640 px, batch 16 y hasta 80 épocas con early stopping a las 20 sin mejora. Solo cambian el modelo base y mosaic, un aumento que pega cuatro imágenes en una:

| Candidato | Modelo base | Mosaic | mAP50 (val) | mAP50-95 (val) |
|-----------|-------------|--------|-------------|----------------|
| `n_mosaic100` | `yolo26n.pt` | 1.0 | 0.981 | 0.814 |
| `n_sin_mosaic` | `yolo26n.pt` | 0.0 | 0.978 | **0.824** |
| `n_mosaic50` | `yolo26n.pt` | 0.5 | 0.995 | 0.797 |
| `s_mosaic50` | `yolo26s.pt` | 0.5 | 0.944 | 0.704 |

Elegimos el candidato con mayor mAP50-95 en validación: YOLO26n sin mosaic. La confianza de trabajo, 0.25, es la que maximiza el F1 macro en validación. Recién con el modelo y la confianza fijos evaluamos en test.

Las diferencias entre los tres modelos `n` son de uno o dos puntos con una sola semilla, así que no alcanzan para decir que una configuración de mosaic es mejor que otra. Para eso habría que repetir con varias semillas o con validación cruzada por sesiones.

## Resultados en test

14 imágenes y 17 objetos, confianza 0.25, emparejamiento uno a uno con IoU de 0.5 o más:

| Métrica | Valor |
|---------|-------|
| Precisión (macro) | 0.875 |
| Recall (macro) | 0.879 |
| F1 (macro) | 0.874 |
| mAP50 | 0.905 |
| mAP50-95 | 0.593 |

F1 por clase: mate 0.947, termo 0.800. La matriz de confusión tiene 9 mates y 6 termos bien detectados, 1 mate clasificado como termo, 1 termo no detectado y 1 falso positivo de termo.

## Primera corrida

`resultados/historico_original/` guarda la primera corrida: YOLO26n con mosaic por defecto, el split anterior (test de 9 imágenes y 11 objetos) y las etiquetas sin corregir. Dio mAP50 0.708 en test. No se puede comparar con la corrida actual porque cambiaron las etiquetas y el split, y ese modelo vio en entrenamiento fotos que ahora son test.
