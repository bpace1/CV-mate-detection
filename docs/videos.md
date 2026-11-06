# Videos

Videos usados en el notebook 02 y revisión de los resultados. Las conclusiones completas están en la sección 7 de `notebooks/02_Procesamiento_Videos_YOLO26.ipynb`.

## Videos de entrada

| Video | Duración | FPS | Resolución | Origen |
|-------|----------|-----|------------|--------|
| `video01.mp4` | 9.3 s | 30 | 576x1024 | Grabado por el grupo |
| `video02.mp4` | 21.7 s | 60 | 464x832 | Grabado por el grupo |
| `video03.mp4` | 10.6 s | 60 | 464x832 | Grabado por el grupo |
| `video04.mp4` | 25.0 s | 25 | 1280x720 | YouTube, ver abajo |

Los videos 01 a 03 los grabamos los integrantes del grupo. Son planos cerrados de una mesa de trabajo con mate, termo, laptop y teclado; de las personas se ven las manos y los brazos. Sirven para el detector propio y la segmentación. Para la pose hacía falta una persona más visible, y por eso sumamos `video04.mp4`.

`video04.mp4` son los primeros 25 segundos de [La forma definitiva de preparar mate](https://www.youtube.com/watch?v=I5JOt0B6iS8), del canal Sudamerican Mate. YouTube no informa una licencia Creative Commons para ese video, así que lo usamos solo como ejemplo académico. La fuente, el recorte y el hash del archivo están en `videos/input/video04_fuente.json`.

## Resultados

| Video | Fotogramas | Mate | Termo | Máscaras COCO | Poses |
|-------|-----------:|-----:|------:|--------------:|------:|
| video01 | 278 | 282 | 98 | 378 | 113 |
| video02 | 1301 | 702 | 913 | 490 | 427 |
| video03 | 633 | 751 | 398 | 1157 | 1779 |
| video04 | 625 | 261 | 868 | 13 | 628 |

Los conteos son apariciones por fotograma, no objetos distintos. La configuración usada, con los hashes de los pesos y de cada video de entrada, está en `videos/output/configuracion_procesamiento.json`.

## Errores observados

- **Termos falsos sobre la persona** en video04: cajas de termo sobre la cabeza o el torso.
- **Mates perdidos.** El mate rojo de video04 no se detecta en varios cuadros. En video03 el termo se pierde en algunos cuadros mientras se sirve el agua.
- **Máscaras sobre mates.** Cuando el detector pierde un mate más de 2 segundos, el segmentador lo pinta como `cup` o `bowl`. Por ejemplo, en video02 cerca del segundo 10.85.
- **Poses de brazos sueltos** en video01 a video03: el modelo de pose toma un brazo o una mano como persona, con uno o dos keypoints.
