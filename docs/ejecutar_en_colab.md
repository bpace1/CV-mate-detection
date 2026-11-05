# Ejecutar la comparación en Google Colab

## Archivos

- `01_Entrenamiento_YOLO26.ipynb`: abrirlo en Colab.
- `CV_mate_termo_Colab.zip`: cargarlo cuando lo solicite la primera celda.

El ZIP contiene las 90 imágenes originales, las etiquetas corregidas, los scripts auxiliares, la partición fija por sesiones, ambos notebooks y la referencia anterior. No incluye PyTorch ni entornos virtuales: Colab ya dispone de PyTorch con CUDA.

## Pasos

1. Abrir https://colab.research.google.com/ y elegir **Archivo > Subir notebook**. Seleccionar el notebook actualizado que acompaña al ZIP.
2. En **Entorno de ejecución > Cambiar tipo de entorno de ejecución**, elegir GPU (T4 si está disponible) y guardar.
3. Ejecutar la primera celda de código. Elegir el ZIP completo al abrirse el selector. Se extrae en `/content/CV-mate-detection/`.
4. Ejecutar las celdas restantes en orden, o usar **Ejecutar todas** tras cargar el ZIP. La instalación conserva el torch/CUDA provisto por Colab. Si Colab solicita reiniciar la sesión por un cambio de bibliotecas, reiniciarla y volver a ejecutar; el ZIP ya extraído se reutiliza mientras se conserve el runtime.
5. Confirmar que la celda de entorno muestra una GPU y no CPU. Durante la comparación aparece una línea por época y los logs completos quedan en `entrenamientos/comparacion_.../`.
6. Al terminar, la última celda muestra la ruta para descargar `entregable_entrenamiento.zip`: contiene el modelo seleccionado, la confianza, métricas, predicciones, configuración y dataset dividido.
7. Guardar también el notebook ejecutado con **Archivo > Descargar > Descargar .ipynb**. No alcanza con descargar la copia sin ejecutar que estaba dentro del ZIP original.

## Qué se compara

YOLO26n con mosaic 1.0, 0.0 y 0.5; YOLO26s con mosaic 0.5. Cada candidato tiene hasta 80 épocas y early stopping de 20. En Colab se usa batch 16 para todos. Solo validación elige modelo y confianza; test se mide al final. La tabla, la matriz y el resumen se generan con los resultados reales de esa ejecución.

## Cómo interpretar los resultados

Las métricas antiguas no son directamente comparables porque se corrigieron etiquetas y se cambió el split. Comparar los cuatro candidatos dentro de la nueva tabla de validación. Si gana la referencia nano con mosaic 1.0, no afirmar que una variante mejoró. Una única semilla y un test de 14 fotos todavía dan una estimación inestable.

Para revisar una ejecución, conservar `comparacion_validacion.csv`, `metricas_test.json`, `configuracion.json`, `seleccion.json`, la matriz y el notebook ejecutado. Los resultados nuevos se guardan en `resultados/comparacion_mejoras/`; la carpeta histórica conserva la referencia anterior.

## Sesión y recursos

El entrenamiento usa los recursos de Colab, no la GPU de tu PC. El tiempo depende de la GPU asignada, la carga y el early stopping; no hay una duración medida todavía para esta nueva comparación. Los archivos en `/content` son temporales: descargar el ZIP y el notebook ejecutado antes de que termine la sesión. No hace falta autorizar acceso a Drive para el flujo de carga por ZIP.

Si hay un error, conservar el mensaje completo y el nombre de la celda. Ante falta de memoria de GPU, reiniciar el runtime y bajar `BATCH` a 8 o 4 **para todos los candidatos**, manteniendo el mismo split.

## Desde Visual Studio Code con la extensión oficial

1. Instalar **Google Colab** (`Google.colab`) y su dependencia Jupyter en VS Code.
2. Abrir el notebook actualizado y elegir **Select Kernel > Colab > New Colab Server**. Iniciar sesión con Google y seleccionar una máquina con GPU, si está disponible. `Auto Connect` también conecta, pero no garantiza que la máquina asignada tenga GPU.
3. En el explorador local de VS Code, hacer clic derecho sobre `CV_mate_termo_Colab.zip` > **Upload to Colab** y esperar a que termine. No alcanza con tener el dataset en tu disco: el kernel remoto solo ve sus archivos en Colab.
4. Ejecutar las celdas del notebook. La primera detecta `/content/CV_mate_termo_Colab.zip` y extrae el proyecto. La celda de entorno debe mostrar una GPU; si no, elegir otro servidor antes de entrenar.
5. Al terminar, abrir **Colab > Contents**, buscar `/content/CV-mate-detection/entregable_entrenamiento.zip` y usar clic derecho > **Download...**. Guardar también el notebook local ejecutado desde VS Code.

`google.colab.files.upload()` y `files.download()` tienen limitaciones en VS Code. El notebook evita usarlos si se sube previamente el ZIP; la descarga web está desactivada por defecto y puede habilitarse solamente en el navegador. En VS Code se usan las acciones de la extensión.

Referencias oficiales: https://github.com/googlecolab/colab-vscode/wiki/User-Guide y https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds .
