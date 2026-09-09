# Clasificador de reseñas de hoteles

Notebook didáctico con Word2Vec Google News congelado y una LSTM para clasificar
reseñas positivas y negativas. Cada celda de código tiene una explicación previa.

## Entorno con uv

Desde la raíz del proyecto, con Python 3.12:

```bash
uv sync
uv run jupyter lab
```

Abre `clasificador_resenas_hoteles.ipynb` y selecciona el kernel de `.venv`.
Ejecuta las celdas en orden. Las dependencias se declaran en `pyproject.toml`;
`uv.lock` fija las versiones resueltas. No se requiere instalar pip por separado.

Si una terminal que ya estaba abierta no encuentra uv, abre otra terminal o usa
`/home/maxkaizo/.local/bin/uv` en este equipo.

## Versión para Google Colab

Sube `clasificador_resenas_hoteles_colab.ipynb` a Colab y ejecuta las celdas en
orden desde una sesión nueva (CPU o GPU, no TPU). No requiere uv ni el repositorio.
Instala Gensim, reutiliza TensorFlow de Colab y lee Google News por partes para
guardar solo los vectores necesarios en RAM. Mantiene la arquitectura y padding
previo de la tarea. `USE_DRIVE = True` permite conservar descargas y resultados
entre sesiones; por defecto se guardan solo en la sesión actual. La última celda
descarga un ZIP de resultados. Las salidas locales no se copian a esta versión.

## Datos y recursos

- CSV local: `data/deceptive-opinion.csv`; si falta, el notebook descarga la fuente de la tarea.
- Word2Vec: `word2vec-google-news-300`, con 300 dimensiones; descarga inicial de aproximadamente 1.74 GB en `data/embeddings/`.
- Gensim guarda el archivo en `data/embeddings/word2vec-google-news-300/word2vec-google-news-300.gz`. La celda 4.1 reutiliza ese archivo sin acceder a la red si ya existe; verifica tamaño y MD5 antes de cargarlo.
- Se conserva el archivo original en disco. Liberar los vectores completos de RAM no borra la descarga ni el dataset de reseñas. La matriz compacta se guarda junto con `embedding_vocabulary.json` para conservar la correspondencia de índices.
- Cargar todos sus vectores requiere varios GB de RAM. Se libera el modelo completo tras construir la matriz del vocabulario.
- El padding previo con máscara requiere la implementación nativa de LSTM; se configura `use_cudnn=False`. El entrenamiento puede tardar según el equipo.
- `artifacts/` contiene matriz, modelo `.keras`, reglas de preprocesamiento, métricas, predicciones y conclusiones. Reejecutar las celdas reemplaza esos resultados.

La ejecución es necesaria para obtener métricas: la libreta no incluye resultados inventados.
El modelo usa entrenamiento/validación/prueba separados y verifica que los embeddings
no cambien y que la recarga conserve predicciones. Para reutilizarlo, la función de
tokenización de la sección 2 debe acompañar a las reglas guardadas.

## Guía visual

La libreta principal contiene las instrucciones de ejecución, la justificación de
las decisiones, los resultados y las limitaciones en sus propias celdas Markdown.
También incluye instrucciones para ejecutarla fuera del repositorio; requiere
instalar las dependencias y disponer de red para la primera descarga de los datos.
La guía visual es complementaria y no es necesaria para entender el entregable.

Abre `index.html` o ejecuta `python3 -m http.server 8000 --bind 127.0.0.1`
y visita `http://localhost:8000`.
