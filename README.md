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

## Validación de GPU

Abre `validacion_gpu_tensorflow.ipynb` con un kernel nuevo. Comprueba una operación
real, gradientes y entrenamiento de un clasificador MNIST pequeño en GPU. También
incluye pruebas independientes de LSTM nativa y cuDNN, sin necesitar Word2Vec.
MNIST se descarga una vez en `.cache/keras/`. El informe se guarda en
`artifacts/gpu_validation/result.json`.

Para ejecutarla desde la terminal:

```bash
uv run python scripts/execute_notebook.py --notebook validacion_gpu_tensorflow.ipynb
```

## Experimento adicional de padding

El anexo A de la libreta compara `pre` nativo, `post` nativo y `post` con cuDNN en
la misma GPU. El entrenamiento y la evaluación principales mantienen `pre`, como
exige la tarea. El anexo usa modelos descartables con pesos iniciales comunes,
calentamiento y tres repeticiones de pasos fijos de entrenamiento; no compara
exactitud tras convergencia. Registra los resultados en `artifacts/padding_comparison/`.
Si no hay GPU o cuDNN falla, lo informa sin fabricar una aceleración. Puede omitirse
con `RUN_PADDING_EXPERIMENT = False` en su primera celda.

## Guía visual

El [documento de acompañamiento](acompanamiento.md) registra la explicación y
justificación de decisiones, empezando por la elección de Word2Vec Google News
y la descarga con Gensim a una caché local reutilizable. Servirá como base de un posible artículo
al terminar el experimento; su formato final todavía no está decidido.

Abre `index.html` o ejecuta `python3 -m http.server 8000 --bind 127.0.0.1`
y visita `http://localhost:8000`.
