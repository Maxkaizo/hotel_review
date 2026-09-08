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
- Cargar todos sus vectores requiere varios GB de RAM. Se libera el modelo completo tras construir la matriz del vocabulario.
- El padding previo con máscara requiere la implementación nativa de LSTM; se configura `use_cudnn=False`. El entrenamiento puede tardar según el equipo.
- `artifacts/` contiene matriz, modelo `.keras`, reglas de preprocesamiento, métricas, predicciones y conclusiones. Reejecutar las celdas reemplaza esos resultados.

La ejecución es necesaria para obtener métricas: la libreta no incluye resultados inventados.
El modelo usa entrenamiento/validación/prueba separados y verifica que los embeddings
no cambien y que la recarga conserve predicciones. Para reutilizarlo, la función de
tokenización de la sección 2 debe acompañar a las reglas guardadas.

## Guía visual

Abre `index.html` o ejecuta `python3 -m http.server 8000 --bind 127.0.0.1`
y visita `http://localhost:8000`.
