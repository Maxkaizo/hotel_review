# Plan de trabajo: clasificador de reseñas de hoteles

## Objetivo

Construir un clasificador binario de reseñas de hoteles mediante una RNN con una capa LSTM y una capa de embeddings no entrenable basada en vectores Word2Vec preentrenados.

## Plan

1. Explorar el dataset y confirmar que `polarity` será la etiqueta objetivo.
2. Preprocesar y tokenizar las reseñas en inglés.
3. Seleccionar y cargar un modelo Word2Vec preentrenado en inglés.
4. Consultar la dimensión y el vocabulario del modelo Word2Vec.
5. Crear el vocabulario numérico de las reseñas.
6. Construir la matriz de embeddings usando los vectores Word2Vec.
7. Aplicar `pad_sequences(padding='pre')` con una longitud máxima conveniente.
8. Diseñar la red con la siguiente estructura:
   - capa `Embedding` congelada;
   - capa `LSTM` con tantas unidades como dimensiones tenga Word2Vec y `dropout=0.2`;
   - capa `Dense` de una neurona con activación `sigmoid`.
9. Entrenar y validar el modelo.
10. Evaluar con métricas de clasificación y matriz de confusión.
11. Redactar las conclusiones, incluyendo limitaciones de Word2Vec y de las RNN.

## Uso de la libreta de referencia

La libreta `references/Word_embeddings_Word2Vec_español.ipynb` se utilizará como referencia para:

- cargar vectores Word2Vec preentrenados con Gensim;
- consultar el vocabulario mediante `index_to_key`;
- obtener vectores y verificar su dimensión;
- explorar similitudes entre palabras, si resulta útil en la etapa de exploración.

Como las reseñas están en inglés, se utilizará un modelo Word2Vec preentrenado en inglés, no el modelo en español de la libreta.

Las analogías, visualizaciones 3D y recorridos manuales de todo el vocabulario son opcionales y no forman parte esencial del clasificador.

## Decisiones de implementación

- Entorno administrado con `uv`, Python 3.12, dependencias en `pyproject.toml` y versiones en `uv.lock`.
- Notebook: `clasificador_resenas_hoteles.ipynb`, con Markdown explicativo antes de cada celda de código.
- Word2Vec Google News de 300 dimensiones; LSTM de 300 unidades.
- División estratificada aproximada 70/15/15 después de resolver duplicados.
- Vocabulario y percentil 95 de longitudes calculados solo con entrenamiento.
- Padding previo, truncamiento posterior y máscara para ignorar el relleno.
- Coincidencia Word2Vec exacta y después en minúsculas; promedio de vectores encontrados como respaldo.
- Embeddings congelados, LSTM con dropout 0.2, salida sigmoid y entropía cruzada binaria.
- Validación para detención temprana; prueba reservada para métricas finales.
- Las métricas y conclusiones numéricas requieren ejecutar el entrenamiento completo: no se presuponen resultados.
