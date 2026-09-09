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

## Acompañamiento y trabajo posterior

- Mantener [acompanamiento.md](acompanamiento.md) como documento vivo de explicación,
  justificación de decisiones, evidencia y asuntos pendientes.
- Decisión confirmada: Google News Word2Vec en inglés, descargado con Gensim usando
  `return_path=True` y guardado en `data/embeddings/word2vec-google-news-300/`.
- Reutilizar el archivo local verificado sin consultar la red; cargar, filtrar y
  guardar matriz compacta y vocabulario. Liberar Word2Vec completo de RAM sin borrar
  su archivo en disco ni las reseñas necesarias para entrenar y evaluar.
- Terminar la primera versión local antes de generar variantes de Colab CPU y GPU.
- Conservar `padding='pre'` en el entregable. Añadir un anexo de rendimiento en la
  misma GPU: pre nativo, post nativo como control y post con cuDNN exigido. Medir
  pasos fijos tras calentamiento, sin modificar el modelo principal ni ajustar con prueba.
- Validación previa completada en `validacion_gpu_tensorflow.ipynb`: operación,
  gradientes y entrenamiento MNIST en RTX 3070 Ti; pruebas pequeñas de las tres
  rutas LSTM correctas. No equivale a entrenar el clasificador de reseñas.
- Idea futura, sin iniciar: convertir el acompañamiento en un artículo how-to o
  write-up, posiblemente HTML editorial, una vez obtenidos los resultados.
