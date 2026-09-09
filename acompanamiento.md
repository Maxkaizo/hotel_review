# De las reseñas al clasificador: notas y decisiones

Documento de acompañamiento de [la libreta](clasificador_resenas_hoteles.ipynb).
Su propósito es explicar qué hacemos, por qué elegimos cada técnica y qué falta
comprobar. Lo actualizaremos conforme avance el proyecto.

**Estado al 8 de septiembre de 2026:** se creó la libreta y se verificó su ejecución
hasta la preparación de secuencias. La selección de embeddings se revisa aquí antes
de su descarga. No hay resultados finales de entrenamiento ni mediciones de
aceleración que permitan concluir sobre el desempeño.

## 1. Qué queremos aprender y qué exige la tarea

El objetivo es clasificar la polaridad de reseñas de hoteles en inglés. Usaremos
`text` como entrada y `polarity` como etiqueta: negativa = 0, positiva = 1.
La autenticidad de la reseña, descrita por `deceptive`, es otro problema.

El enunciado exige Word2Vec preentrenado en una capa Embedding no entrenable,
una LSTM con tantas unidades como dimensiones tengan los vectores, dropout de 0.2,
y una salida binaria de una neurona. También exige secuencias de longitud fija
con `pad_sequences(padding='pre')` y un notebook que incluya las siete etapas del
trabajo. No exige GPU, Google News ni una dimensión concreta.

La elección de técnicas clásicas responde al propósito didáctico: queremos seguir
la transformación del texto y comprender qué aprende cada componente. No estamos
demostrando que esta arquitectura sea la mejor alternativa disponible.

## 2. Decisión: Word2Vec Google News de 300 dimensiones

**Decisión acordada con el usuario:** `word2vec-google-news-300`, distribuido en el
repositorio de datos de Gensim. Lo elegimos por ser un recurso reconocido y por su
ajuste al ejercicio; todavía falta validar su utilidad en nuestro corpus.

### Qué sabemos del recurso

| Propiedad | Dato publicado |
|---|---|
| Idioma | Inglés |
| Corpus de origen | Google News, aproximadamente 100 mil millones de palabras |
| Entradas | 3 millones de palabras y frases |
| Dimensiones por vector | 300 |
| Archivo | `word2vec-google-news-300.gz` |
| Tamaño comprimido | 1,743,563,840 bytes, aproximadamente 1.74 GB |

Fuente de estos metadatos: [catálogo de modelos de Gensim](https://github.com/piskvorky/gensim-data/blob/master/list.json).

### Por qué lo elegimos

1. **Cumple la técnica pedida.** Son vectores Word2Vec ya entrenados; no necesitamos
   entrenar embeddings con nuestras reseñas para satisfacer el enunciado.
2. **Coincide con el idioma.** Las reseñas están en inglés, por lo que el recurso
   español de la lección no es una elección adecuada para este corpus.
3. **Da continuidad a la lección.** Podemos usar `KeyedVectors` para cargar y
   consultar vectores, como en la libreta de referencia. El paso nuevo será
   organizarlos por los índices de nuestro vocabulario para inicializar Embedding.
4. **Tiene una fuente identificable.** Registraremos el recurso, su procedencia y
   la comprobación del archivo para poder repetir el experimento.

Estas razones justifican una elección práctica, no una superioridad empírica. Un
vocabulario preentrenado grande no garantiza cobertura de nuestras palabras ni
mejor clasificación.

### Costes y límites que aceptamos

- **Memoria:** por cálculo, 3,000,000 × 300 × 4 bytes son 3.6 GB solo para los
  vectores en `float32`. La carga completa necesita memoria adicional para claves,
  estructuras de Gensim y el resto del proceso. No es una medición de consumo real.
- **Coste de entrenamiento:** el requisito de la tarea convierte las 300 dimensiones
  en una LSTM de 300 unidades. No elegimos esa dimensión por un ajuste de rendimiento.
- **Cambio de dominio:** noticias y reseñas de hoteles no son el mismo tipo de texto.
  Mediremos la cobertura en entrenamiento y examinaremos palabras ausentes.
- **Vocabulario y limpieza:** hay distinción de mayúsculas y entradas de varias
  palabras. Nuestra tokenización por palabras no aprovecha automáticamente todas
  las frases del recurso. Comprobaremos coincidencia exacta y después minúsculas.
- **Representaciones fijas:** una palabra conserva el mismo vector en diferentes
  reseñas. Las relaciones de contexto de Word2Vec no son etiquetas de sentimiento.
- **Procedencia de uso:** el catálogo registra la licencia del recurso como
  `not found`; no documentaremos el archivo como si tuviera una licencia abierta
  verificada. [Metadatos del recurso](https://github.com/piskvorky/gensim-data/blob/master/list.json).

### Alternativas consideradas conceptualmente

| Alternativa | Por qué no es nuestra elección inicial |
|---|---|
| Word2Vec de la lección en español | No coincide con el idioma de las reseñas. |
| Entrenar Word2Vec desde cero con nuestras reseñas | No satisface la petición de usar un modelo preentrenado externo y cambia el objetivo del ejercicio. |
| Otro Word2Vec preentrenado en inglés | Sería válido; no hemos evaluado un candidato concreto de menor tamaño. |
| GloVe o FastText | Son otras técnicas de embeddings; no las sustituiremos por Word2Vec sin revisar el requisito. |
| Embeddings contextuales de transformers | Quedan fuera de la arquitectura solicitada para esta primera versión. |

El catálogo incluye GloVe convertido a formato Word2Vec: **el formato de archivo
no cambia el algoritmo con el que se aprendieron los vectores**. Esa conversión no
lo convierte en un modelo Word2Vec. [Catálogo de Gensim](https://github.com/piskvorky/gensim-data/blob/master/list.json).

## 3. Decisión acordada: Gensim con archivo local reutilizable

Elegir el modelo y elegir cómo descargarlo son decisiones distintas. Gensim puede
descargar y cargar en una llamada, o descargar sin cargar usando `return_path=True`.
[API oficial de descarga](https://radimrehurek.com/gensim/downloader.html).

Se consideró la descarga por terminal con `curl`, pero finalmente acordamos usar
Gensim, separando la descarga de la carga en RAM. La primera descarga usa
`return_path=True`. En ejecuciones posteriores comprobamos primero la ruta local
y no llamamos al descargador si el archivo ya existe y pasa las verificaciones.
La descarga no depende de una GPU ni de TensorFlow.

Después utilizamos Gensim para cargar el archivo local y consultar los vectores.
No se reduce la descarga por seleccionar luego
solo las palabras que necesitamos: con este procedimiento primero obtenemos el
recurso completo y después construimos la matriz compacta.

La libreta verifica tamaño y MD5 contra los metadatos publicados antes de cargar.
Una descarga incompleta no se acepta únicamente porque la ruta exista; un fallo
detiene el flujo sin borrar automáticamente el archivo. La comprobación lee el
archivo por bloques, sin cargar toda la matriz en RAM. El MD5 comprueba integridad,
no constituye una garantía criptográfica de autenticidad.

Ruta acordada:
`data/embeddings/word2vec-google-news-300/word2vec-google-news-300.gz`.

Tras filtrar, guardamos `embedding_matrix.npy` y `embedding_vocabulary.json` en
`artifacts/`. Liberamos el objeto Word2Vec completo de RAM, conservando tanto el
archivo original en disco como las reseñas y la matriz compacta en memoria.
Liberar memoria no significa borrar el dataset de reseñas ni el archivo descargado.

El archivo local se reutiliza aunque reiniciemos Jupyter. En un entorno efímero
como Colab habría que conservarlo en almacenamiento persistente para reutilizarlo
tras perder la sesión; ese traslado sigue aplazado.

## 4. Cómo se conecta el dataset con los embeddings

El dataset aporta las palabras, su orden y las etiquetas de cada reseña. Con el
entrenamiento construiremos un vocabulario que asigna un identificador a cada
palabra. Por separado, buscaremos su vector en Word2Vec y lo colocaremos en la fila
indicada por ese identificador.

Si `hotel` tiene índice 3, la fila 3 de la matriz compartida contendrá su vector.
Una reseña convertida en índices hará que Embedding seleccione las filas
correspondientes, manteniendo el orden. Las etiquetas se utilizan para calcular
errores; no se incluyen entre los componentes del embedding.

- **V × D:** matriz compartida, con V entradas de vocabulario y D dimensiones.
- **L × D:** representación de una reseña con L posiciones.
- **B × L × D:** representaciones de un lote de B reseñas.

La LSTM procesa cada reseña posición por posición y actualiza su estado. El batch
es un hiperparámetro que organiza varias reseñas juntas. La salida produce una
predicción por reseña. La LSTM y la salida aprenden patrones de sentimiento, mientras
los embeddings permanecen congelados: es transferencia de representaciones fijas.

## 5. Otras decisiones y su justificación

Estas decisiones están implementadas en la libreta; sus resultados aún deben
evaluarse. Ninguna se presenta como óptima por adelantado.

| Decisión | Motivo y límite |
|---|---|
| Resolver duplicados antes de dividir | Evitar que el mismo texto aparezca en entrenamiento y prueba. |
| División estratificada aproximada 70/15/15 | Separar aprendizaje, selección y evaluación manteniendo proporciones de clases. No equivale a separar por hotel. |
| Vocabulario solo de entrenamiento | Evitar que las entradas reservadas influyan en su construcción. |
| Conservar negaciones y orden | Mantener información relevante para el sentimiento. |
| Longitud máxima en el percentil 95 de entrenamiento | Compromiso entre conservar texto y reducir relleno; recorta algunas reseñas. |
| Padding previo y truncamiento posterior | Cumplir el relleno exigido; conservar el inicio al recortar es una elección adicional. |
| Relleno con índice 0 y máscara | Impedir que las posiciones artificiales se traten como palabras. |
| Vector promedio como respaldo | Manejo determinista de desconocidas; no aporta un significado específico para ellas. |
| Sigmoid y entropía cruzada binaria | Una salida probabilística para dos clases y una pérdida acorde. |
| Detención temprana con validación | Seleccionar pesos sin usar el conjunto de prueba. |
| Métricas por clase y referencia mayoritaria | Interpretar el desempeño más allá de la exactitud aislada. |
| Entorno uv y archivo de bloqueo | Registrar versiones y facilitar la reproducción local. |

## 6. CPU, GPU y orden del trabajo

La GPU no es un requisito académico. Queremos terminar primero la versión local,
verificando el uso de la GPU cuando sus dependencias estén listas. Las variantes
de Colab CPU y GPU están aplazadas hasta completar esa primera versión.

Con máscara, la ruta rápida de cuDNN exige padding a la derecha. Nuestra libreta
conserva el padding previo requerido y configura `use_cudnn=False`. La ejecución
nativa todavía puede usar GPU; no hay una aceleración porcentual universal que
podamos afirmar sin medir. [Requisitos de Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/).

Cuando comparemos entornos, mantendremos datos, particiones y arquitectura, y
registraremos dispositivos, versiones y tiempos. Separaremos descarga, carga y
entrenamiento. Comparar dos equipos no aísla por sí solo el coste de prescindir de
cuDNN, y distintas cantidades de épocas tampoco representan el mismo trabajo.

### Decisión adicional: cumplir primero y experimentar después

Conservamos `pre` en el procedimiento y la evaluación principales. El anexo A
compara tres casos en la misma GPU: pre nativo, post nativo y post con cuDNN exigido.
El segundo caso es un control para distinguir mover el relleno de activar cuDNN.
Se conserva el orden de tokens, el truncamiento y la máscara.

El microbenchmark parte de pesos iniciales comunes y modelos descartables. Realiza
dos pasos de calentamiento, restaura pesos y optimizador, mide ocho actualizaciones
y repite tres veces alternando el orden de casos. Comprueba equivalencia aproximada
de las predicciones iniciales sin dropout y que el modelo principal no cambie.
Las máscaras dropout y el cálculo pueden diferir entre implementaciones: no se
promete igualdad bit a bit. Solo se usan lotes de entrenamiento.

Los tiempos excluyen descarga y calentamiento e incluyen la llamada desde Python
y sincronización. No son tiempos de entrenamiento hasta convergencia ni evidencia
de mejor calidad predictiva. Si falta GPU o falla cuDNN, se registra el estado sin
presentar una comparación completa. Las mediciones reales siguen pendientes.

### Propósito de cada prueba del anexo

| Prueba | Propósito | Interpretación |
|---|---|---|
| `pre_native` en GPU | Referencia con el padding exigido por la tarea. | Nativa no significa CPU. |
| `post_native` en GPU | Cambiar solo la posición del relleno. | Frente a `pre_native`, observa el efecto del padding manteniendo implementación. |
| `post_cudnn` en GPU | Exigir la LSTM optimizada de cuDNN. | Frente a `post_native`, mide el efecto de cuDNN con el mismo padding. |

Comparar `pre_native / post_cudnn` expresa el cambio práctico completo, pero mezcla
padding e implementación. Las tres pruebas usan la misma GPU; no comparan CPU y
GPU ni calidad predictiva tras convergencia.

El controlador NVIDIA permite acceder al hardware, CUDA ofrece cálculo en GPU y
cuDNN aporta operaciones especializadas para redes neuronales. Desactivar cuDNN
en una LSTM no desactiva la GPU.

### Validación previa e independiente de la GPU

La libreta `validacion_gpu_tensorflow.ipynb` adapta el tutorial inicial de TensorFlow
con MNIST y su guía de GPU. Verifica detección, operación real, gradientes y cambios
de pesos durante un entrenamiento pequeño. No necesita Word2Vec ni las reseñas.
Añade comprobaciones de LSTM pre nativa, post nativa y post cuDNN con datos sintéticos:
son pruebas de funcionamiento, no resultados del clasificador ni benchmarks.

Fuentes: [tutorial para principiantes](https://www.tensorflow.org/tutorials/quickstart/beginner)
y [guía de GPU](https://www.tensorflow.org/guide/gpu).

**Validación realizada el 8 de septiembre de 2026:** TensorFlow 2.20.0 y Keras 3.15.1
detectaron la RTX 3070 Ti. Se comprobaron multiplicación, variables, gradientes y
actualización de pesos en GPU. El entrenamiento MNIST de dos épocas (5,400 ejemplos
de entrenamiento y 600 de validación) tardó aproximadamente 4.11 segundos en esta
ejecución; obtuvo 0.896 de exactitud sobre 1,000 ejemplos de prueba.
Las comprobaciones LSTM `pre_native`, `post_native` y `post_cudnn` finalizaron
correctamente. Las salidas están en la libreta y el informe local en
`artifacts/gpu_validation/result.json`.

Estos resultados pertenecen exclusivamente al diagnóstico pequeño. No son métricas
del clasificador de reseñas ni demuestran una aceleración respecto a CPU.

## 7. Evidencia pendiente

- Verificar el archivo descargado y registrar su huella.
- Medir cobertura del vocabulario y frecuencia de términos sin vector.
- Comprobar la matriz y que sus pesos permanezcan congelados.
- GPU local validada con operaciones y entrenamiento pequeños; falta medir el
  experimento completo de reseñas y su anexo de rendimiento.
- Entrenar, revisar curvas, evaluar prueba e interpretar errores.
- Confirmar que guardar y recargar el modelo conserve predicciones.
- Completar las conclusiones con resultados reales y sus límites.

La cobertura se usará para revisar la preparación con entrenamiento; no ajustaremos
el modelo basándonos en resultados de prueba. Tampoco interpretaremos el aprendizaje
de este corpus como garantía de funcionamiento en otros hoteles o fuentes.

## 8. Idea futura: artículo de acompañamiento

**Idea registrada, no iniciada:** convertir estas notas en un artículo tipo
*how-to* o *write-up*, posiblemente HTML con el estilo editorial de la guía actual.
El notebook conservaría el procedimiento ejecutable; el artículo explicaría el
problema, decisiones, resultados y aprendizajes de forma narrativa.

El formato se decidirá después de terminar la primera versión. Por ahora este
Markdown es el documento vivo de acompañamiento. No se crea otro HTML ni se
presentan conclusiones de un experimento todavía pendiente.
