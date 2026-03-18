# PHP TLM - Modelo de Lenguaje Pequeño en PHP

_(Experimental)_

¡Bienvenido a **PHP TLM**! Un modelo de lenguaje pequeño (tiny) implementado completamente en PHP que ahora utiliza una **arquitectura RWKV con entrenamiento completo por retropropagación (BPTT)** . Ideal para experimentar, aprender y ejecutar en entornos de alojamiento compartido sin necesidad de GPUs.

## Características

- ✅ **Entrenamiento en texto libre** o en formato de **pregunta-respuesta** (QA).
- ✅ **Interfaz web** con pestañas para entrenar, chatear y depurar.
- ✅ **API compatible con OpenAI** (endpoint `/chat/completions`) para integrar con otras aplicaciones.
- ✅ **Parámetros avanzados**: temperatura, top‑K, top‑P, penalización de frecuencia, penalización de presencia y penalización de repetición.
- ✅ **Persistencia**: el modelo se guarda en disco (`all-models/tiny-php/`) y se recarga automáticamente.
- ✅ **Historial de conversación** y exportación a JSON o texto.
- ✅ **Arquitectura RWKV con BPTT**: 4 capas con time mixing y channel mixing, entrenables mediante retropropagación completa.
- ✅ **Optimizador Adam**: implementado desde cero para una convergencia estable.
- ✅ **Embeddings normalizados**: 128 dimensiones, aprendidos durante el entrenamiento.
- ✅ **Complejidad lineal O(n)**: generación rápida incluso con contextos largos.

## 🧠 Arquitectura del modelo

PHP TLM implementa **RWKV (Receptance Weighted Key Value)** , un modelo de vanguardia que combina la eficiencia de las RNNs con la calidad de los transformers. La arquitectura actual incluye **entrenamiento completo por retropropagación a través del tiempo (BPTT)** , similar a como se entrenan los modelos GPT.

### Componentes principales

| Componente | Descripción |
|------------|-------------|
| **Tokenizer** | Segmentación en tokens usando expresiones regulares (soporta caracteres Unicode y tokens especiales). |
| **Embeddings** | Vectores de 128 dimensiones para cada token, normalizados y aprendidos durante el entrenamiento. |
| **RWKVBlock (x4)** | Bloque RWKV con time mixing y channel mixing, totalmente entrenable. |
| │ ├─ **wk, wv, wr** | Pesos para key, value y receptance (matrices [dim][dim]). |
| │ ├─ **ww** | Vector de decay aprendible (dim). |
| │ ├─ **w1, w2** | Pesos del feed-forward network (channel mixing). |
| │ └─ **cache** | Almacena variables del forward para usar en el backward. |
| **AdamOptimizer** | Optimizador con momento y adaptación por parámetro (beta1=0.9, beta2=0.999). |
| **BPTT** | Backpropagation Through Time completo, con gradientes calculados manualmente para cada operación. |

### Flujo de entrenamiento

1. El texto se tokeniza y convierte a secuencia de IDs.
2. **Forward pass**: se procesa cada token secuencialmente, guardando:
   - Estados de cada capa después de cada paso.
   - Salidas de la última capa.
   - Logits (similitud coseno con embeddings).
3. **Cálculo de pérdida**: cross-entropy entre logits y el siguiente token real.
4. **Backward pass (BPTT)** : desde el último paso al primero:
   - Se calcula el gradiente de la pérdida respecto a los logits.
   - Se propaga hacia atrás a través de las capas RWKV usando las derivadas almacenadas en caché.
   - Se acumulan gradientes de estados previos para propagarlos a pasos anteriores.
5. **Actualización de pesos**:
   - Los gradientes acumulados en cada bloque RWKV se aplican mediante Adam.
   - Los embeddings de entrada y salida se actualizan con SGD simple.
6. **Guardado**: se persisten tokenizer, embeddings y pesos RWKV en disco.

### Flujo de generación

1. El prompt se tokeniza y convierte a IDs.
2. Se inicializan los estados de todas las capas a cero.
3. Para cada token del prompt, se actualizan los estados mediante forward.
4. Durante la generación:
   - El último vector de salida se normaliza y se compara con todos los embeddings (similitud coseno) para obtener logits.
   - Se aplican temperatura, top‑K, top‑P y penalizaciones.
   - Se selecciona el siguiente token.
   - El nuevo token se procesa (forward) actualizando los estados.
5. La respuesta se construye concatenando los tokens generados.

Esta arquitectura **aprende patrones complejos** gracias a la retropropagación completa y **generaliza mejor** que versiones anteriores, manteniendo una velocidad de generación lineal.

## Archivos del proyecto

- `index.php` – Interfaz web principal.
- `OpenAI.php` – Endpoint estilo OpenAI (Chat completions).
- `Models.php` – Endpoint que muestra la lista de modelos disponibles.
- `LLM.php` – Clases `Tokenizer`, `AdamOptimizer`, `RWKVBlock` y `LLM`.

## Requisitos

- PHP 7.4 o superior.
- Extensiones: `json`, `fileinfo` (opcional, para algunos entornos).
- Permisos de escritura en la carpeta `all-models/`.
- **Memoria recomendada**: al menos 256MB para entrenamiento (puede llegar a 512MB con lotes grandes).

## Instalación

1. **Descarga** todos los archivos (`index.php`, `OpenAI.php`, `LLM.php`) en la **raíz** de tu servidor web (por ejemplo, `/var/www/html/`).
2. **Crea la carpeta `all-models`** y dale permisos de escritura:

   ```bash
   mkdir all-models
   chmod 777 all-models
   ```

3. **Accede** a `http://tusitio.com/index.php` desde tu navegador.

¡Ya está listo para usar!

## Uso básico (interfaz web)

### 1. Entrenar el modelo

Puedes entrenar el modelo con texto libre o con pares de preguntas/respuestas.

#### Entrenamiento libre (pestaña "Entrenar")
Pega cualquier texto (cuentos, documentación, conversaciones) y haz clic en **"Entrenar modelo"**. El modelo procesará el texto completo aplicando BPTT. Si el texto no termina con `<|EOS|>`, se añade automáticamente.

#### Entrenamiento con preguntas y respuestas (pestaña "QA")
Recomendamos usar este formato para que el modelo aprenda diálogos. Escribe una **pregunta** y una **respuesta** y presiona **"Entrenar QA"**. Internamente se concatenan y se añade el token `<|EOS|>`.

**Formato preferido de entrenamiento** (aunque no es obligatorio, da mejores resultados):

```
<|USER|>
¿Sabes PHP?
<|EOS|>
<|ASSISTANT|>
Sí, PHP es mi lenguaje nativo 💻
<|EOS|>
<|USER|>
Haz un loop
<|EOS|>
<|ASSISTANT|>
for($i=0;$i<10;$i++){ echo $i; }
<|EOS|>
```

Puedes incluir este texto directamente en la pestaña **"Entrenar"**.

> **Nota importante**: Esta versión utiliza **entrenamiento completo por retropropagación**, lo que requiere más memoria y tiempo que versiones anteriores, pero ofrece **mucho mejor capacidad de aprendizaje**. Para obtener resultados coherentes, se recomienda entrenar con al menos varios miles de tokens y repetir el entrenamiento varias veces sobre el mismo corpus.

### 2. Chatear con el modelo (pestaña "Chatear")

Una vez entrenado, ve a la pestaña **"Chatear"**. Escribe un mensaje y el modelo responderá.

Puedes ajustar los parámetros de generación:

- **Max tokens**: longitud máxima de la respuesta.
- **Temperatura**: controla la creatividad (0.1 = determinista, 1.5 = más creativo).
- **Top‑K**: limita la selección a los K tokens con mayor probabilidad.
- **Top‑P** (nucleus sampling): selecciona tokens hasta acumular probabilidad P.
- **Repetition Penalty**: reduce la repetición de tokens ya generados (valores >1 desalientan repetición).
- **Presence Penalty**: penaliza tokens que ya han aparecido (positivo reduce repetición).
- **Penalidad frecuencia**: reduce la probabilidad de tokens según su frecuencia en la generación actual.

### 3. Gestión del modelo (pestaña "Debug")

- **Eliminar modelo completo**: borra los archivos `tokenizer.json`, `embeddings.bin` y `rwkv.bin` y elimina la carpeta del modelo.
- **Exportar historial**: puedes guardar la conversación en JSON o texto.

## API estilo OpenAI (endpoint `OpenAI.php`)

Si deseas usar el modelo desde otras aplicaciones, envía peticiones POST a `/chat/completions` (o directamente al archivo `OpenAI.php`) con el siguiente formato JSON (similar a la API de OpenAI):

```json
{
  "model": "tiny-php",
  "messages": [
    {"role": "system", "content": "Eres un asistente útil."},
    {"role": "user", "content": "¿Qué es PHP?"}
  ],
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 1,
  "top_k": 10,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0
}
```

La respuesta será algo como:

```json
{
  "success": true,
  "id": "chatcmpl-67d8f1a2b3c4d",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "PHP es un lenguaje de programación..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 37,
    "total_tokens": 79
  },
  "timing_ms": 234
}
```

**Nota:** El modelo se guarda en `all-models/tiny-php/` (por defecto). Puedes cambiar el nombre del modelo en el campo `model` (se creará una subcarpeta dentro de `all-models`).

## Estructura de almacenamiento del modelo

El modelo se guarda en la carpeta `all-models/<nombre-del-modelo>/` con tres archivos:

- `tokenizer.json` – Vocabulario y mapeo token → id.
- `embeddings.bin` – Vectores de embeddings (128d) en formato binario.
- `rwkv.bin` – Pesos serializados de los bloques RWKV (wk, wv, wr, ww, w1, w2).

## Consejos para un mejor entrenamiento

- Usa el formato con `<|USER|>` y `<|ASSISTANT|>` para diálogos.
- Separa cada turno con `<|EOS|>`.
- **Entrena con lotes grandes**: Esta versión procesa todo el texto de una vez, así que asegúrate de tener suficiente memoria.
- Si el modelo no genera bien al principio, **repite el entrenamiento** varias veces sobre el mismo corpus. La retropropagación necesita múltiples épocas.
- Ajusta `$learningRate` en `LLM.php` (0.001 por defecto) si la pérdida no disminuye o si hay inestabilidad.
- Experimenta con `$embedDim` y `$numLayers` (requiere reiniciar el modelo desde cero).

## Limitaciones

- Modelo de tamaño moderado (embeddings 128d, 4 capas, ~500k parámetros). No esperes respuestas extremadamente coherentes en temas complejos sin suficiente entrenamiento.
- El entrenamiento BPTT puede consumir mucha memoria (proporcional a `longitud del texto * embedDim * numLayers`). Para textos muy largos, considera dividirlos manualmente.
- La tokenización es basada en expresiones regulares simples, no usa subword (BPE).
- El cálculo de gradientes es manual y puede tener inestabilidades numéricas en casos extremos (se incluyen protecciones como `1e-8` en divisiones).

## Solución de problemas

- **Error "No se puede escribir en models/"** → Verifica permisos de la carpeta `all-models`.
- **El modelo no responde o da respuestas vacías** → Entrena con más ejemplos o revisa el formato de los mensajes.
- **La interfaz muestra "El servidor devolvió HTML"** → Mira la pestaña **Debug** para ver el error real del servidor.
- **Error de memoria** → Reduce el tamaño del texto de entrenamiento o disminuye `$embedDim` y `$numLayers`.
- **Generación muy lenta** → Reduce `$maxTokens` o aumenta la memoria disponible.

## Historial de versiones

- **v0.1-alpha**: Modelo basado únicamente en PPM (estadístico).
- **v0.2-alpha**: Introducción de embeddings y caché semántico híbrido.
- **v0.3-alpha**: Arquitectura transformer-like con atención lineal, capas convolucionales y mezcladores.
- **v0.4-beta**: Arquitectura RWKV completa con time mixing, channel mixing y estados recurrentes.
- **v0.5-beta**: Arquitectura Echo State Network (ESN) con reservorio fijo.
- **v0.6-beta**: **RWKV con entrenamiento completo por retropropagación (BPTT) y optimizador Adam**. **Máxima capacidad de aprendizaje**, similar a modelos GPT.

---

¡Disfruta experimentando con tu propio LLM en PHP!  
Cualquier mejora o sugerencia, no dudes en compartir.
