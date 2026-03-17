# PHP TLM - Modelo de Lenguaje Pequeño en PHP

¡Bienvenido a **PHP TLM**! Un modelo de lenguaje pequeño (tiny) implementado completamente en PHP que ha evolucionado hasta convertirse en una **arquitectura RWKV (Receptance Weighted Key Value)** completa, combinando lo mejor de transformers y RNNs. Ideal para experimentar, aprender y ejecutar en entornos de alojamiento compartido sin necesidad de GPUs.

## Características

- ✅ **Entrenamiento en texto libre** o en formato de **pregunta-respuesta** (QA).
- ✅ **Interfaz web** con pestañas para entrenar, chatear y depurar.
- ✅ **API compatible con OpenAI** (endpoint `/chat/completions`) para integrar con otras aplicaciones.
- ✅ **Parámetros avanzados**: temperatura, top‑K, top‑P, penalización de frecuencia, penalización de presencia y penalización de repetición.
- ✅ **Persistencia**: el modelo se guarda en disco (`models/tiny-php/`) y se recarga automáticamente.
- ✅ **Historial de conversación** y exportación a JSON o texto.
- ✅ **Arquitectura RWKV**: 4 bloques con time mixing y channel mixing, estados recurrentes.
- ✅ **Complejidad lineal O(n)**: generación rápida incluso con contextos largos.
- ✅ **Memoria recurrente**: cada capa mantiene su propio estado (NUM y DEN) para contexto infinito teórico.

## 🧠 Arquitectura del modelo

PHP TLM ha migrado de una arquitectura transformer-like a **RWKV**, un modelo de vanguardia que combina la eficiencia de las RNNs con la calidad de los transformers. La arquitectura actual es:

### Componentes principales

| Componente | Descripción |
|------------|-------------|
| **Tokenizer** | Segmentación en tokens usando expresiones regulares (soporta caracteres Unicode y tokens especiales). |
| **Embeddings** | Vectores de 128 dimensiones para cada token, aprendidos durante el entrenamiento. |
| **PPMTrie** | Modelo estadístico clásico que complementa las predicciones neuronales (opcional). |
| **RWKVBlock** | Bloque RWKV que incluye: |
| │ ├─ **wk, wv, wr** | Pesos para key, value y receptance. |
| │ ├─ **ww** | Vector de decay (aprendible) para cada dimensión. |
| │ ├─ **w1, w2** | Pesos del feed-forward network (channel mixing). |
| │ └─ **state** | Estado recurrente compuesto por `num` y `den` para cada dimensión. |
| **WKV computation** | Núcleo de RWKV: atención lineal recurrente que actualiza el estado. |
| **Time mixing** | Mezcla información temporal usando el estado recurrente y el decay. |
| **Channel mixing** | FFN con activación ReLU y gating (similar a MLP-Mixer). |

### Flujo de generación

1. El prompt se tokeniza y convierte a IDs.
2. Para cada token del prompt, se ejecuta `forwardRWKV`, que:
   - Toma el embedding del token.
   - Lo pasa por cada bloque RWKV, actualizando los estados recurrentes.
3. Durante la generación:
   - El último vector de salida se compara con todos los embeddings (similitud coseno).
   - Se aplican temperatura, top‑K, top‑P y penalizaciones.
   - Se selecciona el siguiente token.
   - El nuevo token se procesa con `forwardRWKV`, actualizando los estados.
4. La respuesta se construye concatenando los tokens generados.

Esta arquitectura **entiende mejor las relaciones semánticas** y **generaliza de forma más natural** que las versiones anteriores, manteniendo una velocidad de generación competitiva gracias a su complejidad lineal.

## Archivos del proyecto

- `index.php` – Interfaz web principal.
- `OpenAI.php` – Endpoint estilo OpenAI (Chat completions).
- `LLM.php` – Clases `Tokenizer`, `PPMTrie`, `RWKVBlock` y `LLM`.

## Requisitos

- PHP 7.4 o superior.
- Extensiones: `json`, `fileinfo` (opcional, para algunos entornos).
- Permisos de escritura en la carpeta `models/`.

## Instalación

1. **Descarga** todos los archivos (`index.php`, `OpenAI.php`, `LLM.php`) en la **raíz** de tu servidor web (por ejemplo, `/var/www/html/`).
2. **Crea la carpeta `models`** y dale permisos de escritura:

   ```bash
   mkdir models
   chmod 777 models
   ```

3. **Accede** a `http://tusitio.com/index.php` desde tu navegador.

¡Ya está listo para usar!

## Uso básico (interfaz web)

### 1. Entrenar el modelo

Puedes entrenar el modelo con texto libre o con pares de preguntas/respuestas.

#### Entrenamiento libre (pestaña "Entrenar")
Pega cualquier texto (cuentos, documentación, conversaciones) y haz clic en **"Entrenar modelo"**. El modelo procesará el texto en lotes separados por líneas vacías, añadiendo automáticamente `<|EOS|>` si falta.

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

> **Nota importante**: La arquitectura RWKV requiere **entrenamiento sustancial** para que los pesos aprendan representaciones útiles. Para obtener respuestas coherentes, necesitarás al menos varios cientos de ejemplos o un texto largo y variado. La generación será más lenta al principio, pero mejorará a medida que el modelo aprenda.

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

- **Eliminar modelo completo**: borra los archivos `tokenizer.json`, `model.ppm`, `embeddings.bin` y `rwkv.bin` y reinicia el modelo desde cero.
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

**Nota:** El modelo se guarda en `models/tiny-php/` (por defecto). Puedes cambiar el nombre del modelo en el campo `model` (se creará una subcarpeta dentro de `models`).

## Estructura de almacenamiento del modelo

El modelo se guarda en la carpeta `models/<nombre-del-modelo>/` con cuatro archivos:

- `tokenizer.json` – Vocabulario y mapeo token → id.
- `model.ppm` – Árbol PPM en formato binario.
- `embeddings.bin` – Vectores de embeddings en formato binario.
- `rwkv.bin` – Pesos serializados de los bloques RWKV.

## Consejos para un mejor entrenamiento

- Usa el formato con `<|USER|>` y `<|ASSISTANT|>` para diálogos.
- Separa cada turno con `<|EOS|>`.
- **Entrena con mucho texto**: RWKV necesita datos para ajustar sus pesos. Cuanto más variado, mejor.
- Experimenta con los parámetros de generación (especialmente temperatura y top‑K) para ajustar la creatividad.
- Si el modelo no genera bien al principio, **sigue entrenando**. Los pesos RWKV tardan en converger.

## Limitaciones

- Modelo de tamaño moderado (embeddings 128d, 4 capas). No esperes respuestas extremadamente coherentes en temas complejos sin suficiente entrenamiento.
- La tokenización es basada en expresiones regulares simples, no usa subword (BPE).
- El algoritmo PPM y los bloques RWKV pueden consumir memoria si se entrena con mucho texto.
- La generación es más rápida que en transformers gracias a la recurrencia, pero el entrenamiento inicial puede ser lento.

## Solución de problemas

- **Error "No se puede escribir en models/"** → Verifica permisos de la carpeta `models`.
- **El modelo no responde o da respuestas vacías** → Entrena con más ejemplos o revisa el formato de los mensajes.
- **La interfaz muestra "El servidor devolvió HTML"** → Mira la pestaña **Debug** para ver el error real del servidor.
- **Generación muy lenta** → Reduce el número de capas (`$numLayers` en `LLM.php`) o la dimensión de embeddings (`$embedDim`). O simplemente ten paciencia mientras el modelo aprende.

## Historial de versiones

- **v0.1-alpha**: Modelo basado únicamente en PPM (estadístico).
- **v0.2-alpha**: Introducción de embeddings y caché semántico híbrido.
- **v0.3-alpha**: Arquitectura transformer-like con atención lineal, capas convolucionales y mezcladores.
- **v0.4-alpha**: **Arquitectura RWKV completa** con time mixing, channel mixing y estados recurrentes. **Mayor capacidad de comprensión y generalización**, velocidad lineal O(n).

---

¡Disfruta experimentando con tu propio LLM en PHP!  
Cualquier mejora o sugerencia, no dudes en compartir.
