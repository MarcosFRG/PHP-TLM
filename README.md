# PHP TLM - Modelo de Lenguaje Pequeño en PHP

_(Experimental)_

¡Bienvenido a **PHP TLM**! Un modelo de lenguaje pequeño (tiny) implementado completamente en PHP que ahora utiliza una **Echo State Network (ESN)** , un tipo de red neuronal recurrente con reservorio de estado líquido. Ideal para experimentar, aprender y ejecutar en entornos de alojamiento compartido sin necesidad de GPUs.

## Características

- ✅ **Entrenamiento en texto libre** o en formato de **pregunta-respuesta** (QA).
- ✅ **Interfaz web** con pestañas para entrenar, chatear y depurar.
- ✅ **API compatible con OpenAI** (endpoint `/chat/completions`) para integrar con otras aplicaciones.
- ✅ **Parámetros avanzados**: temperatura, top‑K, top‑P, penalización de frecuencia, penalización de presencia y penalización de repetición.
- ✅ **Persistencia**: el modelo se guarda en disco (`all-models/tiny-php/`) y se recarga automáticamente.
- ✅ **Historial de conversación** y exportación a JSON o texto.
- ✅ **Arquitectura ESN**: reservorio de 500 neuronas con conexiones dispersas y fijas, solo se entrena la capa de salida.
- ✅ **Complejidad lineal O(n)**: generación rápida incluso con contextos largos.
- ✅ **Memoria de corto plazo**: el estado del reservorio se actualiza con cada token, manteniendo información contextual.

## 🧠 Arquitectura del modelo

PHP TLM ha adoptado una **Echo State Network**, un modelo recurrente eficiente donde la parte recurrente (el reservorio) se inicializa aleatoriamente y no se entrena, mientras que solo la capa de salida se ajusta mediante gradiente descendente. Esto permite un entrenamiento rápido y una buena capacidad de modelado secuencial.

### Componentes principales

| Componente | Descripción |
|------------|-------------|
| **Tokenizer** | Segmentación en tokens usando expresiones regulares (soporta caracteres Unicode y tokens especiales). |
| **Embeddings** | Vectores de 100 dimensiones para cada token, aprendidos durante el entrenamiento (solo la capa de salida). |
| **Reservorio (500 unidades)** | Red recurrente con conexiones dispersas (10% de densidad) y fijas. Su estado se actualiza con cada token. |
| **Matriz de entrada (`W_in`)** | Conexiones desde el embedding (100d) al reservorio (500d), dispersas y fijas. |
| **Matriz recurrente (`W_res`)** | Conexiones dentro del reservorio, dispersas y fijas, escaladas para controlar el radio espectral. |
| **Bias** | Vector de bias fijo para el reservorio. |
| **Capa de salida (`W_out`)** | Matriz entrenable que mapea el estado del reservorio a logits sobre el vocabulario. Se actualiza con SGD. |
| **Función de activación** | `tanh` en las neuronas del reservorio. |

### Flujo de generación

1. El prompt se tokeniza y convierte a IDs.
2. Para cada token del prompt, se actualiza el estado del reservorio:
   - Se obtiene el embedding del token.
   - Se calcula la nueva entrada combinando la contribución de la entrada (`W_in * embedding`) y la recurrente (`W_res * estado anterior`) más el bias.
   - Se aplica `tanh` para obtener el nuevo estado.
3. Durante la generación:
   - El estado actual del reservorio se multiplica por `W_out` para obtener logits sobre todo el vocabulario.
   - Se aplican temperatura, top‑K, top‑P y penalizaciones.
   - Se selecciona el siguiente token.
   - El nuevo token se procesa igual que en el paso 2, actualizando el estado.
4. La respuesta se construye concatenando los tokens generados.

Esta arquitectura **aprende patrones secuenciales de forma eficiente** y **generaliza bien con relativamente pocos parámetros entrenables**, manteniendo una velocidad de generación constante.

## Archivos del proyecto

- `index.php` – Interfaz web principal.
- `OpenAI.php` – Endpoint estilo OpenAI (Chat completions).
- `Models.php` – Endpoint que muestra la lista de modelos disponibles.
- `LLM.php` – Clases `Tokenizer`, `EchoStateLM` y `LLM`.

## Requisitos

- PHP 7.4 o superior.
- Extensiones: `json`, `fileinfo` (opcional, para algunos entornos).
- Permisos de escritura en la carpeta `all-models/`.

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
Pega cualquier texto (cuentos, documentación, conversaciones) y haz clic en **"Entrenar modelo"**. El modelo procesará el texto dividiéndolo en lotes separados por el patrón `EOS>\n\n<|` (es decir, cada vez que aparece `<|EOS|>` seguido de línea vacía y otro token especial). Si falta `<|EOS|>` al final de un lote, se añade automáticamente.

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

> **Nota importante**: La arquitectura ESN requiere **entrenamiento sustancial** para que la capa de salida aprenda representaciones útiles. Para obtener respuestas coherentes, necesitarás al menos varios cientos de ejemplos o un texto largo y variado. La generación será más lenta al principio, pero mejorará a medida que el modelo aprenda.

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

- **Eliminar modelo completo**: borra los archivos `tokenizer.json` y `model.bin` y reinicia el modelo desde cero.
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

El modelo se guarda en la carpeta `all-models/<nombre-del-modelo>/` con dos archivos:

- `tokenizer.json` – Vocabulario y mapeo token → id.
- `model.bin` – Pesos del modelo ESN en formato binario (incluye embeddings, matrices dispersas, bias y capa de salida).

## Consejos para un mejor entrenamiento

- Usa el formato con `<|USER|>` y `<|ASSISTANT|>` para diálogos.
- Separa cada turno con `<|EOS|>`.
- **Entrena con mucho texto**: La ESN necesita datos para ajustar la capa de salida. Cuanto más variado, mejor.
- Experimenta con los parámetros de generación (especialmente temperatura y top‑K) para ajustar la creatividad.
- Si el modelo no genera bien al principio, **sigue entrenando**. La capa de salida tarda en converger.

## Limitaciones

- Modelo de tamaño moderado (reservorio 500, embeddings 100d). No esperes respuestas extremadamente coherentes en temas complejos sin suficiente entrenamiento.
- La tokenización es basada en expresiones regulares simples, no usa subword (BPE).
- El reservorio es fijo y no se adapta durante el entrenamiento; toda la plasticidad recae en la capa de salida.
- La generación es rápida (O(n) lineal), pero el entrenamiento puede ser lento con lotes muy grandes.

## Solución de problemas

- **Error "No se puede escribir en models/"** → Verifica permisos de la carpeta `all-models`.
- **El modelo no responde o da respuestas vacías** → Entrena con más ejemplos o revisa el formato de los mensajes.
- **La interfaz muestra "El servidor devolvió HTML"** → Mira la pestaña **Debug** para ver el error real del servidor.
- **Generación muy lenta** → Reduce el tamaño del reservorio (`reservoirSize` en `EchoStateLM`) o la dimensión de embeddings (`embedDim`).

## Historial de versiones

- **v0.1-alpha**: Modelo basado únicamente en PPM (estadístico).
- **v0.2-alpha**: Introducción de embeddings y caché semántico híbrido.
- **v0.3-alpha**: Arquitectura transformer-like con atención lineal, capas convolucionales y mezcladores.
- **v0.4-beta**: Arquitectura RWKV completa con time mixing, channel mixing y estados recurrentes.
- **v0.5-beta**: **Arquitectura Echo State Network (ESN)** con reservorio fijo y capa de salida entrenable. **Mayor eficiencia y buena capacidad de modelado secuencial**, velocidad lineal O(n).

---

¡Disfruta experimentando con tu propio LLM en PHP!  
Cualquier mejora o sugerencia, no dudes en compartir.
