PHP LLM - Modelo de Lenguaje Pequeño en PHP

¡Bienvenido a PHP LLM! Un modelo de lenguaje pequeño (tiny) implementado completamente en PHP, basado en PPM (Prediction by Partial Matching) y tokenización simple. Ideal para experimentar, aprender y ejecutar en entornos de alojamiento compartido sin necesidad de GPUs.

Características

· ✅ Entrenamiento en texto libre o en formato de pregunta-respuesta (QA).
· ✅ Interfaz web con pestañas para entrenar, chatear y depurar.
· ✅ API compatible con OpenAI (endpoint OpenAI.php) para integrar con otras aplicaciones.
· ✅ Parámetros avanzados: temperatura, top‑K, top‑P, penalización de frecuencia, penalización de presencia y penalización de repetición.
· ✅ Persistencia: el modelo se guarda en disco (models/tiny-php/) y se recarga automáticamente.
· ✅ Historial de conversación y exportación a JSON o texto.

Archivos del proyecto

· index.php – Interfaz web principal.
· OpenAI.php – Endpoint estilo OpenAI (Chat completions).
· LLM.php – Clases Tokenizer, PPMTrie y LLM (núcleo del modelo).

Requisitos

· PHP 7.4 o superior.
· Extensiones: json, fileinfo (opcional, para algunos entornos).
· Permisos de escritura en la carpeta models/.

Instalación

1. Descarga todos los archivos (index.php, OpenAI.php, LLM.php) en la raíz de tu servidor web (por ejemplo, /var/www/html/).
2. Crea la carpeta models y dale permisos de escritura:
   ```bash
   mkdir models
   chmod 777 models
   ```
3. Accede a http://tusitio.com/index.php desde tu navegador.

¡Ya está listo para usar!

Uso básico (interfaz web)

1. Entrenar el modelo

Puedes entrenar el modelo con texto libre o con pares de preguntas/respuestas.

Entrenamiento libre (pestaña "Entrenar")

Pega cualquier texto (cuentos, documentación, conversaciones) y haz clic en "Entrenar modelo". El modelo aprenderá de ese texto.

Entrenamiento con preguntas y respuestas (pestaña "QA")

Recomendamos usar este formato para que el modelo aprenda diálogos. Escribe una pregunta y una respuesta y presiona "Entrenar QA". Internamente se concatenan y se añade el token <|EOS|>.

Formato preferido de entrenamiento (aunque no es obligatorio, da mejores resultados):

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

Puedes incluir este texto directamente en la pestaña "Entrenar".

2. Chatear con el modelo (pestaña "Chatear")

Una vez entrenado, ve a la pestaña "Chatear". Escribe un mensaje y el modelo responderá.

Puedes ajustar los parámetros de generación:

· Max tokens: longitud máxima de la respuesta.
· Temperatura: controla la creatividad (0.1 = determinista, 1.5 = más creativo).
· Top‑K: limita la selección a los K tokens con mayor probabilidad.
· Top‑P (nucleus sampling): selecciona tokens hasta acumular probabilidad P.
· Repetition Penalty: reduce la repetición de tokens ya generados (valores >1 desalientan repetición).
· Presence Penalty: penaliza tokens que ya han aparecido (positivo reduce repetición).
· Penalidad frecuencia: reduce la probabilidad de tokens según su frecuencia en la generación actual.

3. Gestión del modelo (pestaña "Debug")

· Eliminar modelo completo: borra los archivos tokenizer.json y model.ppm y reinicia el modelo desde cero.
· Exportar historial: puedes guardar la conversación en JSON o texto.

API estilo OpenAI (endpoint OpenAI.php)

Si deseas usar el modelo desde otras aplicaciones, envía peticiones POST a OpenAI.php con el siguiente formato JSON (similar a la API de OpenAI):

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

La respuesta será:

```json
{
  "success": true,
  "id": "chatcmpl-1234567890",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "PHP es un lenguaje de programación..."
      }
    }
  ]
}
```

Nota: El modelo se guarda en models/tiny-php/ (por defecto). Puedes cambiar el nombre del modelo en el campo model (se creará una subcarpeta dentro de models).

Estructura de almacenamiento del modelo

El modelo se guarda en la carpeta models/<nombre-del-modelo>/ con dos archivos:

· tokenizer.json – Vocabulario y mapeo token → id.
· model.ppm – Árbol PPM en formato binario.

Consejos para un mejor entrenamiento

· Usa el formato con <|USER|> y <|ASSISTANT|> para diálogos.
· Separa cada turno con <|EOS|>.
· Entrena con al menos varios cientos de líneas para obtener respuestas coherentes.
· Experimenta con los parámetros de generación (especialmente temperatura y top‑K) para ajustar la creatividad.

Limitaciones

· Modelo muy pequeño (contexto máximo 512 tokens). No esperes respuestas largas ni extremadamente coherentes en temas complejos.
· La tokenización es basada en expresiones regulares simples, no usa subword (BPE).
· El algoritmo PPM es rápido pero puede consumir memoria si se entrena con mucho texto.

Solución de problemas

· Error "No se puede escribir en models/" → Verifica permisos de la carpeta models.
· El modelo no responde o da respuestas vacías → Entrena con más ejemplos o revisa el formato de los mensajes.
· La interfaz muestra "El servidor devolvió HTML" → Mira la pestaña Debug para ver el error real del servidor.

---

¡Disfruta experimentando con tu propio LLM en PHP!
Cualquier mejora o sugerencia, no dudes en compartir.
