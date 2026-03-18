<?php
require_once 'LLM.php';
session_start();
$modelDir = 'models/tiny-php';
$maxContext = 512;
$llm = new LLM($modelDir, $maxContext);
$action = $_POST['action'] ?? $_GET['action'] ?? '';
if(!isset($_SESSION['chat_history'])) $_SESSION['chat_history'] = [];
if ($action === 'train' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    try {
        $text = $_POST['text'] ?? '';
        if (empty($text)) {
            echo json_encode(['success' => false, 'message' => 'El texto no puede estar vacío.']);
            exit;
        }

        // Dividir el texto por lotes usando líneas vacías como separadores
        $batches = preg_split('/\n\s*\n/', $text);
        $totalTokens = 0;
        $processedBatches = 0;
        $errors = [];

        foreach ($batches as $batch) {
            $batch = trim($batch);
            if (empty($batch)) continue;

            try {
                $llm->train($batch);
                $totalTokens = $llm->getVocabSize();
                $processedBatches++;
            } catch (Exception $e) {
                $errors[] = "Error en lote " . ($processedBatches + 1) . ": " . $e->getMessage();
            }
        }

        $message = "Modelo entrenado con $processedBatches lote(s). Vocabulario: $totalTokens tokens.";
        if (!empty($errors)) $message .= " Errores: " . implode('; ', $errors);

        echo json_encode([
            'success' => true, 
            'message' => $message,
            'batches_processed' => $processedBatches,
            'total_tokens' => $totalTokens,
            'received_length' => strlen($text),
            'received_preview' => substr($text, 0, 200) . (strlen($text) > 200 ? '…' : '')
        ]);
    } catch (Exception $e) {
        echo json_encode(['success' => false, 'message' => 'Error: ' . $e->getMessage()]);
    }
    exit;
}
if ($action === 'train_qa' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    try {
        $question = $_POST['question'] ?? '';
        $answer = $_POST['answer'] ?? '';
        if (empty($question) || empty($answer)) {
            echo json_encode(['success' => false, 'message' => 'La pregunta y la respuesta no pueden estar vacías.']);
            exit;
        }
        $text = "<|USER|>\n$question\n<|EOS|>\n<|ASSISTANT|>\n$answer\n<|EOS|>";
        $llm->train($text);
        echo json_encode([
            'success' => true,
            'message' => 'Modelo entrenado con QA. Vocabulario: ' . $llm->getVocabSize() . ' tokens.',
            'received_length' => strlen($text),
            'received_preview' => substr($text, 0, 200) . (strlen($text) > 200 ? '…' : '')
        ]);
    } catch (Exception $e) {
        echo json_encode(['success' => false, 'message' => 'Error: ' . $e->getMessage()]);
    }
    exit;
}
if ($action === 'chat' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    try {
        $message = $_POST['message'] ?? '';
        $maxTokens = (int)($_POST['max_tokens'] ?? 50);
        $temperature = (float)($_POST['temperature'] ?? 0.8);
        $topK = isset($_POST['top_k']) ? (int)$_POST['top_k'] : null;
        $frequencyPenalty = (float)($_POST['frequency_penalty'] ?? 0.0);
        // Nuevos parámetros
        $topP = isset($_POST['top_p']) ? (float)$_POST['top_p'] : 1.0;
        $repetitionPenalty = (float)($_POST['repetition_penalty'] ?? 1.0);
        $presencePenalty = (float)($_POST['presence_penalty'] ?? 0.0);

        if (empty($message)) {
            echo json_encode(['response' => '']);
            exit;
        }
        $response = $llm->generate($message, $maxTokens, $temperature, $topK, $frequencyPenalty, [], $topP, $repetitionPenalty, $presencePenalty);
        $_SESSION['chat_history'][] = [
            'user' => $message,
            'bot' => $response,
            'timestamp' => date('Y-m-d H:i:s')
        ];
        echo json_encode(['response' => $response]);
    } catch (Exception $e) {
        echo json_encode(['response' => 'Error: ' . $e->getMessage()]);
    }
    exit;
}
if ($action === 'export' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    try {
        $format = $_POST['format'] ?? 'json';
        $history = $_SESSION['chat_history'] ?? [];
        if ($format === 'json') {
            echo json_encode([
                'success' => true,
                'data' => $history,
                'count' => count($history)
            ]);
        } elseif ($format === 'text') {
            $text = "=== CONVERSACIÓN EXPORTADA ===\n";
            $text .= "Fecha: " . date('Y-m-d H:i:s') . "\n";
            $text .= "Total mensajes: " . count($history) . "\n\n";
            foreach ($history as $i => $msg) {
                $text .= "[{$msg['timestamp']}]\n";
                $text .= "👤 Usuario: {$msg['user']}\n";
                $text .= "🤖 Bot: {$msg['bot']}\n";
                $text .= str_repeat("-", 50) . "\n\n";
            }
            echo json_encode([
                'success' => true,
                'data' => $text,
                'count' => count($history)
            ]);
        } else {
            echo json_encode(['success' => false, 'message' => 'Formato no soportado']);
        }
    } catch (Exception $e) {
        echo json_encode(['success' => false, 'message' => 'Error: ' . $e->getMessage()]);
    }
    exit;
}
if ($action === 'clear_history' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    $_SESSION['chat_history'] = [];
    echo json_encode(['success' => true, 'message' => 'Historial borrado']);
    exit;
}
if ($action === 'set_topk' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    $topK = (int)($_POST['top_k'] ?? 10);
    $llm->setDefaultTopK($topK);
    echo json_encode(['success' => true, 'message' => "Top-K cambiado a $topK"]);
    exit;
}
if ($action === 'delete_model' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    try {
        $files = [
            $modelDir . '/tokenizer.json',
            $modelDir . '/model.ppm',
            $modelDir . '/embeddings.bin',
            $modelDir . '/rwkv.bin'
        ];
        $deleted = [];
        $errors = [];
        foreach ($files as $file) {
            if (file_exists($file)) {
                if (unlink($file)) {
                    $deleted[] = basename($file);
                } else {
                    $errors[] = basename($file);
                }
            }
        }
        $_SESSION['chat_history'] = [];
        if(!is_dir($modelDir)) mkdir($modelDir, 0777, true);
        $llm = null;
        $llm = new LLM($modelDir, $maxContext);
        echo json_encode([
            'success' => true,
            'message' => 'Modelo eliminado correctamente.',
            'deleted' => $deleted,
            'errors' => $errors
        ]);
    } catch (Exception $e) {
        echo json_encode(['success' => false, 'message' => 'Error al eliminar: ' . $e->getMessage()]);
    }
    exit;
}
?>
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Producción - Con Top-K, Top-P y Penalizaciones</title>
    <style>
        * { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }
        body { background: #f2f4f8; margin: 0; min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 16px; }
        .container { max-width: 1100px; width: 100%; background: white; border-radius: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow: hidden; }
        .tabs { display: flex; background: #f0f2f5; border-bottom: 1px solid #ddd; flex-wrap: wrap; }
        .tab { flex: 1; text-align: center; padding: 16px; font-weight: 600; cursor: pointer; min-width: 120px; }
        .tab.active { background: white; border-bottom: 2px solid #1a73e8; color: #1a73e8; }
        .content { padding: 24px; display: none; }
        .content.active { display: block; }
        h2 { margin-top: 0; color: #1e293b; }
        label { display: block; margin: 16px 0 6px; font-weight: 500; }
        textarea, input[type="number"], select { width: 100%; padding: 12px; border: 1px solid #ccc; border-radius: 12px; font-size: 1rem; }
        button { background: #1a73e8; color: white; border: none; border-radius: 40px; padding: 12px 24px; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 16px; margin-right: 8px; }
        button:hover { background: #1557b0; }
        button.secondary { background: #6c757d; }
        button.secondary:hover { background: #5a6268; }
        button.success { background: #28a745; }
        button.success:hover { background: #218838; }
        button.warning { background: #ffc107; color: #212529; }
        button.warning:hover { background: #e0a800; }
        button.danger { background: #dc3545; color: white; }
        button.danger:hover { background: #c82333; }
        .button-group { display: flex; gap: 8px; flex-wrap: wrap; }
        .message { margin-top: 16px; padding: 12px; border-radius: 12px; display: none; }
        .success { background: #d1fae5; color: #065f46; display: block; }
        .error { background: #fee2e2; color: #991b1b; display: block; }
        #chat-area { height: 350px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 16px; padding: 16px; margin-bottom: 16px; background: #fafafa; display: flex; flex-direction: column; gap: 8px; }
        .chat-message { max-width: 80%; padding: 10px 16px; border-radius: 18px; word-wrap: break-word; }
        .user-message { align-self: flex-end; background: #1a73e8; color: white; border-bottom-right-radius: 4px; }
        .bot-message { align-self: flex-start; background: white; border: 1px solid #ddd; border-bottom-left-radius: 4px; }
        .chat-input { display: flex; gap: 8px; }
        .chat-input input { flex: 1; padding: 12px; border-radius: 40px; border: 1px solid #ccc; }
        .params { display: flex; gap: 16px; margin-top: 12px; margin-bottom: 12px; flex-wrap: wrap; }
        .params div { flex: 1; min-width: 140px; }
        .loader { display: inline-block; width: 16px; height: 16px; border: 2px solid #ccc; border-top-color: #1a73e8; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .debug-box { background: #1e1e2f; color: #00ff99; padding: 16px; border-radius: 12px; margin-top: 20px; font-family: monospace; white-space: pre-wrap; max-height: 300px; overflow-y: auto; border: 1px solid #333; display: none; }
        .debug-box.visible { display: block; }
        .debug-header { display: flex; justify-content: space-between; margin-bottom: 10px; color: #aaa; }
        .debug-title { font-weight: bold; color: #fff; }
        .debug-clear { cursor: pointer; color: #ff6b6b; }
        .stats { display: flex; gap: 20px; margin: 10px 0; color: #555; font-size: 0.9rem; flex-wrap: wrap; }
        .stat-item { background: #f8f9fa; padding: 8px 12px; border-radius: 20px; }
        .error-detail { color: #ff6b6b; font-family: monospace; margin-top: 10px; padding: 10px; background: #2d2d2d; border-radius: 8px; white-space: pre-wrap; }
        .history-controls { margin-bottom: 16px; }
        .timestamp { font-size: 0.7rem; color: #999; margin-top: 2px; text-align: right; }
        .export-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); justify-content: center; align-items: center; z-index: 1000; }
        .export-modal.active { display: flex; }
        .modal-content { background: white; padding: 24px; border-radius: 16px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; }
        .modal-content pre { background: #f4f4f4; padding: 12px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
        .info-badge { background: #e7f3ff; color: #1a73e8; padding: 4px 12px; border-radius: 40px; font-size: 0.8rem; display: inline-block; margin-left: 10px; }
        .model-stats { margin-top: 20px; padding: 16px; background: #f8f9fa; border-radius: 12px; border: 1px solid #e0e0e0; }
    </style>
</head>
<body>
<div class="container">
    <div class="tabs">
        <div class="tab active" data-tab="train">Entrenar</div>
        <div class="tab" data-tab="qa">QA</div>
        <div class="tab" data-tab="chat">Chatear <span class="info-badge">Top-K/P</span></div>
        <div class="tab" data-tab="debug">Debug</div>
    </div>
    <div id="train" class="content active">
        <h2>Entrenar modelo</h2>
        <form id="train-form">
            <label>Texto de entrenamiento:</label>
            <textarea id="train-text" rows="10" placeholder="Pega aquí tu texto..."></textarea>
            <div class="stats">
                <span class="stat-item" id="char-count">0 caracteres</span>
                <span class="stat-item" id="word-count">0 palabras</span>
            </div>
            <button type="submit" id="train-btn">🚀 Entrenar modelo</button>
        </form>
        <div id="train-message" class="message"></div>
    </div>
    <div id="qa" class="content">
        <h2>Entrenar con preguntas y respuestas</h2>
        <form id="qa-form">
            <label>Pregunta:</label>
            <textarea id="qa-question" rows="3" placeholder="Escribe la pregunta..."></textarea>
            <label>Respuesta:</label>
            <textarea id="qa-answer" rows="3" placeholder="Escribe la respuesta..."></textarea>
            <div class="stats">
                <span class="stat-item" id="qa-char-count">0 caracteres totales</span>
                <span class="stat-item" id="qa-word-count">0 palabras totales</span>
            </div>
            <button type="submit" id="qa-btn">🎓 Entrenar QA</button>
        </form>
        <div id="qa-message" class="message"></div>
    </div>
    <div id="chat" class="content">
        <h2>Chatear <span class="info-badge">Con Top-K, Top-P y Penalizaciones</span></h2>
        <div class="history-controls button-group">
            <button class="secondary" id="export-json-btn">📥 Exportar JSON</button>
            <button class="secondary" id="export-text-btn">📥 Exportar Texto</button>
            <button class="warning" id="clear-history-btn">🗑️ Borrar historial</button>
        </div>
        <div id="chat-area">
            <?php if (!empty($_SESSION['chat_history'])): ?>
                <?php foreach ($_SESSION['chat_history'] as $msg): ?>
                    <div class="chat-message user-message"><?php echo htmlspecialchars($msg['user']); ?></div>
                    <div class="chat-message bot-message">
                        <?php echo htmlspecialchars($msg['bot']); ?>
                        <div class="timestamp"><?php echo $msg['timestamp']; ?></div>
                    </div>
                <?php endforeach; ?>
            <?php else: ?>
                <div class="chat-message bot-message">Modelo listo. Escribe algo...</div>
            <?php endif; ?>
        </div>
        <div class="chat-input">
            <input type="text" id="chat-input" placeholder="Mensaje..." autocomplete="off">
            <button id="send-btn">Enviar</button>
        </div>
        <div class="params">
            <div>
                <label>Max tokens</label>
                <input type="number" id="max-tokens" value="30" min="1" max="200">
            </div>
            <div>
                <label>Temperatura</label>
                <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
            </div>
            <div>
                <label>Top-K</label>
                <select id="top-k">
                    <option value="1">1</option>
                    <option value="3">3</option>
                    <option value="5">5</option>
                    <option value="10">10</option>
                    <option value="20">20</option>
                    <option value="50" selected>50</option>
                    <option value="0">0 (todas)</option>
                </select>
            </div>
            <div>
                <label>Top-P</label>
                <input type="number" id="top-p" value="1.0" min="0.0" max="1.0" step="0.05">
            </div>
            <div>
                <label>Repetition Penalty</label>
                <input type="number" id="repetition-penalty" value="1.0" min="0.1" max="2.0" step="0.1">
            </div>
            <div>
                <label>Presence Penalty</label>
                <input type="number" id="presence-penalty" value="0.0" min="-2.0" max="2.0" step="0.1">
            </div>
            <div>
                <label>Penalidad frecuencia</label>
                <input type="number" id="frequency-penalty" value="0.0" min="0.0" max="2.0" step="0.1">
            </div>
        </div>
        <div style="font-size:0.8rem; color:#666; margin-top:8px; text-align:center;">
            ⚡ Repetition Penalty (>1 reduce repetición, <1 la favorece). Presence Penalty (positivo desalienta tokens ya vistos).
        </div>
    </div>
    <div id="debug" class="content">
        <h2>Debug - Respuestas del servidor</h2>
        <div class="debug-box visible" id="debug-box">
            <div class="debug-header">
                <span class="debug-title">📡 Comunicaciones</span>
                <span class="debug-clear" onclick="clearDebug()">🗑️ Limpiar</span>
            </div>
            <div id="debug-content">Esperando actividad...</div>
        </div>
        <div id="raw-error" class="error-detail" style="display: none;"></div>
        <div class="model-stats">
            <h3>Gestión del modelo</h3>
            <p><strong>Directorio:</strong> <?php echo $modelDir; ?></p>
            <button class="danger" id="delete-model-btn">🔥 Eliminar modelo completo</button>
            <p style="font-size:0.8rem; color:#666; margin-top:8px;">
                ⚠️ Esto eliminará todos los archivos del modelo y reiniciará.
            </p>
            <div id="delete-model-message" class="message"></div>
        </div>
    </div>
</div>
<div class="export-modal" id="export-modal">
    <div class="modal-content">
        <h3 id="modal-title">Exportar conversación</h3>
        <pre id="modal-content"></pre>
        <div class="button-group" style="margin-top: 16px;">
            <button class="secondary" onclick="copyToClipboard()">📋 Copiar</button>
            <button class="success" onclick="downloadExport()">💾 Descargar</button>
            <button onclick="closeModal()">Cerrar</button>
        </div>
    </div>
</div>
<script>
    const tabs = document.querySelectorAll('.tab');
    const contents = {
        train: document.getElementById('train'),
        qa: document.getElementById('qa'),
        chat: document.getElementById('chat'),
        debug: document.getElementById('debug')
    };
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            for (let key in contents) {
                contents[key].classList.remove('active');
            }
            contents[tab.dataset.tab].classList.add('active');
        });
    });
    const trainText = document.getElementById('train-text');
    const charCount = document.getElementById('char-count');
    const wordCount = document.getElementById('word-count');
    trainText.addEventListener('input', () => {
        const text = trainText.value;
        charCount.textContent = text.length + ' caracteres';
        wordCount.textContent = (text.trim() ? text.trim().split(/\s+/).length : 0) + ' palabras';
    });
    const qaQuestion = document.getElementById('qa-question');
    const qaAnswer = document.getElementById('qa-answer');
    const qaCharCount = document.getElementById('qa-char-count');
    const qaWordCount = document.getElementById('qa-word-count');
    function updateQaCounts() {
        const q = qaQuestion.value;
        const a = qaAnswer.value;
        const total = q + ' ' + a;
        qaCharCount.textContent = total.length + ' caracteres totales';
        qaWordCount.textContent = (total.trim() ? total.trim().split(/\s+/).length : 0) + ' palabras totales';
    }
    qaQuestion.addEventListener('input', updateQaCounts);
    qaAnswer.addEventListener('input', updateQaCounts);
    function addDebugEntry(type, data) {
        const debugContent = document.getElementById('debug-content');
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.style.marginBottom = '15px';
        entry.style.borderBottom = '1px solid #333';
        entry.style.paddingBottom = '10px';
        entry.style.whiteSpace = 'pre-wrap';
        entry.style.wordBreak = 'break-word';
        let content = `[${timestamp}] ${type}\n`;
        if (typeof data === 'object') {
            try {
                content += JSON.stringify(data, null, 2);
            } catch (e) {
                content += String(data);
            }
        } else {
            content += data;
        }
        entry.textContent = content;
        debugContent.appendChild(entry);
        debugContent.scrollTop = debugContent.scrollHeight;
    }
    function showRawError(html) {
        const errorDiv = document.getElementById('raw-error');
        errorDiv.style.display = 'block';
        errorDiv.innerHTML = '🔴 ERROR RAW DEL SERVIDOR:\n' + html.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        document.querySelector('[data-tab="debug"]').click();
    }
    function clearDebug() {
        document.getElementById('debug-content').innerHTML = '';
        document.getElementById('raw-error').style.display = 'none';
    }
    let currentExportData = '';
    let currentExportFormat = '';
    function openModal(title, content) {
        document.getElementById('modal-title').textContent = title;
        document.getElementById('modal-content').textContent = content;
        document.getElementById('export-modal').classList.add('active');
        currentExportData = content;
    }
    function closeModal() {
        document.getElementById('export-modal').classList.remove('active');
    }
    function copyToClipboard() {
        navigator.clipboard.writeText(currentExportData).then(() => {
            alert('✅ Copiado al portapapeles');
        }).catch(() => {
            alert('❌ Error al copiar');
        });
    }
    function downloadExport() {
        const extension = currentExportFormat === 'json' ? 'json' : 'txt';
        const blob = new Blob([currentExportData], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversacion.${extension}`;
        a.click();
        URL.revokeObjectURL(url);
    }
    async function exportHistory(format) {
        currentExportFormat = format;
        const formData = new FormData();
        formData.append('action', 'export');
        formData.append('format', format);
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                const title = format === 'json' ? '📊 Exportar JSON' : '📝 Exportar Texto';
                const content = format === 'json' ? JSON.stringify(data.data, null, 2) : data.data;
                openModal(title, content);
                addDebugEntry(`📤 EXPORT ${format.toUpperCase()}`, { count: data.count });
            } else {
                alert('Error: ' + data.message);
            }
        } catch (err) {
            alert('Error de conexión');
        }
    }
    document.getElementById('export-json-btn').addEventListener('click', () => exportHistory('json'));
    document.getElementById('export-text-btn').addEventListener('click', () => exportHistory('text'));
    document.getElementById('clear-history-btn').addEventListener('click', async () => {
        if (!confirm('¿Borrar todo el historial de conversación?')) return;
        const formData = new FormData();
        formData.append('action', 'clear_history');
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                location.reload();
            } else {
                alert('Error: ' + data.message);
            }
        } catch (err) {
            alert('Error de conexión');
        }
    });
    document.getElementById('train-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = trainText.value.trim();
        const msgDiv = document.getElementById('train-message');
        const trainBtn = document.getElementById('train-btn');
        if (!text) {
            msgDiv.className = 'message error';
            msgDiv.textContent = 'El texto no puede estar vacío.';
            return;
        }
        msgDiv.className = 'message';
        msgDiv.textContent = 'Entrenando...';
        trainBtn.disabled = true;
        const formData = new FormData();
        formData.append('action', 'train');
        formData.append('text', text);
        addDebugEntry('📤 ENVIANDO ENTRENAMIENTO', {
            length: text.length,
            preview: text.substring(0, 100) + (text.length > 100 ? '...' : '')
        });
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const responseText = await res.text();
            try {
                const data = JSON.parse(responseText);
                addDebugEntry('📥 RESPUESTA RECIBIDA (JSON)', data);
                if (data.success) {
                    msgDiv.className = 'message success';
                    msgDiv.textContent = data.message;
                    if (data.received_length) {
                        msgDiv.textContent += ` (Recibidos ${data.received_length} caracteres)`;
                    }
                } else {
                    msgDiv.className = 'message error';
                    msgDiv.textContent = data.message;
                }
            } catch (jsonError) {
                addDebugEntry('❌ ERROR: RESPUESTA NO ES JSON', responseText.substring(0, 500));
                showRawError(responseText);
                msgDiv.className = 'message error';
                msgDiv.textContent = 'El servidor devolvió HTML. Revisa Debug.';
            }
        } catch (err) {
            msgDiv.className = 'message error';
            msgDiv.textContent = 'Error de conexión: ' + err.message;
            addDebugEntry('❌ ERROR FETCH', err.message);
        } finally {
            trainBtn.disabled = false;
        }
    });
    document.getElementById('qa-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = qaQuestion.value.trim();
        const answer = qaAnswer.value.trim();
        const msgDiv = document.getElementById('qa-message');
        const qaBtn = document.getElementById('qa-btn');
        if (!question || !answer) {
            msgDiv.className = 'message error';
            msgDiv.textContent = 'La pregunta y la respuesta no pueden estar vacías.';
            return;
        }
        msgDiv.className = 'message';
        msgDiv.textContent = 'Entrenando QA...';
        qaBtn.disabled = true;
        const formData = new FormData();
        formData.append('action', 'train_qa');
        formData.append('question', question);
        formData.append('answer', answer);
        const fullText = question + ' ' + answer;
        addDebugEntry('📤 ENVIANDO ENTRENAMIENTO QA', {
            question: question,
            answer: answer,
            length: fullText.length,
            preview: fullText.substring(0, 100) + (fullText.length > 100 ? '...' : '')
        });
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const responseText = await res.text();
            try {
                const data = JSON.parse(responseText);
                addDebugEntry('📥 RESPUESTA QA', data);
                if (data.success) {
                    msgDiv.className = 'message success';
                    msgDiv.textContent = data.message;
                    if (data.received_length) {
                        msgDiv.textContent += ` (Recibidos ${data.received_length} caracteres)`;
                    }
                } else {
                    msgDiv.className = 'message error';
                    msgDiv.textContent = data.message;
                }
            } catch (jsonError) {
                addDebugEntry('❌ ERROR: RESPUESTA NO ES JSON', responseText.substring(0, 500));
                showRawError(responseText);
                msgDiv.className = 'message error';
                msgDiv.textContent = 'El servidor devolvió HTML. Revisa Debug.';
            }
        } catch (err) {
            msgDiv.className = 'message error';
            msgDiv.textContent = 'Error de conexión: ' + err.message;
            addDebugEntry('❌ ERROR FETCH', err.message);
        } finally {
            qaBtn.disabled = false;
        }
    });
    document.getElementById('delete-model-btn').addEventListener('click', async () => {
        if (!confirm('⚠️ ¿Estás seguro de que quieres eliminar TODO el modelo?\n\nEsto borrará permanentemente todos los archivos de entrenamiento y reiniciará el modelo desde cero.')) return;
        const msgDiv = document.getElementById('delete-model-message');
        const deleteBtn = document.getElementById('delete-model-btn');
        msgDiv.className = 'message';
        msgDiv.textContent = 'Eliminando modelo...';
        deleteBtn.disabled = true;
        const formData = new FormData();
        formData.append('action', 'delete_model');
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                msgDiv.className = 'message success';
                msgDiv.textContent = data.message;
                if (data.deleted && data.deleted.length > 0) {
                    msgDiv.textContent += ' Archivos eliminados: ' + data.deleted.join(', ');
                }
                addDebugEntry('🗑️ MODELO ELIMINADO', data);
                document.getElementById('chat-area').innerHTML = '<div class="chat-message bot-message">Modelo reiniciado. Escribe algo...</div>';
                setTimeout(() => {
                    location.reload();
                }, 2000);
            } else {
                msgDiv.className = 'message error';
                msgDiv.textContent = data.message;
                deleteBtn.disabled = false;
            }
        } catch (err) {
            msgDiv.className = 'message error';
            msgDiv.textContent = 'Error de conexión: ' + err.message;
            deleteBtn.disabled = false;
            addDebugEntry('❌ ERROR ELIMINAR MODELO', err.message);
        }
    });
    const chatArea = document.getElementById('chat-area');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const maxTokensInput = document.getElementById('max-tokens');
    const temperatureInput = document.getElementById('temperature');
    const topKSelect = document.getElementById('top-k');
    const topPInput = document.getElementById('top-p');
    const repetitionPenaltyInput = document.getElementById('repetition-penalty');
    const presencePenaltyInput = document.getElementById('presence-penalty');
    const frequencyPenaltyInput = document.getElementById('frequency-penalty');

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        appendMessage(message, 'user', new Date().toLocaleTimeString());
        chatInput.value = '';
        const loader = document.createElement('div');
        loader.className = 'chat-message bot-message';
        loader.innerHTML = '<span class="loader"></span> Pensando...';
        chatArea.appendChild(loader);
        chatArea.scrollTop = chatArea.scrollHeight;
        const formData = new FormData();
        formData.append('action', 'chat');
        formData.append('message', message);
        formData.append('max_tokens', maxTokensInput.value);
        formData.append('temperature', temperatureInput.value);
        formData.append('top_k', topKSelect.value);
        formData.append('top_p', topPInput.value);
        formData.append('repetition_penalty', repetitionPenaltyInput.value);
        formData.append('presence_penalty', presencePenaltyInput.value);
        formData.append('frequency_penalty', frequencyPenaltyInput.value);
        addDebugEntry('📤 ENVIANDO MENSAJE', {
            message: message,
            max_tokens: maxTokensInput.value,
            temperature: temperatureInput.value,
            top_k: topKSelect.value,
            top_p: topPInput.value,
            repetition_penalty: repetitionPenaltyInput.value,
            presence_penalty: presencePenaltyInput.value,
            frequency_penalty: frequencyPenaltyInput.value
        });
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const responseText = await res.text();
            try {
                const data = JSON.parse(responseText);
                addDebugEntry('📥 RESPUESTA CHAT', data);
                loader.remove();
                if (data.response) {
                    appendMessage(data.response, 'bot', new Date().toLocaleTimeString());
                } else {
                    appendMessage('(respuesta vacía)', 'bot', new Date().toLocaleTimeString());
                }
            } catch (jsonError) {
                addDebugEntry('❌ ERROR: RESPUESTA NO ES JSON', responseText.substring(0, 500));
                showRawError(responseText);
                loader.remove();
                appendMessage('Error: El servidor no respondió con JSON', 'bot', new Date().toLocaleTimeString());
            }
        } catch (err) {
            loader.remove();
            appendMessage('Error: ' + err.message, 'bot', new Date().toLocaleTimeString());
            addDebugEntry('❌ ERROR FETCH', err.message);
        }
    }
    function appendMessage(text, sender, timestamp) {
        const div = document.createElement('div');
        div.className = `chat-message ${sender}-message`;
        div.innerHTML = `${text}<div class="timestamp">${timestamp || new Date().toLocaleTimeString()}</div>`;
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
    }
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
    trainText.dispatchEvent(new Event('input'));
    updateQaCounts();
    chatArea.scrollTop = chatArea.scrollHeight;
</script>
</body>
</html>