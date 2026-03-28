<?php
require_once 'LLM.php';
session_start();
if(!file_exists('all-models')) mkdir('all-models');
$modelDir = 'all-models/'.($_SESSION['model']??'tiny-php');
$maxContext = 512;
$action = $_POST['action'] ?? $_GET['action'] ?? '';
if(!isset($_SESSION['chat_history'])) $_SESSION['chat_history'] = [];

if($_SERVER['REQUEST_METHOD'] == 'POST'){
  $llm = new LLM($modelDir, $maxContext);
  if($action === 'train') {
    header('Content-Type: application/json');
    try {
        $text = $_POST['text'] ?? '';
        if (empty($text)) {
            echo json_encode(['success' => false, 'message' => 'El texto no puede estar vacío.']);
            exit;
        }

        // Entrenar directamente sin dividir en lotes
        $llm->train($text);
        $totalTokens = $llm->getVocabSize();

        echo json_encode([
            'success' => true,
            'message' => "Modelo entrenado correctamente. Vocabulario: $totalTokens tokens.",
            'total_tokens' => $totalTokens,
            'received_length' => strlen($text)
        ]);
    } catch (Exception $e) {
        echo json_encode(['success' => false, 'message' => 'Error: ' . $e->getMessage()]);
    }
    exit;
  }elseif($action === 'train_qa' && $_SERVER['REQUEST_METHOD'] === 'POST'){
    header('Content-Type: application/json');
    try {
        $question = $_POST['question'] ?? '';
        $answer = $_POST['answer'] ?? '';
        if (empty($question) || empty($answer)) {
            echo json_encode(['success' => false, 'message' => 'La pregunta y la respuesta no pueden estar vacías.']);
            exit;
        }
        $text = "<|USER|>$question<|ASSISTANT|>$answer";
        $llm->train($text);
        echo json_encode([
            'success' => true,
            'message' => 'Modelo entrenado con QA. Vocabulario: ' . $llm->getVocabSize() . ' tokens.'
        ]);
    } catch (Exception $e) {
        echo json_encode(['success' => false, 'message' => 'Error: ' . $e->getMessage()]);
    }
    exit;
  }elseif($action === 'chat' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');
    try {
        $message = $_POST['message'] ?? '';
        $maxTokens = (int)($_POST['max_tokens'] ?? 50);
        $temperature = (float)($_POST['temperature'] ?? 0.8);
        $topK = isset($_POST['top_k']) ? (int)$_POST['top_k'] : null;
        $frequencyPenalty = (float)($_POST['frequency_penalty'] ?? 0.0);
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
  }elseif($action === 'export' && $_SERVER['REQUEST_METHOD'] === 'POST'){
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
  }elseif($action === 'clear_history' && $_SERVER['REQUEST_METHOD'] === 'POST'){
    header('Content-Type: application/json');
    $_SESSION['chat_history'] = [];
    echo json_encode(['success' => true, 'message' => 'Historial borrado']);
    exit;
  }elseif($action === 'delete_model' && $_SERVER['REQUEST_METHOD'] === 'POST'){
    header('Content-Type: application/json');
    try {
        $llm->deleteModel();
        $_SESSION['chat_history'] = [];
        echo json_encode([
            'success' => true,
            'message' => 'Modelo eliminado correctamente.'
        ]);
    } catch (Exception $e) {
        echo json_encode(['success' => false, 'message' => 'Error al eliminar: ' . $e->getMessage()]);
    }
    exit;
  }
}
?>
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Studio · RWKV entrenable</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        body {
            background: linear-gradient(145deg, #f9fafc 0%, #f1f4f9 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
        }

        .app-container {
            max-width: 1300px;
            width: 100%;
            background: #ffffffb3;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 2.5rem;
            box-shadow: 0 25px 50px -12px #00000040;
            overflow: hidden;
            border: 1px solid #ffffff80;
        }

        /* Tabs */
        .tabs {
            display: flex;
            background: #ffffff66;
            border-bottom: 1px solid #0000000d;
            padding: 0 2rem;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .tab {
            padding: 1.2rem 1.8rem;
            font-weight: 600;
            color: #4b5563;
            cursor: pointer;
            transition: all 0.2s ease;
            border-bottom: 3px solid transparent;
            letter-spacing: 0.3px;
        }
        .tab:hover {
            color: #4f46e5;
            background: #4f46e50a;
        }
        .tab.active {
            color: #4f46e5;
            border-bottom-color: #4f46e5;
            background: linear-gradient(to top, #4f46e514, transparent);
        }

        /* Contenido */
        .content {
            display: none;
            padding: 2rem;
            background: white;
        }
        .content.active {
            display: block;
        }

        /* Tipografía */
        h2 {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 1.8rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }
        .badge {
            background: #e0e7ff;
            color: #4f46e5;
            font-size: 0.9rem;
            padding: 0.3rem 1rem;
            border-radius: 40px;
            font-weight: 500;
        }

        /* Formularios */
        label {
            display: block;
            font-weight: 500;
            color: #334155;
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
        }
        textarea, input[type="number"], select {
            width: 100%;
            padding: 0.9rem 1.2rem;
            border: 1px solid #e2e8f0;
            border-radius: 1.2rem;
            font-size: 1rem;
            transition: border 0.15s ease, box-shadow 0.15s ease;
            background: #fafdff;
            resize:vertical;
        }
        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px #4f46e533;
        }

        /* Botones */
        button {
            background: #4f46e5;
            color: white;
            border: none;
            padding: 0.9rem 2rem;
            border-radius: 3rem;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease;
            box-shadow: 0 4px 6px -2px #4f46e54d;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        button:hover {
            background: #4338ca;
            box-shadow: 0 8px 12px -4px #4f46e566;
        }
        button:active {
            transform: scale(0.98);
        }
        button.secondary {
            background: white;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            box-shadow: none;
        }
        button.secondary:hover {
            background: #f8fafc;
            border-color: #cbd5e1;
        }
        button.success {
            background: #10b981;
            box-shadow: 0 4px 6px -2px #10b9814d;
        }
        button.success:hover { background: #059669; }
        button.warning {
            background: #f59e0b;
            color: white;
            box-shadow: 0 4px 6px -2px #f59e0b4d;
        }
        button.warning:hover { background: #d97706; }
        button.danger {
            background: #ef4444;
            box-shadow: 0 4px 6px -2px #ef44444d;
        }
        button.danger:hover { background: #dc2626; }

        .button-group {
            display: flex;
            gap: 0.8rem;
            flex-wrap: wrap;
            margin: 1.5rem 0 0.5rem;
        }

        /* Mensajes flotantes */
        .message {
            margin-top: 1.5rem;
            padding: 1rem 1.5rem;
            border-radius: 1.5rem;
            font-weight: 500;
            display: none;
            animation: fadeIn 0.3s ease;
        }
        .message.success {
            background: #d1fae5;
            color: #065f46;
            display: block;
        }
        .message.error {
            background: #fee2e2;
            color: #991b1b;
            display: block;
        }

        /* Estadísticas pequeñas */
        .stats {
            display: flex;
            gap: 1.5rem;
            margin: 1rem 0 1.5rem;
            color: #4b5563;
            font-size: 0.9rem;
            flex-wrap: wrap;
        }
        .stat-item {
            background: #f1f4f9;
            padding: 0.3rem 1.2rem;
            border-radius: 3rem;
            font-weight: 500;
        }

        /* Chat */
        #chat-area {
            height: 350px;
            overflow-y: auto;
            border: 1px solid #edf2f7;
            border-radius: 1.8rem;
            padding: 1.5rem;
            background: #fafdff;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .chat-message {
            max-width: 80%;
            padding: 0.8rem 1.4rem;
            border-radius: 1.8rem;
            word-wrap: break-word;
            line-height: 1.5;
            position: relative;
            animation: slideIn 0.2s ease;
        }
        .user-message {
            align-self: flex-end;
            background: #4f46e5;
            color: white;
            border-bottom-right-radius: 0.4rem;
        }
        .bot-message {
            align-self: flex-start;
            background: white;
            border: 1px solid #e2e8f0;
            border-bottom-left-radius: 0.4rem;
        }
        .timestamp {
            font-size: 0.65rem;
            margin-top: 0.3rem;
            opacity: 0.7;
            text-align: right;
        }
        .chat-input {
            display: flex;
            gap: 0.8rem;
        }
        .chat-input input {
            flex: 1;
            padding: 1rem 1.5rem;
            border-radius: 3rem;
            border: 1px solid #e2e8f0;
        }

        .params-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1.5rem;
            background: #f9fafc;
            padding: 1.5rem;
            border-radius: 2rem;
            margin: 1.5rem 0;
        }
        .param-item {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }
        .param-item label {
            margin-bottom: 0;
            font-size: 0.9rem;
            color: #4b5563;
        }
        .param-item input[type=range] {
            width: 100%;
            margin: 0.2rem 0;
        }
        .param-value {
            font-weight: 600;
            color: #4f46e5;
            background: #e0e7ff;
            padding: 0.2rem 0.8rem;
            border-radius: 3rem;
            display: inline-block;
            width: fit-content;
            font-size: 0.9rem;
        }

        /* Debug */
        .debug-box {
            background: #1e1e2f;
            color: #a5f3fc;
            padding: 1.5rem;
            border-radius: 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #334155;
            margin-top: 1.5rem;
        }
        .debug-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
            color: #94a3b8;
        }
        .debug-clear {
            cursor: pointer;
            color: #f87171;
            font-weight: 500;
        }
        .error-detail {
            background: #2d1a1a;
            color: #fca5a5;
            padding: 1rem;
            border-radius: 1rem;
            font-family: monospace;
            white-space: pre-wrap;
            margin-top: 1rem;
        }

        /* Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: #00000099;
            backdrop-filter: blur(5px);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: white;
            padding: 2rem;
            border-radius: 2rem;
            max-width: 700px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 30px 60px #0000004d;
        }
        .modal-content pre {
            background: #f1f4f9;
            padding: 1.5rem;
            border-radius: 1.2rem;
            overflow-x: auto;
            white-space: pre-wrap;
            font-size: 0.9rem;
        }

        /* Animaciones */
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideIn { from { opacity: 0; transform: translateX(10px); } to { opacity: 1; transform: translateX(0); } }

        /* Loader */
        .loader {
            display: inline-block;
            width: 1.2rem;
            height: 1.2rem;
            border: 2px solid #4f46e54d;
            border-top-color: #4f46e5;
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="app-container">
    <!-- Tabs -->
    <div class="tabs">
        <div class="tab active" data-tab="train">🧠 Entrenar (recomendado)</div>
        <div class="tab" data-tab="qa">❓ QA</div>
        <div class="tab" data-tab="chat">💬 Chatear <span class="badge">Top-K/P</span></div>
        <div class="tab" data-tab="debug">🐞 Debug</div>
    </div>

    <!-- Panel Train -->
    <div id="train" class="content active">
        <h2>Entrenar modelo con texto libre</h2>
        <form id="train-form">
            <label>📄 Texto de entrenamiento:</label>
            <textarea id="train-text" rows="8" placeholder="Pega aquí cualquier texto, artículo, libro…"></textarea>
            <div class="stats">
                <span class="stat-item" id="char-count">0 caracteres</span>
                <span class="stat-item" id="word-count">0 palabras</span>
            </div>
            <button type="submit" id="train-btn">🚀 Iniciar entrenamiento</button>
        </form>
        <div id="train-message" class="message"></div>
    </div>

    <!-- Panel QA -->
    <div id="qa" class="content">
        <h2>Entrenar con pregunta/respuesta</h2>
        <form id="qa-form">
            <label>❓ Pregunta:</label>
            <textarea id="qa-question" rows="3" placeholder="Escribe la pregunta…"></textarea>
            <label>💡 Respuesta:</label>
            <textarea id="qa-answer" rows="3" placeholder="Escribe la respuesta…"></textarea>
            <div class="stats">
                <span class="stat-item" id="qa-char-count">0 caracteres</span>
                <span class="stat-item" id="qa-word-count">0 palabras</span>
            </div>
            <button type="submit" id="qa-btn">🎓 Entrenar QA</button>
        </form>
        <div id="qa-message" class="message"></div>
    </div>

    <!-- Panel Chat -->
    <div id="chat" class="content">
        <h2>Conversar con el modelo</h2>
        <div class="button-group history-controls">
            <button class="secondary" id="export-json-btn">📥 JSON</button>
            <button class="secondary" id="export-text-btn">📥 Texto</button>
            <button class="warning" id="clear-history-btn">🗑️ Borrar historial</button>
        </div>
        <div id="chat-area">
            <?php if (!empty($_SESSION['chat_history'])){ ?>
                <?php foreach ($_SESSION['chat_history'] as $msg){ ?>
                    <div class="chat-message user-message"><?php echo htmlspecialchars($msg['user']); ?></div>
                    <div class="chat-message bot-message">
                        <?php echo htmlspecialchars($msg['bot']); ?>
                        <div class="timestamp"><?php echo $msg['timestamp']; ?></div>
                    </div>
                <?php } ?>
            <?php } ?>
        </div>
        <div class="chat-input">
            <input type="text" id="chat-input" placeholder="Mensaje…" autocomplete="off">
            <button id="send-btn">Enviar</button>
        </div>

        <!-- Parámetros de generación -->
        <div class="params-grid">
            <div class="param-item">
                <label>Max tokens <span class="param-value" id="max-tokens-val">30</span></label>
                <input type="range" id="max-tokens" min="1" max="200" value="30">
            </div>
            <div class="param-item">
                <label>Temperatura <span class="param-value" id="temperature-val">0.7</span></label>
                <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="0.7">
            </div>
            <div class="param-item">
                <label>Top‑K <span class="param-value" id="top-k-val">50</span></label>
                <select id="top-k">
                    <option value="1">1</option><option value="3">3</option><option value="5">5</option>
                    <option value="10">10</option><option value="20">20</option><option value="50" selected>50</option>
                    <option value="0">0 (todas)</option>
                </select>
            </div>
            <div class="param-item">
                <label>Top‑P <span class="param-value" id="top-p-val">1.0</span></label>
                <input type="range" id="top-p" min="0.0" max="1.0" step="0.05" value="1.0">
            </div>
            <div class="param-item">
                <label>Repetition penalty <span class="param-value" id="repetition-penalty-val">1.0</span></label>
                <input type="range" id="repetition-penalty" min="0.1" max="2.0" step="0.05" value="1.0">
            </div>
            <div class="param-item">
                <label>Presence penalty <span class="param-value" id="presence-penalty-val">0.0</span></label>
                <input type="range" id="presence-penalty" min="-2.0" max="2.0" step="0.1" value="0.0">
            </div>
            <div class="param-item">
                <label>Frequency penalty <span class="param-value" id="frequency-penalty-val">0.0</span></label>
                <input type="range" id="frequency-penalty" min="0.0" max="2.0" step="0.1" value="0.0">
            </div>
        </div>
        <div style="font-size:0.85rem; color:#64748b; text-align:center;">💡 Repetition >1 reduce repetición; Presence positivo desalienta palabras ya vistas.</div>
    </div>

    <!-- Panel Debug -->
    <div id="debug" class="content">
        <h2>🔍 Depuración</h2>
        <div class="debug-box" id="debug-box">
            <div class="debug-header">
                <span>📡 Comunicaciones</span>
                <span class="debug-clear" onclick="clearDebug()">Limpiar</span>
            </div>
            <div id="debug-content">Esperando actividad…</div>
        </div>
        <div id="raw-error" class="error-detail" style="display: none;"></div>
        <div style="margin-top: 2rem;">
            <h3>Gestión del modelo</h3>
            <p><strong>Directorio:</strong> <?php echo $modelDir; ?></p>
            <button class="danger" id="delete-model-btn">🔥 Eliminar modelo completo</button>
            <div id="delete-model-message" class="message" style="margin-top:1rem;"></div>
        </div>
    </div>
</div>

<!-- Modal de exportación -->
<div class="modal" id="export-modal">
    <div class="modal-content">
        <h3 id="modal-title">Exportar conversación</h3>
        <pre id="modal-content"></pre>
        <div class="button-group" style="margin-top: 1.5rem;">
            <button class="secondary" onclick="copyToClipboard()">📋 Copiar</button>
            <button class="success" onclick="downloadExport()">💾 Descargar</button>
            <button onclick="closeModal()">Cerrar</button>
        </div>
    </div>
</div>
<script>
    // Elementos DOM
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
            for (let key in contents) contents[key].classList.remove('active');
            contents[tab.dataset.tab].classList.add('active');
        });
    });

    // Actualizar contadores
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
        qaCharCount.textContent = total.length + ' caracteres';
        qaWordCount.textContent = (total.trim() ? total.trim().split(/\s+/).length : 0) + ' palabras';
    }
    qaQuestion.addEventListener('input', updateQaCounts);
    qaAnswer.addEventListener('input', updateQaCounts);
    updateQaCounts();

    // Parámetros con sliders (actualizar valores mostrados)
    const paramIds = ['max-tokens', 'temperature', 'top-p', 'repetition-penalty', 'presence-penalty', 'frequency-penalty'];
    paramIds.forEach(id => {
        const input = document.getElementById(id);
        const span = document.getElementById(id + '-val');
        if (input && span) {
            const update = () => span.textContent = input.value;
            input.addEventListener('input', update);
            update();
        }
    });
    document.getElementById('top-k').addEventListener('change', e => {
        document.getElementById('top-k-val').textContent = e.target.value;
    });

    // Debug
    function addDebugEntry(type, data) {
        const debugContent = document.getElementById('debug-content');
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.style.marginBottom = '12px';
        entry.style.borderBottom = '1px solid #334155';
        entry.style.paddingBottom = '8px';
        entry.style.whiteSpace = 'pre-wrap';
        entry.style.wordBreak = 'break-word';
        let content = `[${timestamp}] ${type}\n`;
        content += typeof data === 'object' ? JSON.stringify(data, null, 2) : data;
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

    window.clearDebug = function() {
        document.getElementById('debug-content').innerHTML = '';
        document.getElementById('raw-error').style.display = 'none';
    };

    // Modal exportación
    let currentExportData = '', currentExportFormat = '';
    function openModal(title, content) {
        document.getElementById('modal-title').textContent = title;
        document.getElementById('modal-content').textContent = content;
        document.getElementById('export-modal').classList.add('active');
        currentExportData = content;
    }
    function closeModal() {
        document.getElementById('export-modal').classList.remove('active');
    }
    window.copyToClipboard = function() {
        navigator.clipboard.writeText(currentExportData).then(() => alert('✅ Copiado')).catch(() => alert('❌ Error'));
    };
    window.downloadExport = function() {
        const ext = currentExportFormat === 'json' ? 'json' : 'txt';
        const blob = new Blob([currentExportData], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversacion.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
    };

    // Exportar historial
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
            } else alert('Error: ' + data.message);
        } catch (err) { alert('Error de conexión'); }
    }
    document.getElementById('export-json-btn').addEventListener('click', () => exportHistory('json'));
    document.getElementById('export-text-btn').addEventListener('click', () => exportHistory('text'));

    // Borrar historial
    document.getElementById('clear-history-btn').addEventListener('click', async () => {
        if (!confirm('¿Borrar todo el historial de conversación?')) return;
        const formData = new FormData(); formData.append('action', 'clear_history');
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) document.getElementById('chat-area').innerHTML = ''; else alert('Error: ' + data.message);
        } catch (err) { alert('Error de conexión'); }
    });

    // Entrenamiento (sin batches)
    document.getElementById('train-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = trainText.value.trim();
        const msgDiv = document.getElementById('train-message');
        const btn = document.getElementById('train-btn');
        if (!text) { msgDiv.className = 'message error'; msgDiv.textContent = 'El texto no puede estar vacío.'; return; }
        msgDiv.className = 'message'; msgDiv.textContent = 'Entrenando…';
        btn.disabled = true;
        const formData = new FormData(); formData.append('action', 'train'); formData.append('text', text);
        addDebugEntry('📤 ENVIANDO ENTRENAMIENTO', { length: text.length, preview: text.substring(0,100)+'…' });
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const responseText = await res.text();
            try {
                const data = JSON.parse(responseText);
                addDebugEntry('📥 RESPUESTA ENTRENAMIENTO', data);
                if (data.success) {
                    msgDiv.className = 'message success';
                    msgDiv.textContent = data.message;
                } else {
                    msgDiv.className = 'message error';
                    msgDiv.textContent = data.message;
                }
            } catch (jsonError) {
                addDebugEntry('❌ ERROR: RESPUESTA NO ES JSON', responseText.substring(0,500));
                showRawError(responseText);
                msgDiv.className = 'message error';
                msgDiv.textContent = 'El servidor devolvió HTML. Revisa Debug.';
            }
        } catch (err) {
            msgDiv.className = 'message error';
            msgDiv.textContent = 'Error de conexión: ' + err.message;
            addDebugEntry('❌ ERROR FETCH', err.message);
        } finally { btn.disabled = false; }
    });

    // Entrenamiento QA
    document.getElementById('qa-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = qaQuestion.value.trim();
        const answer = qaAnswer.value.trim();
        const msgDiv = document.getElementById('qa-message');
        const btn = document.getElementById('qa-btn');
        if (!question || !answer) { msgDiv.className = 'message error'; msgDiv.textContent = 'Completa ambos campos.'; return; }
        msgDiv.className = 'message'; msgDiv.textContent = 'Entrenando QA…';
        btn.disabled = true;
        const formData = new FormData(); formData.append('action', 'train_qa'); formData.append('question', question); formData.append('answer', answer);
        addDebugEntry('📤 ENVIANDO QA', { question, answer });
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const responseText = await res.text();
            try {
                const data = JSON.parse(responseText);
                addDebugEntry('📥 RESPUESTA QA', data);
                if (data.success) {
                    msgDiv.className = 'message success';
                    msgDiv.textContent = data.message;
                } else {
                    msgDiv.className = 'message error';
                    msgDiv.textContent = data.message;
                }
            } catch (jsonError) {
                addDebugEntry('❌ ERROR: RESPUESTA NO ES JSON', responseText.substring(0,500));
                showRawError(responseText);
                msgDiv.className = 'message error';
                msgDiv.textContent = 'El servidor devolvió HTML. Revisa Debug.';
            }
        } catch (err) {
            msgDiv.className = 'message error';
            msgDiv.textContent = 'Error de conexión: ' + err.message;
            addDebugEntry('❌ ERROR FETCH', err.message);
        } finally { btn.disabled = false; }
    });

    // Eliminar modelo
    document.getElementById('delete-model-btn').addEventListener('click', async () => {
        if (!confirm('⚠️ ¿Eliminar TODO el modelo permanentemente?')) return;
        const msgDiv = document.getElementById('delete-model-message');
        const btn = document.getElementById('delete-model-btn');
        msgDiv.className = 'message'; msgDiv.textContent = 'Eliminando…';
        btn.disabled = true;
        const formData = new FormData(); formData.append('action', 'delete_model');
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                msgDiv.className = 'message success';
                msgDiv.textContent = data.message;
                addDebugEntry('🗑️ MODELO ELIMINADO', data);
                document.getElementById('chat-area').innerHTML = '<div class="chat-message bot-message">Modelo reiniciado.</div>';
            } else {
                msgDiv.className = 'message error';
                msgDiv.textContent = data.message;
            }
        } catch (err) {
            msgDiv.className = 'message error';
            msgDiv.textContent = 'Error de conexión: ' + err.message;
            addDebugEntry('❌ ERROR ELIMINAR', err.message);
        } finally { btn.disabled = false; }
    });

    // Chat
    const chatArea = document.getElementById('chat-area');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    function appendMessage(text, sender, timestamp) {
        const div = document.createElement('div');
        div.className = `chat-message ${sender}-message`;
        div.innerHTML = `${text}<div class="timestamp">${timestamp || new Date().toLocaleTimeString()}</div>`;
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
    }
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        appendMessage(message, 'user', new Date().toLocaleTimeString());
        chatInput.value = '';
        const loader = document.createElement('div');
        loader.className = 'chat-message bot-message';
        loader.innerHTML = '<span class="loader"></span> Pensando…';
        chatArea.appendChild(loader);
        chatArea.scrollTop = chatArea.scrollHeight;
        const formData = new FormData();
        formData.append('action', 'chat');
        formData.append('message', message);
        formData.append('max_tokens', document.getElementById('max-tokens').value);
        formData.append('temperature', document.getElementById('temperature').value);
        formData.append('top_k', document.getElementById('top-k').value);
        formData.append('top_p', document.getElementById('top-p').value);
        formData.append('repetition_penalty', document.getElementById('repetition-penalty').value);
        formData.append('presence_penalty', document.getElementById('presence-penalty').value);
        formData.append('frequency_penalty', document.getElementById('frequency-penalty').value);
        addDebugEntry('📤 ENVIANDO MENSAJE', { message, ...Object.fromEntries(formData) });
        try {
            const res = await fetch('', { method: 'POST', body: formData });
            const responseText = await res.text();
            try {
                const data = JSON.parse(responseText);
                addDebugEntry('📥 RESPUESTA CHAT', data);
                loader.remove();
                if (data.response) appendMessage(data.response, 'bot', new Date().toLocaleTimeString());
                else appendMessage('(respuesta vacía)', 'bot', new Date().toLocaleTimeString());
            } catch (jsonError) {
                addDebugEntry('❌ ERROR: RESPUESTA NO ES JSON', responseText.substring(0,500));
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
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

    // Inicializar
    trainText.dispatchEvent(new Event('input'));
    chatArea.scrollTop = chatArea.scrollHeight;
</script>