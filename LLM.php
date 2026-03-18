<?php
class Tokenizer {
    private array $vocab = [];
    private array $reverseVocab = [];
    private int $nextId = 0;
    private array $cache = [];

    public function __construct(array $vocab = []) {
        if (!empty($vocab)) {
            $this->vocab = $vocab;
            $this->reverseVocab = array_flip($vocab);
            $this->nextId = count($vocab);
        }
    }

    public function tokenize(string $text, bool $useCache = true): array {
        $text = trim($text);
        if ($text === '') return [];
        if ($useCache && isset($this->cache[$text])) return $this->cache[$text];
        preg_match_all('/(<\|[^>]+\|>|<[^>]+>|\p{L}+|\p{N}+|\p{So}+|\p{P}+|\n)/u', $text, $matches);
        $tokens = $matches[0];
        if ($useCache) $this->cache[$text] = $tokens;
        return $tokens;
    }

    public function encode(array $tokens): array {
        $ids = [];
        foreach ($tokens as $token) {
            if (!isset($this->reverseVocab[$token])) {
                $this->reverseVocab[$token] = $this->nextId;
                $this->vocab[$this->nextId] = $token;
                $ids[] = $this->nextId;
                $this->nextId++;
            } else {
                $ids[] = $this->reverseVocab[$token];
            }
        }
        return $ids;
    }

    public function decode(array $ids): array {
        $tokens = [];
        foreach ($ids as $id) $tokens[] = $this->vocab[$id] ?? '<UNK>';
        return $tokens;
    }

    public function save(string $path): void {
        file_put_contents($path, json_encode([
            'vocab' => $this->vocab,
            'reverseVocab' => $this->reverseVocab,
            'nextId' => $this->nextId
        ]));
    }

    public function load(string $path): void {
        $data = json_decode(file_get_contents($path), true);
        $this->vocab = $data['vocab'];
        $this->reverseVocab = $data['reverseVocab'];
        $this->nextId = $data['nextId'];
    }

    public function getVocabSize(): int {
        return count($this->vocab);
    }
}

class AdamOptimizer {
    private array $m = [];
    private array $v = [];
    private int $t = 0;
    private float $beta1 = 0.9;
    private float $beta2 = 0.999;
    private float $eps = 1e-8;

    public function update(array &$param, array $grad, float $lr = 0.001): void {
        $this->t++;
        if (is_array($param) && isset($param[0]) && is_array($param[0])) {
            // matriz 2D
            foreach ($param as $i => &$row) {
                foreach ($row as $j => &$p) {
                    $g = $grad[$i][$j] ?? 0;
                    $idx = "$i,$j";
                    if (!isset($this->m[$idx])) {
                        $this->m[$idx] = 0.0;
                        $this->v[$idx] = 0.0;
                    }
                    $this->m[$idx] = $this->beta1 * $this->m[$idx] + (1 - $this->beta1) * $g;
                    $this->v[$idx] = $this->beta2 * $this->v[$idx] + (1 - $this->beta2) * $g * $g;

                    $m_hat = $this->m[$idx] / (1 - pow($this->beta1, $this->t));
                    $v_hat = $this->v[$idx] / (1 - pow($this->beta2, $this->t));

                    $p -= $lr * $m_hat / (sqrt($v_hat) + $this->eps);
                }
            }
        } elseif (is_array($param)) {
            // vector 1D
            foreach ($param as $i => &$p) {
                $g = $grad[$i] ?? 0;
                $idx = (string)$i;
                if (!isset($this->m[$idx])) {
                    $this->m[$idx] = 0.0;
                    $this->v[$idx] = 0.0;
                }
                $this->m[$idx] = $this->beta1 * $this->m[$idx] + (1 - $this->beta1) * $g;
                $this->v[$idx] = $this->beta2 * $this->v[$idx] + (1 - $this->beta2) * $g * $g;

                $m_hat = $this->m[$idx] / (1 - pow($this->beta1, $this->t));
                $v_hat = $this->v[$idx] / (1 - pow($this->beta2, $this->t));

                $p -= $lr * $m_hat / (sqrt($v_hat) + $this->eps);
            }
        }
    }
}

class RWKVBlock {
    public int $dim;
    public array $wk, $wv, $wr; // [dim][dim]
    public array $ww; // [dim]
    public array $w1, $w2; // w1: [dim*4][dim], w2: [dim][dim*4]

    private array $grad_wk, $grad_wv, $grad_wr, $grad_ww, $grad_w1, $grad_w2;
    private array $cache; // almacena variables del último forward

    public function __construct(int $dim) {
        $this->dim = $dim;
        $this->wk = $this->randomMatrix($dim, $dim);
        $this->wv = $this->randomMatrix($dim, $dim);
        $this->wr = $this->randomMatrix($dim, $dim);
        $this->ww = $this->randomVector($dim, 0.0, 1.0);
        $this->w1 = $this->randomMatrix($dim * 4, $dim);
        $this->w2 = $this->randomMatrix($dim, $dim * 4);
        $this->resetGradients();
    }

    private function randomMatrix(int $rows, int $cols): array {
        $m = [];
        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) $m[$i][$j] = (mt_rand(-100, 100) / 1000);
        }
        return $m;
    }

    private function randomVector(int $dim, float $min = -0.1, float $max = 0.1): array {
        $v = [];
        for ($i = 0; $i < $dim; $i++) $v[] = $min + mt_rand() / mt_getrandmax() * ($max - $min);
        return $v;
    }

    private function resetGradients(): void {
        $dim = $this->dim;
        $this->grad_wk = array_fill(0, $dim, array_fill(0, $dim, 0.0));
        $this->grad_wv = array_fill(0, $dim, array_fill(0, $dim, 0.0));
        $this->grad_wr = array_fill(0, $dim, array_fill(0, $dim, 0.0));
        $this->grad_ww = array_fill(0, $dim, 0.0);
        $this->grad_w1 = array_fill(0, $dim * 4, array_fill(0, $dim, 0.0));
        $this->grad_w2 = array_fill(0, $dim, array_fill(0, $dim * 4, 0.0));
    }

    private function matMul(array $mat, array $vec): array {
        $rows = count($mat);
        $cols = count($vec);
        $res = array_fill(0, $rows, 0.0);
        for ($i = 0; $i < $rows; $i++) {
            $sum = 0.0;
            for ($j = 0; $j < $cols; $j++) $sum += $mat[$i][$j] * $vec[$j];
            $res[$i] = $sum;
        }
        return $res;
    }

    private function matMulTranspose(array $mat, array $vec): array {
        $rows = count($mat);
        $cols = count($mat[0]);
        $res = array_fill(0, $cols, 0.0);
        for ($j = 0; $j < $cols; $j++) {
            $sum = 0.0;
            for ($i = 0; $i < $rows; $i++) $sum += $mat[$i][$j] * $vec[$i];
            $res[$j] = $sum;
        }
        return $res;
    }

    private function sigmoid(float $x): float {
        return 1.0 / (1.0 + exp(-$x));
    }

    public function forward(array $x, array &$state): array {
        $dim = $this->dim;
        // time mixing
        $k = $this->matMul($this->wk, $x);
        $v = $this->matMul($this->wv, $x);
        $r = $this->matMul($this->wr, $x);
        $w = $this->ww;

        $num_prev = $state['num'] ?? array_fill(0, $dim, 0.0);
        $den_prev = $state['den'] ?? array_fill(0, $dim, 0.0);

        $num = [];
        $den = [];
        $wkv_out = [];
        for ($i = 0; $i < $dim; $i++) {
            $expk = exp($k[$i]);
            $decay = exp(-$w[$i]);
            $num_i = $num_prev[$i] * $decay + $v[$i] * $expk;
            $den_i = $den_prev[$i] * $decay + $expk;
            $num[] = $num_i;
            $den[] = $den_i;
            $wkv_out[] = $num_i / ($den_i + 1e-8);
        }

        $state = ['num' => $num, 'den' => $den];

        $r_sig = array_map([$this, 'sigmoid'], $r);
        $time_out = [];
        for ($i = 0; $i < $dim; $i++) $time_out[$i] = $r_sig[$i] * $wkv_out[$i];

        // channel mixing
        $hidden = $this->matMul($this->w1, $time_out);
        for ($i = 0; $i < count($hidden); $i++) if ($hidden[$i] < 0) $hidden[$i] = 0.0;
        $gated = array_map(fn($h) => $h * $h, $hidden);
        $out = $this->matMul($this->w2, $gated);

        // Guardar todo en caché para el backward
        $this->cache = [
            'x' => $x,
            'k' => $k,
            'v' => $v,
            'r' => $r,
            'w' => $w,
            'num_prev' => $num_prev,
            'den_prev' => $den_prev,
            'num' => $num,
            'den' => $den,
            'wkv_out' => $wkv_out,
            'r_sig' => $r_sig,
            'time_out' => $time_out,
            'hidden' => $hidden,
            'gated' => $gated,
            'out' => $out,
        ];

        return $out;
    }

    public function backward(array $dy): array {
        $c = $this->cache;
        $dim = $this->dim;

        // --- Gradientes canal mixing ---
        // out = w2 * gated
        $dgated = $this->matMulTranspose($this->w2, $dy);
        $dhidden = [];
        $hidden = $c['hidden'];
        for ($i = 0; $i < count($hidden); $i++) {
            $dhidden[$i] = 2 * $hidden[$i] * $dgated[$i];
            if ($hidden[$i] <= 0) $dhidden[$i] = 0.0;
        }
        $dtime_out = $this->matMulTranspose($this->w1, $dhidden);

        // Acumular gradientes de w2
        for ($i = 0; $i < $dim; $i++) {
            for ($j = 0; $j < count($c['gated']); $j++) $this->grad_w2[$i][$j] += $dy[$i] * $c['gated'][$j];
        }
        // Acumular gradientes de w1
        for ($i = 0; $i < count($c['hidden']); $i++) {
            for ($j = 0; $j < $dim; $j++) $this->grad_w1[$i][$j] += $dhidden[$i] * $c['time_out'][$j];
        }

        // --- Gradientes time mixing ---
        $dr_sig = [];
        $dwkv_out = [];
        for ($i = 0; $i < $dim; $i++) {
            $dr_sig[$i] = $dtime_out[$i] * $c['wkv_out'][$i];
            $dwkv_out[$i] = $dtime_out[$i] * $c['r_sig'][$i];
        }
        $dr = [];
        $r_sig = $c['r_sig'];
        for ($i = 0; $i < $dim; $i++) $dr[$i] = $dr_sig[$i] * $r_sig[$i] * (1 - $r_sig[$i]);

        // Gradientes de WKV
        $dk = array_fill(0, $dim, 0.0);
        $dv = array_fill(0, $dim, 0.0);
        $dw = array_fill(0, $dim, 0.0);
        $dnum_prev = array_fill(0, $dim, 0.0);
        $dden_prev = array_fill(0, $dim, 0.0);

        $num_prev = $c['num_prev'];
        $den_prev = $c['den_prev'];
        $num = $c['num'];
        $den = $c['den'];
        $k = $c['k'];
        $v = $c['v'];
        $w = $c['w'];

        for ($i = 0; $i < $dim; $i++) {
            $expk = exp($k[$i]);
            $decay = exp(-$w[$i]);
            $denom = $den[$i] + 1e-8;
            $dout_i = $dwkv_out[$i];

            $dv[$i] = $dout_i * ($expk / $denom);
            $dk[$i] = $dout_i * ( ($v[$i] * $expk * $denom - $num[$i] * $expk) / ($denom * $denom) );
            $dwkv_ddecay = ($num_prev[$i] * $den[$i] - $num[$i] * $den_prev[$i]) / ($denom * $denom);
            $dw[$i] = $dout_i * $dwkv_ddecay * (-$decay);
            $dnum_prev[$i] = $dout_i * ($decay / $denom);
            $dden_prev[$i] = $dout_i * ( - $num[$i] * $decay / ($denom * $denom) );
        }

        // Acumular gradientes de wk, wv, wr, ww
        for ($i = 0; $i < $dim; $i++) {
            for ($j = 0; $j < $dim; $j++) {
                $this->grad_wk[$i][$j] += $dk[$i] * $c['x'][$j];
                $this->grad_wv[$i][$j] += $dv[$i] * $c['x'][$j];
                $this->grad_wr[$i][$j] += $dr[$i] * $c['x'][$j];
            }
            $this->grad_ww[$i] += $dw[$i];
        }

        // Gradiente de entrada dx
        $dx = array_fill(0, $dim, 0.0);
        $temp = $this->matMulTranspose($this->wk, $dk);
        for ($i = 0; $i < $dim; $i++) $dx[$i] += $temp[$i];
        $temp = $this->matMulTranspose($this->wv, $dv);
        for ($i = 0; $i < $dim; $i++) $dx[$i] += $temp[$i];
        $temp = $this->matMulTranspose($this->wr, $dr);
        for ($i = 0; $i < $dim; $i++) $dx[$i] += $temp[$i];

        return ['dx' => $dx, 'dnum_prev' => $dnum_prev, 'dden_prev' => $dden_prev];
    }

    public function applyGradients(AdamOptimizer $opt, float $lr): void {
        $opt->update($this->wk, $this->grad_wk, $lr);
        $opt->update($this->wv, $this->grad_wv, $lr);
        $opt->update($this->wr, $this->grad_wr, $lr);
        $opt->update($this->ww, $this->grad_ww, $lr);
        $opt->update($this->w1, $this->grad_w1, $lr);
        $opt->update($this->w2, $this->grad_w2, $lr);
        $this->resetGradients();
    }

    public function resetState(): void {
        $this->cache = [];
    }
}

class LLM {
    private Tokenizer $tokenizer;
    private string $modelDir;
    private int $maxContext;
    private array $specialTokens = ['<|SYSTEM|>', '<|USER|>', '<|ASSISTANT|>', '<|EOS|>'];
    private array $noSpaceBefore = ['.', ',', '!', '?', ';', ':', ')', ']', '}', '”', '’', '»'];
    private array $noSpaceAfter = ['(', '[', '{', '“', '‘', '«', '¡', '¿'];

    // Embeddings
    private array $embeddings = [];
    private int $embedDim = 128;
    private float $learningRate = 0.001;

    // RWKV
    private array $rwkvBlocks = [];
    private int $numLayers = 4;
    private array $optimizers; // optimizadores por bloque y embeddings

    // Para BPTT: almacenar estados y salidas durante forward
    private array $forwardStates; // estados después de cada paso por capa
    private array $forwardOutputs; // salida de la última capa en cada paso
    private array $forwardInputs; // embedding de entrada en cada paso

    public function __construct(string $modelDir, int $maxContext = 512) {
        $this->modelDir = $modelDir;
        $this->maxContext = $maxContext;
        if (!is_dir($modelDir)) mkdir($modelDir, 0777, true);

        $tokenizerPath = $modelDir . '/tokenizer.json';
        $embedPath = $modelDir . '/embeddings.bin';
        $rwkvPath = $modelDir . '/rwkv.bin';

        if (file_exists($tokenizerPath)) {
            $this->tokenizer = new Tokenizer();
            $this->tokenizer->load($tokenizerPath);
        } else {
            $this->tokenizer = new Tokenizer();
            $this->tokenizer->encode($this->specialTokens);
        }

        if (file_exists($embedPath)) {
            $this->loadEmbeddings($embedPath);
        } else {
            $this->initEmbeddings();
        }

        if (file_exists($rwkvPath)) {
            $this->loadRWKV($rwkvPath);
        } else {
            for ($i = 0; $i < $this->numLayers; $i++) $this->rwkvBlocks[] = new RWKVBlock($this->embedDim);
        }

        // Inicializar optimizadores (uno por bloque y uno para embeddings)
        $this->optimizers['emb'] = new AdamOptimizer();
        for ($i = 0; $i < $this->numLayers; $i++) $this->optimizers["block_$i"] = new AdamOptimizer();
    }

    private function initEmbeddings(): void {
        $vocabSize = $this->tokenizer->getVocabSize();
        for ($i = 0; $i < $vocabSize; $i++) {
            $vec = $this->randomVector();
            $this->embeddings[$i] = $this->normalizeVector($vec);
        }
    }

    private function randomVector(): array {
        $vec = [];
        for ($i = 0; $i < $this->embedDim; $i++) $vec[] = (mt_rand(-100, 100) / 1000);
        return $vec;
    }

    private function normalizeVector(array $vec): array {
        $norm = 0.0;
        foreach ($vec as $v) $norm += $v * $v;
        $norm = sqrt($norm);
        if ($norm < 1e-10) return $vec;
        $normalized = [];
        foreach ($vec as $v) $normalized[] = $v / $norm;
        return $normalized;
    }

    private function saveEmbeddings(string $path): void {
        $fp = fopen($path, 'wb');
        if (!$fp) return;
        fwrite($fp, pack('V', $this->embedDim));
        fwrite($fp, pack('V', count($this->embeddings)));
        foreach ($this->embeddings as $id => $vec) {
            fwrite($fp, pack('V', $id));
            foreach ($vec as $val) fwrite($fp, pack('f', $val));
        }
        fclose($fp);
    }

    private function loadEmbeddings(string $path): void {
        $fp = fopen($path, 'rb');
        if (!$fp) return;
        $dim = unpack('V', fread($fp, 4))[1];
        $this->embedDim = $dim;
        $count = unpack('V', fread($fp, 4))[1];
        for ($i = 0; $i < $count; $i++) {
            $id = unpack('V', fread($fp, 4))[1];
            $vec = [];
            for ($j = 0; $j < $dim; $j++) $vec[] = unpack('f', fread($fp, 4))[1];
            $this->embeddings[$id] = $vec;
        }
        fclose($fp);
    }

    private function saveRWKV(string $path): void {
        $fp = fopen($path, 'wb');
        if (!$fp) return;

        fwrite($fp, pack('V', count($this->rwkvBlocks)));
        fwrite($fp, pack('V', $this->embedDim));

        foreach ($this->rwkvBlocks as $block) {
            $dim = $block->dim;
            // wk
            for ($i = 0; $i < $dim; $i++) {
                for ($j = 0; $j < $dim; $j++) fwrite($fp, pack('f', $block->wk[$i][$j]));
            }
            // wv
            for ($i = 0; $i < $dim; $i++) {
                for ($j = 0; $j < $dim; $j++) fwrite($fp, pack('f', $block->wv[$i][$j]));
            }
            // wr
            for ($i = 0; $i < $dim; $i++) {
                for ($j = 0; $j < $dim; $j++) fwrite($fp, pack('f', $block->wr[$i][$j]));
            }
            // ww
            for ($i = 0; $i < $dim; $i++) fwrite($fp, pack('f', $block->ww[$i]));
            // w1
            $rowsW1 = $dim * 4;
            for ($i = 0; $i < $rowsW1; $i++) {
                for ($j = 0; $j < $dim; $j++) fwrite($fp, pack('f', $block->w1[$i][$j]));
            }
            // w2
            $colsW2 = $dim * 4;
            for ($i = 0; $i < $dim; $i++) {
                for ($j = 0; $j < $colsW2; $j++) fwrite($fp, pack('f', $block->w2[$i][$j]));
            }
        }
        fclose($fp);
    }

    private function loadRWKV(string $path): void {
        $fp = fopen($path, 'rb');
        if (!$fp) return;

        $numLayers = unpack('V', fread($fp, 4))[1];
        $dim = unpack('V', fread($fp, 4))[1];

        for ($layer = 0; $layer < $numLayers; $layer++) {
            $block = new RWKVBlock($dim);
            // wk
            for ($i = 0; $i < $dim; $i++) {
                for ($j = 0; $j < $dim; $j++) $block->wk[$i][$j] = unpack('f', fread($fp, 4))[1];
            }
            // wv
            for ($i = 0; $i < $dim; $i++) {
                for ($j = 0; $j < $dim; $j++) $block->wv[$i][$j] = unpack('f', fread($fp, 4))[1];
            }
            // wr
            for ($i = 0; $i < $dim; $i++) {
                for ($j = 0; $j < $dim; $j++) $block->wr[$i][$j] = unpack('f', fread($fp, 4))[1];
            }
            // ww
            for ($i = 0; $i < $dim; $i++) $block->ww[$i] = unpack('f', fread($fp, 4))[1];
            // w1
            $rowsW1 = $dim * 4;
            for ($i = 0; $i < $rowsW1; $i++) {
                for ($j = 0; $j < $dim; $j++) $block->w1[$i][$j] = unpack('f', fread($fp, 4))[1];
            }
            // w2
            $colsW2 = $dim * 4;
            for ($i = 0; $i < $dim; $i++) {
                for ($j = 0; $j < $colsW2; $j++) $block->w2[$i][$j] = unpack('f', fread($fp, 4))[1];
            }
            $this->rwkvBlocks[] = $block;
        }
        fclose($fp);
    }

    public function train(string $text): void {
        $text = preg_replace('/\s*(<\|[^|]+\|>)\s*/', '$1', $text);
        if (!str_ends_with($text, '<|EOS|>')) $text .= '<|EOS|>';

        $this->ensureEmbeddingsSize();

        $tokens = $this->tokenizer->tokenize($text, false);
        $ids = $this->tokenizer->encode($tokens);
        $len = count($ids);
        if ($len < 2) return;

        // Inicializar arrays para guardar información del forward
        $states = []; // estados por capa en cada paso
        $outputs = []; // salida de la última capa en cada paso
        $inputs = []; // embedding de entrada en cada paso
        $logits_list = []; // logits para cada paso para pérdida

        // Inicializar estados de todas las capas
        $current_states = array_fill(0, $this->numLayers, ['num' => array_fill(0, $this->embedDim, 0.0), 'den' => array_fill(0, $this->embedDim, 0.0)]);

        // Forward pass: guardar todo
        for ($pos = 0; $pos < $len - 1; $pos++) {
            $x = $this->embeddings[$ids[$pos]] ?? array_fill(0, $this->embedDim, 0.0);
            $inputs[] = $x;

            // Pasar por capas
            for ($l = 0; $l < $this->numLayers; $l++) $x = $this->rwkvBlocks[$l]->forward($x, $current_states[$l]);
            $outputs[] = $x; // salida de la última capa en este paso

            // Guardar estados después del paso
            $states[$pos] = $current_states;
            /*$states[$pos] = array_map(function($st) {
                return ['num' => $st['num'], 'den' => $st['den']];
            }, $current_states);*/

            // Calcular logits (producto punto con embeddings normalizados)
            $logits = [];
            $norm_out = $this->normalizeVector($x);
            foreach ($this->embeddings as $emb_id => $emb) {
                $dot = 0.0;
                for ($d = 0; $d < $this->embedDim; $d++) $dot += $norm_out[$d] * $emb[$d];
                $logits[$emb_id] = $dot;
            }
            $logits_list[] = $logits;
        }

        // --- Backward pass (BPTT) ---
        // Inicializar acumuladores de gradientes de estados por capa y paso
        $dstate_acum = [];
        for ($l = 0; $l < $this->numLayers; $l++) {
            for ($t = 0; $t < $len - 1; $t++) $dstate_acum[$l][$t] = ['num' => array_fill(0, $this->embedDim, 0.0), 'den' => array_fill(0, $this->embedDim, 0.0)];
        }

        // Bucle desde el último paso hasta el primero
        for ($t = $len - 2; $t >= 0; $t--) {
            // Gradiente de la pérdida en este paso
            $target = $ids[$t + 1];
            $logits = $logits_list[$t];
            // Softmax
            $max = max($logits);
            $sumExp = 0.0;
            foreach ($logits as $l) $sumExp += exp($l - $max);
            $softmax = [];
            foreach ($logits as $id => $l) $softmax[$id] = exp($l - $max) / $sumExp;
            // Derivada cross-entropy: softmax - one_hot
            $dlogits = [];
            foreach ($softmax as $id => $p) $dlogits[$id] = $p - ($id == $target ? 1.0 : 0.0);

            // Gradiente respecto a la salida de la última capa (output)
            $dout = array_fill(0, $this->embedDim, 0.0);
            foreach ($dlogits as $id => $dl) {
                if (isset($this->embeddings[$id])) {
                    $emb = $this->embeddings[$id];
                    for ($d = 0; $d < $this->embedDim; $d++) $dout[$d] += $dl * $emb[$d];
                }
            }

            // Propagar hacia atrás a través de las capas
            $dy = $dout;
            for ($l = $this->numLayers - 1; $l >= 0; $l--) {
                $block = $this->rwkvBlocks[$l];
                // Establecer el estado de la capa en este paso (debe coincidir con el forward)
                // No necesitamos pasarlo porque el bloque ya lo tiene en su caché del forward (se guardó al llamar a forward)
                // Llamamos a backward
                $res = $block->backward($dy);
                $dx = $res['dx'];
                $dnum_prev = $res['dnum_prev'];
                $dden_prev = $res['dden_prev'];

                // Acumular gradientes de estados previos (para el paso t-1)
                if ($t > 0) {
                    for ($i = 0; $i < $this->embedDim; $i++) {
                        $dstate_acum[$l][$t-1]['num'][$i] += $dnum_prev[$i];
                        $dstate_acum[$l][$t-1]['den'][$i] += $dden_prev[$i];
                    }
                }
                // El gradiente para la capa inferior es $dx
                $dy = $dx;
            }

            // Actualizar embedding del token de entrada ($ids[$t])
            $input_id = $ids[$t];
            if (isset($this->embeddings[$input_id])) {
                for ($d = 0; $d < $this->embedDim; $d++) $this->embeddings[$input_id][$d] -= $this->learningRate * $dy[$d];
                $this->embeddings[$input_id] = $this->normalizeVector($this->embeddings[$input_id]);
            }

            foreach ($dlogits as $id => $dl) {
                if ($dl != 0 && isset($this->embeddings[$id])) {
                    $factor = -$this->learningRate * $dl; // negativo porque queremos minimizar pérdida
                    $out_norm = $this->normalizeVector($outputs[$t]);
                    for ($d = 0; $d < $this->embedDim; $d++) $this->embeddings[$id][$d] += $factor * $out_norm[$d];
                    $this->embeddings[$id] = $this->normalizeVector($this->embeddings[$id]);
                }
            }
        }

        // Aplicar gradientes acumulados en los bloques RWKV
        for ($l = 0; $l < $this->numLayers; $l++) $this->rwkvBlocks[$l]->applyGradients($this->optimizers["block_$l"], $this->learningRate);

        // Guardar modelo
        $this->tokenizer->save($this->modelDir . '/tokenizer.json');
        $this->saveEmbeddings($this->modelDir . '/embeddings.bin');
        $this->saveRWKV($this->modelDir . '/rwkv.bin');
    }

    public function generate(
        string $prompt,
        int $maxTokens = 50,
        float $temperature = 0.8,
        ?int $topK = null,
        float $frequencyPenalty = 0.0,
        array $stopTokens = [],
        ?float $topP = null,
        float $repetitionPenalty = 1.0,
        float $presencePenalty = 0.0
    ): string {
        if (!str_contains($prompt, '<|')) $prompt = "<|USER|>\n" . trim($prompt) . "\n<|EOS|>\n<|ASSISTANT|>\n";
        $prompt = preg_replace('/\s*(<\|[^|]+\|>)\s*/', '$1', $prompt);
        $allStopTokens = array_merge($stopTokens, ['<|EOS|>']);
        $tokens = $this->tokenizer->tokenize($prompt, true);
        $ids = $this->tokenizer->encode($tokens);
        if ($this->tokenizer->getVocabSize() === 0) return $prompt;

        // Estados para generación
        $states = array_fill(0, $this->numLayers, ['num' => array_fill(0, $this->embedDim, 0.0), 'den' => array_fill(0, $this->embedDim, 0.0)]);
        $lastOutput = null;
        foreach ($ids as $id) {
            $x = $this->embeddings[$id] ?? array_fill(0, $this->embedDim, 0.0);
            for ($l = 0; $l < $this->numLayers; $l++) $x = $this->rwkvBlocks[$l]->forward($x, $states[$l]);
            $lastOutput = $x;
        }

        $generatedIds = [];
        $freqCount = [];

        for ($i = 0; $i < $maxTokens; $i++) {
            $lastNorm = $this->normalizeVector($lastOutput);
            $logits = [];
            foreach ($this->embeddings as $id => $emb) {
                $dot = 0.0;
                for ($j = 0; $j < $this->embedDim; $j++) $dot += $lastNorm[$j] * $emb[$j];
                $logits[$id] = $dot;
            }

            // Aplicar temperatura y sampling
            $maxLogit = max($logits);
            $expSum = 0.0;
            foreach ($logits as $id => $l) {
                $exp = exp(($l - $maxLogit) / max($temperature, 0.01));
                $logits[$id] = $exp;
                $expSum += $exp;
            }
            $probs = [];
            foreach ($logits as $id => $e) $probs[$id] = $e / $expSum;

            if ($topK !== null && $topK > 0 && count($probs) > $topK) {
                arsort($probs);
                $probs = array_slice($probs, 0, $topK, true);
                $sum = array_sum($probs);
                foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
            }

            if ($topP !== null && $topP > 0.0 && $topP < 1.0) {
                arsort($probs);
                $cum = 0.0;
                $filtered = [];
                foreach ($probs as $id => $p) {
                    $cum += $p;
                    $filtered[$id] = $p;
                    if ($cum >= $topP) break;
                }
                $sum = array_sum($filtered);
                foreach ($filtered as $id => $p) $filtered[$id] = $p / $sum;
                $probs = $filtered;
            }

            // Penalizaciones
            if ($frequencyPenalty > 0) {
                foreach ($probs as $id => $p) {
                    $count = $freqCount[$id] ?? 0;
                    $probs[$id] = $p / (1 + $frequencyPenalty * $count);
                }
                $sum = array_sum($probs);
                foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
            }

            if ($repetitionPenalty != 1.0) {
                foreach ($probs as $id => $p) {
                    if (isset($freqCount[$id])) $probs[$id] = $p / $repetitionPenalty;
                }
                $sum = array_sum($probs);
                foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
            }

            if ($presencePenalty != 0.0) {
                $factor = exp(-$presencePenalty);
                foreach ($probs as $id => $p) {
                    if (isset($freqCount[$id])) $probs[$id] = $p * $factor;
                }
                $sum = array_sum($probs);
                foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
            }

            // Muestreo
            $rand = mt_rand() / mt_getrandmax();
            $cum = 0.0;
            $selected = null;
            foreach ($probs as $id => $p) {
                $cum += $p;
                if ($rand <= $cum) { $selected = $id; break; }
            }
            if ($selected === null) $selected = array_key_first($probs) ?? 0;

            $token = $this->tokenizer->decode([$selected])[0];
            if (in_array($token, $allStopTokens, true)) break;

            $generatedIds[] = $selected;
            $freqCount[$selected] = ($freqCount[$selected] ?? 0) + 1;

            // Forward del token seleccionado para continuar
            $x = $this->embeddings[$selected] ?? array_fill(0, $this->embedDim, 0.0);
            for ($l = 0; $l < $this->numLayers; $l++) $x = $this->rwkvBlocks[$l]->forward($x, $states[$l]);
            $lastOutput = $x;
        }

        return trim($this->joinTokens(array_filter($this->tokenizer->decode($generatedIds), fn($t) => !in_array($t, $this->specialTokens))));
    }

    private function ensureEmbeddingsSize(): void {
        $vocabSize = $this->tokenizer->getVocabSize();
        $currentSize = count($this->embeddings);
        if ($vocabSize > $currentSize) {
            for ($i = $currentSize; $i < $vocabSize; $i++) {
                $vec = $this->randomVector();
                $this->embeddings[$i] = $this->normalizeVector($vec);
            }
        }
    }

    private function joinTokens(array $tokens): string {
        $result = '';
        $prevToken = '';
        foreach ($tokens as $token) {
            if ($token === "\n") { $result .= "\n"; $prevToken = "\n"; continue; }
            $addSpace = false;
            if ($result !== '') {
                if (!in_array($token, $this->noSpaceBefore) && !in_array($prevToken, $this->noSpaceAfter)) $addSpace = true;
            }
            if ($addSpace) $result .= ' ';
            $result .= $token;
            $prevToken = $token;
        }
        return $result;
    }

    public function deleteModel(): void {
        $files = glob($this->modelDir . '/*.bin');
        foreach ($files as $f) unlink($f);
        $tokenizerFile = $this->modelDir . '/tokenizer.json';
        if (file_exists($tokenizerFile)) unlink($tokenizerFile);
        rmdir($this->modelDir);
    }

    public function getVocabSize(): int {
        return $this->tokenizer->getVocabSize();
    }
}