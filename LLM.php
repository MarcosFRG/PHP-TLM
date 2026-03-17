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
    if($useCache) $this->cache[$text] = $tokens;
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

class PPMTrie {
  private array $root;
  private array $globalCounts;
  private int $totalTokens;

  public function __construct() {
    $this->root = [];
    $this->globalCounts = [];
    $this->totalTokens = 0;
  }

  public function add(array $context, int $nextToken): void {
    $node = &$this->root;
    foreach ($context as $token) {
      if (!isset($node['children'][$token])) $node['children'][$token] = ['counts' => [], 'children' => []];
      $node = &$node['children'][$token];
    }
    if (!isset($node['counts'][$nextToken])) {
      $node['counts'][$nextToken] = 1;
    } else {
      $node['counts'][$nextToken]++;
    }
    if (!isset($this->globalCounts[$nextToken])) {
      $this->globalCounts[$nextToken] = 1;
    } else {
      $this->globalCounts[$nextToken]++;
    }
    $this->totalTokens++;
  }

  private function getNode(array $context): ?array {
    $node = $this->root;
    foreach ($context as $token) {
      if (!isset($node['children'][$token])) return null;
      $node = $node['children'][$token];
    }
    return $node;
  }

  public function predict(array $context, int $maxOrder, float $temperature = 1.0, ?int $topK = null, ?float $topP = null): array {
    $order = min(count($context), $maxOrder);
    $probs = [];
    $weight = 1.0;
    $seen = [];

    for ($o = $order; $o >= 0; $o--) {
      $subContext = array_slice($context, -$o);
      $node = $o === 0 ? ['counts' => $this->globalCounts] : $this->getNode($subContext);
      if (!$node) continue;

      $counts = $node['counts'];
      if (empty($counts)) continue;

      $total = array_sum($counts);
      $numSymbols = count($counts);
      $escapeProb = $numSymbols / ($total + $numSymbols);
      $factor = $weight * (1 - $escapeProb);

      foreach ($counts as $token => $freq) {
        if (!isset($seen[$token])) {
          $probs[$token] = ($probs[$token] ?? 0) + $factor * ($freq / $total);
          $seen[$token] = true;
        }
      }
      $weight *= $escapeProb;
      if ($weight < 1e-6) break;
    }

    if (empty($probs)) $probs[0] = 1.0;

    if ($temperature != 1.0) {
      $sum = 0.0;
      foreach ($probs as $token => $p) {
        $p = pow($p, 1.0 / max($temperature, 0.01));
        $probs[$token] = $p;
        $sum += $p;
      }
      if ($sum > 0) {
        foreach ($probs as $token => $p) $probs[$token] = $p / $sum;
      }
    }

    if ($topK !== null && $topK > 0 && count($probs) > $topK) {
      arsort($probs);
      $probs = array_slice($probs, 0, $topK, true);
      $sum = array_sum($probs);
      if ($sum > 0) {
        foreach ($probs as $token => $p) $probs[$token] = $p / $sum;
      }
    }

    if ($topP !== null && $topP > 0.0 && $topP < 1.0) {
      arsort($probs);
      $cum = 0.0;
      $filtered = [];
      foreach ($probs as $token => $p) {
        $cum += $p;
        $filtered[$token] = $p;
        if ($cum >= $topP) break;
      }
      $sum = array_sum($filtered);
      if ($sum > 0) {
        foreach ($filtered as $token => $p) $filtered[$token] = $p / $sum;
      }
      $probs = $filtered;
    }

    return $probs;
  }

  public function save(string $path): void {
    $fp = fopen($path, 'wb');
    if (!$fp) return;

    $writeNode = function($node) use (&$writeNode, $fp) {
      $counts = $node['counts'] ?? [];
      fwrite($fp, pack('V', count($counts)));
      foreach ($counts as $token => $count) {
        fwrite($fp, pack('V', $token));
        fwrite($fp, pack('V', $count));
      }
      $children = $node['children'] ?? [];
      fwrite($fp, pack('V', count($children)));
      foreach ($children as $token => $child) {
        fwrite($fp, pack('V', $token));
        $writeNode($child);
      }
    };

    fwrite($fp, pack('V', count($this->globalCounts)));
    foreach ($this->globalCounts as $token => $count) {
      fwrite($fp, pack('V', $token));
      fwrite($fp, pack('V', $count));
    }

    fwrite($fp, pack('V', $this->totalTokens));
    $writeNode($this->root);
    fclose($fp);
  }

  public function load(string $path): void {
    if (!file_exists($path)) return;

    $fp = fopen($path, 'rb');
    if (!$fp) return;

    $readNode = function() use (&$readNode, $fp) {
      $node = ['counts' => [], 'children' => []];
      $countSize = unpack('V', fread($fp, 4))[1];
      for ($i = 0; $i < $countSize; $i++) {
        $token = unpack('V', fread($fp, 4))[1];
        $count = unpack('V', fread($fp, 4))[1];
        $node['counts'][$token] = $count;
      }
      $childrenSize = unpack('V', fread($fp, 4))[1];
      for ($i = 0; $i < $childrenSize; $i++) {
        $token = unpack('V', fread($fp, 4))[1];
        $node['children'][$token] = $readNode();
      }
      return $node;
    };

    $globalSize = unpack('V', fread($fp, 4))[1];
    for ($i = 0; $i < $globalSize; $i++) {
      $token = unpack('V', fread($fp, 4))[1];
      $count = unpack('V', fread($fp, 4))[1];
      $this->globalCounts[$token] = $count;
    }

    $this->totalTokens = unpack('V', fread($fp, 4))[1];
    $this->root = $readNode();
    fclose($fp);
  }
}

class RWKVBlock {
  public array $wk;
  public array $wv;
  public array $wr;
  public array $ww;
  public array $w1;
  public array $w2;
  public int $dim;
  public array $state;

  public function __construct(int $dim) {
    $this->dim = $dim;
    $this->wk = $this->randomMatrix($dim, $dim);
    $this->wv = $this->randomMatrix($dim, $dim);
    $this->wr = $this->randomMatrix($dim, $dim);
    $this->ww = $this->randomVector($dim);
    $this->w1 = $this->randomMatrix($dim * 4, $dim);
    $this->w2 = $this->randomMatrix($dim, $dim * 4);
    $this->state = array_fill(0, $dim, 0.0);
  }

  private function randomMatrix(int $rows, int $cols): array {
    $m = [];
    for ($i = 0; $i < $rows; $i++) {
      for ($j = 0; $j < $cols; $j++) $m[$i][$j] = (mt_rand(-100, 100) / 1000);
    }
    return $m;
  }

  private function randomVector(int $dim): array {
    $v = [];
    for ($i = 0; $i < $dim; $i++) $v[] = (mt_rand(-100, 100) / 1000);
    return $v;
  }

  private function matMul(array $mat, array $vec): array {
    $res = array_fill(0, count($mat), 0.0);
    for ($i = 0; $i < count($mat); $i++) {
      for ($j = 0; $j < count($vec); $j++) $res[$i] += $mat[$i][$j] * $vec[$j];
    }
    return $res;
  }

  private function sigmoid(float $x): float {
    return 1.0 / (1.0 + exp(-$x));
  }

  // WKV computation recurrente
  private function wkv(array $k, array $v, array $w, array &$state): array {
    // w es decay por canal (vector)
    $dim = $this->dim;
    $out = array_fill(0, $dim, 0.0);
    $newState = $state;

    if (!isset($state['num'])) {
      $state['num'] = array_fill(0, $dim, 0.0);
      $state['den'] = array_fill(0, $dim, 0.0);
    }

    $num = &$state['num'];
    $den = &$state['den'];

    for ($i = 0; $i < $dim; $i++) {
      $expk = exp($k[$i]);
      $decay = exp(-$w[$i]);
      $num[$i] = $num[$i] * $decay + $v[$i] * $expk;
      $den[$i] = $den[$i] * $decay + $expk;
      $out[$i] = $num[$i] / ($den[$i] + 1e-8);
    }

    return $out;
  }

  public function timeMixing(array $x): array {
    $k = $this->matMul($this->wk, $x);
    $v = $this->matMul($this->wv, $x);
    $r = $this->matMul($this->wr, $x);
    $w = $this->ww;

    $wkv = $this->wkv($k, $v, $w, $this->state);

    $r_sig = array_map([$this, 'sigmoid'], $r);
    $out = [];
    for ($i = 0; $i < $this->dim; $i++) $out[$i] = $r_sig[$i] * $wkv[$i];
    return $out;
  }

  public function channelMixing(array $x): array {
    // FFN con gateo
    $hidden = $this->matMul($this->w1, $x);
    $hidden = array_map(fn($v) => max(0, $v), $hidden);
    $gated = array_map(fn($v) => $v * $v, $hidden);
    return $this->matMul($this->w2, $gated);
  }

  public function forward(array $x): array {
    $x = $this->timeMixing($x);
    $x = $this->channelMixing($x);
    return $x;
  }
}

class LLM {
  private Tokenizer $tokenizer;
  private PPMTrie $trie;
  private string $modelDir;
  private int $maxContext;
  private array $specialTokens = ['<|SYSTEM|>', '<|USER|>', '<|ASSISTANT|>', '<|EOS|>'];
  private array $noSpaceBefore = ['.', ',', '!', '?', ';', ':', ')', ']', '}', '”', '’', '»'];
  private array $noSpaceAfter = ['(', '[', '{', '“', '‘', '«', '¡', '¿'];

  private array $embeddings = [];
  private int $embedDim = 128;
  private float $learningRate = 0.001;

  private array $rwkvBlocks = [];
  private int $numLayers = 4;
  private array $states = [];

  public function __construct(string $modelDir, int $maxContext = 512) {
    $this->modelDir = $modelDir;
    $this->maxContext = $maxContext;
    if (!is_dir($modelDir)) mkdir($modelDir, 0777, true);

    $tokenizerPath = $modelDir . '/tokenizer.json';
    $triePath = $modelDir . '/model.ppm';
    $embedPath = $modelDir . '/embeddings.bin';
    $rwkvPath = $modelDir . '/rwkv.bin';

    if (file_exists($tokenizerPath)) {
      $this->tokenizer = new Tokenizer();
      $this->tokenizer->load($tokenizerPath);
    } else {
      $this->tokenizer = new Tokenizer();
      $this->tokenizer->encode($this->specialTokens);
    }

    if (file_exists($triePath)) {
      $this->trie = new PPMTrie();
      $this->trie->load($triePath);
    } else {
      $this->trie = new PPMTrie();
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
  }

  private function initEmbeddings(): void {
    $vocabSize = $this->tokenizer->getVocabSize();
    for ($i = 0; $i < $vocabSize; $i++) $this->embeddings[$i] = $this->randomVector();
  }

  private function randomVector(): array {
    $vec = [];
    for ($i = 0; $i < $this->embedDim; $i++) $vec[] = (mt_rand(-100, 100) / 1000);
    return $vec;
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
      for ($j = 0; $j < $dim; $j++) {
        $val = unpack('f', fread($fp, 4))[1];
        $vec[] = $val;
      }
      $this->embeddings[$id] = $vec;
    }
    fclose($fp);
  }

  private function saveRWKV(string $path): void {
    $fp = fopen($path, 'wb');
    if (!$fp) return;

    fwrite($fp, pack('V', count($this->rwkvBlocks))); // numLayers
    fwrite($fp, pack('V', $this->embedDim));          // dim

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
      // w1 (dim*4 filas, dim columnas)
      $rowsW1 = $dim * 4;
      for ($i = 0; $i < $rowsW1; $i++) {
        for ($j = 0; $j < $dim; $j++) fwrite($fp, pack('f', $block->w1[$i][$j]));
      }
      // w2 (dim filas, dim*4 columnas)
      $colsW2 = $dim * 4;
      for ($i = 0; $i < $dim; $i++) {
        for ($j = 0; $j < $colsW2; $j++) fwrite($fp, pack('f', $block->w2[$i][$j]));
      }
    }
    fclose($fp);
  }

  /**
   * Carga los bloques RWKV desde un archivo binario.
   */
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
      // w1 (dim * 4 filas, dim columnas)
      $rowsW1 = $dim * 4;
      for ($i = 0; $i < $rowsW1; $i++) {
        for ($j = 0; $j < $dim; $j++) $block->w1[$i][$j] = unpack('f', fread($fp, 4))[1];
      }
      // w2 (dim filas, dim*4 columnas)
      $colsW2 = $dim * 4;
      for ($i = 0; $i < $dim; $i++) {
        for ($j = 0; $j < $colsW2; $j++) $block->w2[$i][$j] = unpack('f', fread($fp, 4))[1];
      }

      $this->rwkvBlocks[] = $block;
    }
    fclose($fp);
  }

  private function averageEmbedding(array $ids): array {
    $sum = array_fill(0, $this->embedDim, 0.0);
    $count = 0;
    foreach ($ids as $id) {
      if (isset($this->embeddings[$id])) {
        $vec = $this->embeddings[$id];
        for ($i = 0; $i < $this->embedDim; $i++) $sum[$i] += $vec[$i];
        $count++;
      }
    }
    if ($count === 0) return $sum;
    for ($i = 0; $i < $this->embedDim; $i++) $sum[$i] /= $count;
    return $sum;
  }

  private function cosineSimilarity(array $a, array $b): float {
    $dot = 0.0;
    $normA = 0.0;
    $normB = 0.0;
    for ($i = 0; $i < $this->embedDim; $i++) {
      $dot += $a[$i] * $b[$i];
      $normA += $a[$i] * $a[$i];
      $normB += $b[$i] * $b[$i];
    }
    if ($normA == 0 || $normB == 0) return 0;
    return $dot / (sqrt($normA) * sqrt($normB));
  }

  // Procesa un token a través de los bloques RWKV de forma recurrente
  private function forwardRWKV(int $tokenId, array &$states): array {
    $x = $this->embeddings[$tokenId] ?? array_fill(0, $this->embedDim, 0.0);

    foreach ($this->rwkvBlocks as $i => $block) {
      // restaurar estado de la capa
      if (isset($states[$i])) {
        $block->state = $states[$i];
      } else {
        $block->state = ['num' => array_fill(0, $this->embedDim, 0.0), 'den' => array_fill(0, $this->embedDim, 0.0)];
      }
      $x = $block->forward($x);
      // guardar nuevo estado
      $states[$i] = $block->state;
    }
    return $x;
  }

  public function train(string $text): void {
    $batches = preg_split('/(?<=<\|EOS\|>)\n\s*\n\s*(?=<\|)/', $text);
    foreach ($batches as $batch) {
      $batch = trim($batch);
      if (empty($batch)) continue;
      if (!str_ends_with($batch, '<|EOS|>')) $batch .= '<|EOS|>';

      $tokens = $this->tokenizer->tokenize($batch, false);
      $ids = $this->tokenizer->encode($tokens);
      $len = count($ids);

      // Entrenar PPM
      for ($i = 0; $i < $len - 1; $i++) {
        $maxOrder = min($this->maxContext, $i + 1);
        for ($order = 1; $order <= $maxOrder; $order++) {
          $context = array_slice($ids, $i - $order + 1, $order);
          $next = $ids[$i + 1];
          $this->trie->add($context, $next);
        }
      }

      // Entrenar embeddings (Hebbian)
      for ($i = 0; $i < $len - 1; $i++) {
        $context = array_slice($ids, max(0, $i - 5), $i - max(0, $i - 5) + 1);
        $next = $ids[$i + 1];
        if (!isset($this->embeddings[$next])) $this->embeddings[$next] = $this->randomVector();
        $contextAvg = $this->averageEmbedding($context);
        $current = $this->embeddings[$next];
        for ($d = 0; $d < $this->embedDim; $d++) $this->embeddings[$next][$d] += $this->learningRate * ($contextAvg[$d] - $current[$d]);
      }
    }

    $this->tokenizer->save($this->modelDir . '/tokenizer.json');
    $this->trie->save($this->modelDir . '/model.ppm');
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
    if(!str_contains($prompt, '<|')) $prompt = "<|USER|>\n".trim($prompt)."\n<|EOS|>\n<|ASSISTANT|>\n";
    $allStopTokens = array_merge($stopTokens, ['<|EOS|>']);
    $tokens = $this->tokenizer->tokenize($prompt, true);
    $ids = $this->tokenizer->encode($tokens);
    if ($this->tokenizer->getVocabSize() === 0) return $prompt;

    $states = [];

    // Procesar todos los tokens del prompt para inicializar estados
    foreach ($ids as $id) $this->forwardRWKV($id, $states);

    $generatedIds = [];
    $freqCount = [];

    for ($i = 0; $i < $maxTokens; $i++) {
      $lastVec = $this->forwardRWKV(end($ids), $states);
      $currentVec = $lastVec;
    }

    $states = [];
    $currentIds = $ids;
    $lastOutput = null;

    foreach ($currentIds as $id) $lastOutput = $this->forwardRWKV($id, $states);

    for ($i = 0; $i < $maxTokens; $i++) {
      $logits = [];
      foreach ($this->embeddings as $id => $emb) $logits[$id] = $this->cosineSimilarity($lastOutput, $emb);

      // Aplicar temperatura
      $maxLogit = max($logits);
      $expSum = 0.0;
      foreach ($logits as $id => $l) {
        $exp = exp(($l - $maxLogit) / max($temperature, 0.01));
        $logits[$id] = $exp;
        $expSum += $exp;
      }
      $probs = [];
      foreach ($logits as $id => $e) $probs[$id] = $e / $expSum;

      // Aplicar topK, topP, penalizaciones (igual que antes)
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

      // Seleccionar token
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

      $currentIds[] = $selected;
      $generatedIds[] = $selected;
      $freqCount[$selected] = ($freqCount[$selected] ?? 0) + 1;

      // Procesar el nuevo token para actualizar estados y obtener nuevo lastOutput
      $lastOutput = $this->forwardRWKV($selected, $states);
    }

    $outputTokens = $this->tokenizer->decode($generatedIds);
    $filteredTokens = array_filter($outputTokens, fn($t) => !in_array($t, $this->specialTokens));
    return trim($this->joinTokens($filteredTokens));
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

  public function getVocabSize(): int {
    return $this->tokenizer->getVocabSize();
  }
}