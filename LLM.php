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

class LinearAttention {
  public array $Wq, $Wk, $Wv;
  public int $dim;

  public function __construct(int $dim) {
    $this->dim = $dim;
    $this->Wq = $this->randomMatrix($dim, $dim);
    $this->Wk = $this->randomMatrix($dim, $dim);
    $this->Wv = $this->randomMatrix($dim, $dim);
  }

  private function randomMatrix(int $rows, int $cols): array {
    $m = [];
    for ($i = 0; $i < $rows; $i++) {
      for ($j = 0; $j < $cols; $j++) {
        $m[$i][$j] = (mt_rand(-100, 100) / 1000);
      }
    }
    return $m;
  }

  public function matMul(array $mat, array $vec): array {
    $res = array_fill(0, count($mat), 0.0);
    for ($i = 0; $i < count($mat); $i++) {
      for ($j = 0; $j < count($vec); $j++) {
        $res[$i] += $mat[$i][$j] * $vec[$j];
      }
    }
    return $res;
  }
}

class TemporalConvLayer {
  public array $filters;
  public int $kernelSize = 3;
  public int $dim;

  public function __construct(int $dim) {
    $this->dim = $dim;
    $this->filters = $this->randomMatrix($dim, $this->kernelSize);
  }

  private function randomMatrix(int $rows, int $cols): array {
    $m = [];
    for ($i = 0; $i < $rows; $i++) {
      for ($j = 0; $j < $cols; $j++) {
        $m[$i][$j] = (mt_rand(-100, 100) / 1000);
      }
    }
    return $m;
  }
}

class ChannelMixer {
  public array $W1, $W2;
  public int $hiddenDim;

  public function __construct(int $dim) {
    $this->hiddenDim = $dim * 4;
    $this->W1 = $this->randomMatrix($this->hiddenDim, $dim);
    $this->W2 = $this->randomMatrix($dim, $this->hiddenDim);
  }

  private function randomMatrix(int $rows, int $cols): array {
    $m = [];
    for ($i = 0; $i < $rows; $i++) {
      for ($j = 0; $j < $cols; $j++) {
        $m[$i][$j] = (mt_rand(-100, 100) / 1000);
      }
    }
    return $m;
  }

  public function matMul(array $mat, array $vec): array {
    $res = array_fill(0, count($mat), 0.0);
    for ($i = 0; $i < count($mat); $i++) {
      for ($j = 0; $j < count($vec); $j++) {
        $res[$i] += $mat[$i][$j] * $vec[$j];
      }
    }
    return $res;
  }

  public function forward(array $x): array {
    $hidden = $this->matMul($this->W1, $x);
    $hidden = array_map(fn($v) => max(0, $v), $hidden);
    $gated = array_map(fn($v) => $v * $v, $hidden);
    return $this->matMul($this->W2, $gated);
  }
}

class ConversationalMemory {
  private array $compressed = [];
  private array $recent = [];
  private int $maxRecent = 20;
  private int $dim;
  private float $compressionThreshold = 0.7;

  public function __construct(int $dim) {
    $this->dim = $dim;
  }

  public function add(array $vector, int $position): void {
    $this->recent[] = [
      'vec' => $vector,
      'pos' => $position,
      'time' => microtime(true)
    ];
    
    if (count($this->recent) > $this->maxRecent) {
      $this->compress();
    }
  }

  private function compress(): void {
    $toCompress = array_splice($this->recent, 0, 10);
    if (empty($toCompress)) return;

    $avg = array_fill(0, $this->dim, 0.0);
    foreach ($toCompress as $item) {
      for ($i = 0; $i < $this->dim; $i++) $avg[$i] += $item['vec'][$i];
    }
    for ($i = 0; $i < $this->dim; $i++) $avg[$i] /= count($toCompress);

    $this->compressed[] = [
      'vec' => $avg,
      'start' => $toCompress[0]['pos'],
      'end' => $toCompress[count($toCompress)-1]['pos'],
      'time' => microtime(true)
    ];

    if (count($this->compressed) > 20) array_shift($this->compressed);
  }

  public function retrieve(array $queryVec, int $currentPos): array {
    $results = [];

    foreach ($this->recent as $r) {
      $sim = $this->cosineSimilarity($queryVec, $r['vec']);
      $recency = 1.0 / (1.0 + ($currentPos - $r['pos']) * 0.1);
      $results[] = ['vec' => $r['vec'], 'score' => $sim * $recency];
    }

    foreach ($this->compressed as $c) {
      $sim = $this->cosineSimilarity($queryVec, $c['vec']);
      $positionBonus = ($c['end'] > $currentPos - 50) ? 0.5 : 0.0;
      $results[] = ['vec' => $c['vec'], 'score' => $sim * 0.8 + $positionBonus];
    }

    usort($results, fn($a,$b) => $b['score'] <=> $a['score']);
    return array_slice($results, 0, 3);
  }

  private function cosineSimilarity(array $a, array $b): float {
    $dot = 0.0; $na = 0.0; $nb = 0.0;
    for ($i = 0; $i < $this->dim; $i++) {
      $dot += $a[$i] * $b[$i];
      $na += $a[$i] * $a[$i];
      $nb += $b[$i] * $b[$i];
    }
    if ($na == 0 || $nb == 0) return 0;
    return $dot / (sqrt($na) * sqrt($nb));
  }

  public function clear(): void {
    $this->recent = [];
    $this->compressed = [];
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
  private float $learningRate = 0.01;

  private array $layers = [];
  private int $numLayers = 4;

  // Caches para generación incremental
  private array $kvCache = [];
  private array $lastLayerOutputs = [];
  private ?array $tokenClusters = null;
  private array $vocabIds = [];

  public function __construct(string $modelDir, int $maxContext = 512) {
    $this->modelDir = $modelDir;
    $this->maxContext = $maxContext;
    if (!is_dir($modelDir)) mkdir($modelDir, 0777, true);

    $tokenizerPath = $modelDir . '/tokenizer.json';
    $triePath = $modelDir . '/model.ppm';
    $embedPath = $modelDir . '/embeddings.bin';

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

    for ($i = 0; $i < $this->numLayers; $i++) {
      $this->layers[] = [
        'attn' => new LinearAttention($this->embedDim),
        'conv' => new TemporalConvLayer($this->embedDim),
        'mix' => new ChannelMixer($this->embedDim)
      ];
    }

    $this->vocabIds = array_keys($this->embeddings);
    // Los clusters se construirán bajo demanda en generate
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

  private function layerNorm(array $vec): array {
    $mean = array_sum($vec) / $this->embedDim;
    $variance = 0.0;
    foreach ($vec as $v) $variance += ($v - $mean) ** 2;
    $variance /= $this->embedDim;
    $std = sqrt($variance + 1e-5);
    $out = [];
    for ($i = 0; $i < $this->embedDim; $i++) $out[$i] = ($vec[$i] - $mean) / $std;
    return $out;
  }

  // Versión incremental: procesa un nuevo token dado el contexto anterior
  private function forwardIncremental(int $newTokenId, array $prevTokenIds, ConversationalMemory $mem): array {
    $x = $this->embeddings[$newTokenId] ?? array_fill(0, $this->embedDim, 0.0);
    $pos = count($prevTokenIds); // posición actual

    foreach ($this->layers as $layerIdx => $layer) {
      // Atención incremental con KV cache
      $attOut = $this->incrementalAttention($x, $layer['attn'], $layerIdx, $prevTokenIds);
      for ($d = 0; $d < $this->embedDim; $d++) $x[$d] += $attOut[$d];

      // Convolución incremental
      $convOut = $this->incrementalConv($x, $layer['conv'], $layerIdx, $prevTokenIds);
      for ($d = 0; $d < $this->embedDim; $d++) $x[$d] += $convOut[$d];

      // Mixer (igual, pero solo para este vector)
      $mixed = $layer['mix']->forward($x);
      for ($d = 0; $d < $this->embedDim; $d++) $x[$d] += $mixed[$d];

      // LayerNorm
      $x = $this->layerNorm($x);
      $this->lastLayerOutputs[$layerIdx] = $x;
    }

    $mem->add($x, $pos);
    return $x;
  }

  private function incrementalAttention(array $x, LinearAttention $attn, int $layerIdx, array $prevTokenIds): array {
    // Calcular Q para el nuevo token
    $q = $attn->matMul($attn->Wq, $x);

    // Inicializar cache si es necesario
    if (!isset($this->kvCache[$layerIdx])) {
      $this->kvCache[$layerIdx] = ['k' => [], 'v' => []];
    }

    // Calcular K y V para los tokens previos que falten
    $start = count($this->kvCache[$layerIdx]['k']);
    for ($i = $start; $i < count($prevTokenIds); $i++) {
      $tokenVec = $this->embeddings[$prevTokenIds[$i]] ?? array_fill(0, $this->embedDim, 0.0);
      $this->kvCache[$layerIdx]['k'][] = $attn->matMul($attn->Wk, $tokenVec);
      $this->kvCache[$layerIdx]['v'][] = $attn->matMul($attn->Wv, $tokenVec);
    }

    $K = $this->kvCache[$layerIdx]['k'];
    $V = $this->kvCache[$layerIdx]['v'];
    $out = array_fill(0, $this->embedDim, 0.0);

    if (empty($K)) return $out;

    // Calcular pesos de atención para este token
    $expSum = 0.0;
    $weights = [];
    for ($j = 0; $j < count($K); $j++) {
      $score = 0.0;
      for ($d = 0; $d < $this->embedDim; $d++) {
        $score += $q[$d] * $K[$j][$d];
      }
      $score = exp($score / sqrt($this->embedDim));
      $weights[$j] = $score;
      $expSum += $score;
    }

    if ($expSum > 0) {
      foreach ($weights as $j => $w) {
        $w /= $expSum;
        for ($d = 0; $d < $this->embedDim; $d++) {
          $out[$d] += $w * $V[$j][$d];
        }
      }
    }

    return $out;
  }

  private function incrementalConv(array $x, TemporalConvLayer $conv, int $layerIdx, array $prevTokenIds): array {
    $kernelSize = $conv->kernelSize;
    $dim = $this->embedDim;

    // Obtener vectores de los últimos tokens (incluyendo el actual)
    $contextVecs = [];
    $start = max(0, count($prevTokenIds) - $kernelSize + 1);
    for ($i = $start; $i < count($prevTokenIds); $i++) {
      $contextVecs[] = $this->embeddings[$prevTokenIds[$i]] ?? array_fill(0, $dim, 0.0);
    }
    $contextVecs[] = $x; // el token actual

    // Rellenar con ceros si es necesario
    while (count($contextVecs) < $kernelSize) {
      array_unshift($contextVecs, array_fill(0, $dim, 0.0));
    }

    // Aplicar filtros
    $out = array_fill(0, $dim, 0.0);
    for ($d = 0; $d < $dim; $d++) {
      for ($k = 0; $k < $kernelSize; $k++) {
        $out[$d] += $conv->filters[$d][$k] * $contextVecs[$k][$d];
      }
    }

    return $out;
  }

  // Construye clusters de tokens para búsqueda rápida
  private function buildTokenClusters(): array {
    $k = 20; // número de clusters
    $vocabIds = $this->vocabIds;
    if (count($vocabIds) < $k) $k = max(1, count($vocabIds));

    // Inicializar centros aleatorios
    shuffle($vocabIds);
    $centroids = [];
    for ($i = 0; $i < $k; $i++) {
      $centroids[$i] = $this->embeddings[$vocabIds[$i]];
    }

    // 5 iteraciones de k-means
    $assignments = [];
    for ($iter = 0; $iter < 5; $iter++) {
      $assignments = array_fill(0, $k, []);
      foreach ($vocabIds as $id) {
        $bestC = 0;
        $bestSim = -INF;
        foreach ($centroids as $cid => $cent) {
          $sim = $this->cosineSimilarity($this->embeddings[$id], $cent);
          if ($sim > $bestSim) {
            $bestSim = $sim;
            $bestC = $cid;
          }
        }
        $assignments[$bestC][] = $id;
      }

      // Recalcular centros
      foreach ($assignments as $cid => $ids) {
        if (empty($ids)) continue;
        $newCent = array_fill(0, $this->embedDim, 0.0);
        foreach ($ids as $id) {
          $vec = $this->embeddings[$id];
          for ($d = 0; $d < $this->embedDim; $d++) $newCent[$d] += $vec[$d];
        }
        for ($d = 0; $d < $this->embedDim; $d++) $newCent[$d] /= count($ids);
        $centroids[$cid] = $newCent;
      }
    }

    $result = [];
    foreach ($assignments as $cid => $ids) {
      if (empty($ids)) continue;
      $result[$cid] = [
        'centroid' => $centroids[$cid],
        'tokens' => $ids
      ];
    }
    return $result;
  }

  // Búsqueda rápida de tokens usando clustering
  private function fastTokenSearch(array $queryVec, array $freqCount, float $temperature, ?int $topK): array {
    if ($this->tokenClusters === null) {
      $this->tokenClusters = $this->buildTokenClusters();
    }

    // Encontrar el cluster más cercano
    $bestCluster = null;
    $bestSim = -INF;
    foreach ($this->tokenClusters as $cid => $cluster) {
      $sim = $this->cosineSimilarity($queryVec, $cluster['centroid']);
      if ($sim > $bestSim) {
        $bestSim = $sim;
        $bestCluster = $cid;
      }
    }

    if ($bestCluster === null) {
      // Fallback a todos los tokens
      $candidates = $this->vocabIds;
    } else {
      $candidates = $this->tokenClusters[$bestCluster]['tokens'];
      // Añadir tokens de clusters cercanos para mayor cobertura (opcional)
      // Por simplicidad, solo usamos el mejor cluster.
    }

    $probs = [];
    foreach ($candidates as $id) {
      $sim = $this->cosineSimilarity($queryVec, $this->embeddings[$id]);
      $probs[$id] = exp($sim / max($temperature, 0.01));
    }

    $sum = array_sum($probs);
    if ($sum > 0) {
      foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
    }

    // Aplicar topK si se solicita
    if ($topK !== null && $topK > 0 && count($probs) > $topK) {
      arsort($probs);
      $probs = array_slice($probs, 0, $topK, true);
      $sum = array_sum($probs);
      if ($sum > 0) {
        foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
      }
    }

    return $probs;
  }

  public function train(string $text): void {
    // Dividir por lotes usando líneas vacías
    $batches = preg_split('/\n\s*\n/', $text);
    $totalTokens = 0;

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
        if (!isset($this->embeddings[$next])) {
          $this->embeddings[$next] = $this->randomVector();
        }
        $contextAvg = $this->averageEmbedding($context);
        $current = $this->embeddings[$next];
        for ($d = 0; $d < $this->embedDim; $d++) {
          $this->embeddings[$next][$d] += $this->learningRate * ($contextAvg[$d] - $current[$d]);
        }
      }

      $totalTokens += $len;
    }

    $this->tokenizer->save($this->modelDir . '/tokenizer.json');
    $this->trie->save($this->modelDir . '/model.ppm');
    $this->saveEmbeddings($this->modelDir . '/embeddings.bin');

    // Invalidar clusters después de entrenar (se reconstruirán en la próxima generación)
    $this->tokenClusters = null;
    $this->vocabIds = array_keys($this->embeddings);
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

    // Reiniciar caches para esta generación
    $this->kvCache = [];
    $this->lastLayerOutputs = [];

    $mem = new ConversationalMemory($this->embedDim);
    $generatedIds = [];
    $freqCount = [];
    $currentIds = $ids;

    // Añadir tokens iniciales a la memoria
    for ($pos = 0; $pos < count($currentIds); $pos++) {
      $vec = $this->embeddings[$currentIds[$pos]] ?? array_fill(0, $this->embedDim, 0.0);
      $mem->add($vec, $pos);
    }

    for ($i = 0; $i < $maxTokens; $i++) {
      // El último token es el que vamos a procesar (inicialmente el último del prompt)
      $lastTokenId = $currentIds[count($currentIds)-1];
      $prevTokenIds = array_slice($currentIds, 0, -1);

      $lastVec = $this->forwardIncremental($lastTokenId, $prevTokenIds, $mem);

      // Obtener probabilidades mediante búsqueda rápida
      $probs = $this->fastTokenSearch($lastVec, $freqCount, $temperature, $topK);

      // Aplicar penalizaciones (igual que antes)
      if ($frequencyPenalty > 0) {
        foreach ($probs as $id => $p) {
          $count = $freqCount[$id] ?? 0;
          $probs[$id] = $p / (1 + $frequencyPenalty * $count);
        }
        $sum = array_sum($probs);
        if ($sum > 0) {
          foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
        }
      }

      if ($repetitionPenalty != 1.0) {
        foreach ($probs as $id => $p) {
          if (isset($freqCount[$id])) $probs[$id] = $p / $repetitionPenalty;
        }
        $sum = array_sum($probs);
        if ($sum > 0) {
          foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
        }
      }

      if ($presencePenalty != 0.0) {
        $factor = exp(-$presencePenalty);
        foreach ($probs as $id => $p) {
          if (isset($freqCount[$id])) $probs[$id] = $p * $factor;
        }
        $sum = array_sum($probs);
        if ($sum > 0) {
          foreach ($probs as $id => $p) $probs[$id] = $p / $sum;
        }
      }

      // Top-P (nucleus sampling)
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
        if ($sum > 0) {
          foreach ($filtered as $id => $p) $filtered[$id] = $p / $sum;
        }
        $probs = $filtered;
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

      // El nuevo token se añadirá a la memoria en la siguiente iteración (ya se añadió en forwardIncremental)
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