<?php
$input = json_decode(file_get_contents('php://input') ?? '', true);
if(empty($input)){
  http_response_code(400);
  echo '{"success":false,"message":"Bad Request","status":400}';
  exit;
}

$model = preg_replace('/[^a-zA-Z0-9_\-]/', '', 'models/'.($input['model'] ?? 'tiny-php'));
if(!file_exists($model)){
  http_response_code(404);
  echo '{"success":false,"message":"Not Found","status":404}';
  exit;
}

$temperature = $input['temperature'] ?? 0.7;
$topP = $input['top_p'] ?? 1;
$topK = $input['top_k'] ?? 0;
$messages = $input['messages'] ?? [];
$gInput = '';

foreach($messages as $msg){
  switch($msg['role'] ?? ''){
    case 'system':
      $gInput .= "<|SYSTEM|>\n{$msg['content']}\n<|EOS|>\n";
      break;
    case 'assistant':
      $gInput .= "<|ASSISTANT|>\n{$msg['content']}\n<|EOS|>\n";
      break;
    case 'user':
      $text = $msg['content'] ?? $msg['content']['text'] ?? '';
      $gInput .= "<|USER|>\n$text\n<|EOS|>\n";
  }
}

$maxTokens = $input['max_tokens'] ?? 50;
$repetitionPenalty = $input['repetition_penalty'] ?? 1;
$frequencyPenalty = $input['frequency_penalty'] ?? 0;
$presencePenalty = $input['presence_penalty'] ?? 0;

require_once "LLM.php";
$llm = new LLM($model, 512);

$topK = $topK > 0 ? $topK : null;

$startTime = microtime(true);
$response = $llm->generate($gInput."<|ASSISTANT|>\n", $maxTokens, $temperature, $topK, $frequencyPenalty, [], $topP, $repetitionPenalty, $presencePenalty);
$msTime = round((microtime(true) - $startTime) * 1000);

$promptTokens = count($llm->tokenizer->tokenize($gInput));
$outputTokens = count($llm->tokenizer->tokenize($response));

echo json_encode(['success' => true, 'id' => 'chatcmpl-'.uniqid(), 'choices' => [['message' => ['role' => 'assistant', 'content' => $response]]], 'usage' => ['prompt_tokens' => $promptTokens, 'completion_tokens' => $outputTokens, 'total_tokens' => $promptTokens + $outputTokens], 'timing_ms' => $msTime]);
?>