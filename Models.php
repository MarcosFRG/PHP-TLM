<?php
$models = file_exists('all-models')?scandir('all-models'):[];
header('Content-Type: application/json');
if(empty($models)){
  echo '{"success":false,"error":"No hay modelos disponibles"}';
}else{
  echo json_encode($models);
}
?>