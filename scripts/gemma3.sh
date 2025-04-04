./dllama inference \
  --model dllama_model_gemma3-4b-it_q40.m \
  --tokenizer dllama_tokenizer_gemma3-4b-it.t \
  --buffer-float-type q80 \
  --max-seq-len 128 \
  --prompt "Hello world"

./dllama chat \
  --model dllama_model_gemma3-4b-it_q40.m \
  --tokenizer dllama_tokenizer_gemma3-4b-it.t \
  --buffer-float-type q80 \
  --prompt "Hello world"


./dllama inference \
  --model dllama_model_gemma3-4b-it_f32.m \
  --tokenizer dllama_tokenizer_gemma3-4b-it.t \
  --buffer-float-type f32 \
  --prompt "Hello world"