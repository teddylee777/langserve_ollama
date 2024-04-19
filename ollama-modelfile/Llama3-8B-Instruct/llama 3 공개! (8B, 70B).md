# Llama 3 공개(2024.04.19)

- https://llama.meta.com/llama3/
- 아직은 한국어 파인튜닝 모델 X

## STEP 1. gguf 모델 다운로드
- 모델 링크: https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/tree/main

## STEP 2. Modelfile

```bash
FROM Meta-Llama-3-8B-Instruct.Q8_0.gguf

TEMPLATE """{{- if .System }}
<|begin_of_text|>system {{ .System }}<|end_of_text|>
{{- end }}
<|begin_of_text|>user
{{ .Prompt }}<|end_of_text|>
<|begin_of_text|>assistant
"""

SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop <|begin_of_text|>
PARAMETER stop <|end_of_text|>
PARAMETER stop <|eot_id|>
PARAMETER stop <|end_of_text|>
```

## STEP 3. Ollama 모델 생성

```bash
ollama create llama3-instruct-8b -f Modelfile
```

## STEP 4. 잘 만들어 졌는지 확인

```bash
ollama list
```
출력되는 결과에 내가 만든 모델이 뜨는지 확인해 주세요.

## STEP 5. Ollama 실행

```bash
ollama run llama3-instruct-8b
```