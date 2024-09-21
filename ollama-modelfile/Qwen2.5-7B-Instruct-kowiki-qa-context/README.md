# Qwen2.5-7B-Instruct-kowiki-qa-context(2024.09.22)

- https://huggingface.co/beomi/Qwen2.5-7B-Instruct-kowiki-qa
- Qwen2.5 7B 베이스 한국어 파인튜닝 모델

## STEP 1. gguf 모델 다운로드
- 모델 링크: https://huggingface.co/teddylee777/Qwen2.5-7B-Instruct-kowiki-qa-context-gguf

## STEP 2. Modelfile

```bash
FROM Qwen2.5-7B-Instruct-kowiki-qa-context-Q8_0.gguf

TEMPLATE """{{- if .System }}
<|im_start|>system
{{ .System }}
<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}
<|im_end|>
<|im_start|>assistant
"""

SYSTEM """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. 모든 대답은 한국어로 해주세요."""

PARAMETER temperature 0
PARAMETER num_ctx 128000
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
```

## STEP 3. Ollama 모델 생성

```bash
ollama create qwen2.5-7b-instruct-kowiki -f Modelfile
```

## STEP 4. 잘 만들어 졌는지 확인

```bash
ollama list
```
출력되는 결과에 내가 만든 모델이 뜨는지 확인해 주세요.

## STEP 5. Ollama 실행

```bash
ollama run qwen2.5-7b-instruct-kowiki
```