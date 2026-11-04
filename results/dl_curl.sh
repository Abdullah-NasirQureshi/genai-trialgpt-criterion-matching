#!/bin/bash
# Resumable download of Qwen models with curl (HF python downloader stalls on this connection)
cd "$(dirname "$0")/../models"
for m in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct; do
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do
    out="$m/$f"
    until curl -s -L -C - --retry 5 --retry-delay 3 --connect-timeout 20 --speed-limit 1000 --speed-time 60 \
          -o "$out" "https://huggingface.co/Qwen/$m/resolve/main/$f"; do sleep 3; done
    echo "got $out $(stat -c %s "$out")"
  done
done
echo ALLDONE
