#!/bin/bash
# Resumable download of the two Qwen base models with curl into <repo>/models/
# (the Hugging Face python downloader stalled on a slow connection; curl -C - resumes after interruptions).
# Run from anywhere:  bash results/dl_curl.sh        (on Windows use Git Bash)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for m in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct; do
  mkdir -p "$ROOT/models/$m"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do
    out="$ROOT/models/$m/$f"
    until curl -s -L -C - --retry 5 --retry-delay 3 --connect-timeout 20 --speed-limit 1000 --speed-time 60 \
          -o "$out" "https://huggingface.co/Qwen/$m/resolve/main/$f"; do sleep 3; done
    echo "got models/$m/$f $(stat -c %s "$out")"
  done
done
echo ALLDONE
