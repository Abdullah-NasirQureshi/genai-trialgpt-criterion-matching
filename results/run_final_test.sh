#!/bin/bash
# FINAL test-set evaluation: run exactly once, after the adapter was chosen on validation.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
.venv/Scripts/python.exe -I src/eval_model.py --split test --allow-test --tag base_zeroshot
.venv/Scripts/python.exe -I src/eval_model.py --split test --allow-test --adapter results/lora_final/adapter --tag lora_final
echo FINALDONE
