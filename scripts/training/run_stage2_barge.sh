#!/usr/bin/env bash
set -euo pipefail
: "${MANIFEST:?set MANIFEST}" "${LLM:?set LLM}" "${ENCODER_DIM:?set ENCODER_DIM}" "${STAGE1_ADAPTER:?set STAGE1_ADAPTER}"
STACK_FACTOR="${STACK_FACTOR:-2}"; RUN_DIR="${RUN_DIR:-runs/stage2}"
python scripts/training/train_stage.py --manifest "$MANIFEST" --llm "$LLM" --encoder-dim "$ENCODER_DIM" --stack-factor "$STACK_FACTOR" --stage lora --tasks BARGE --init-adapter "$STAGE1_ADAPTER" --epochs "${EPOCHS:-2}" --batch-size "${BATCH_SIZE:-2}" --lr "${LR:-5e-5}" --lora-lr "${LORA_LR:-2e-5}" --precision "${PRECISION:-fp16}" --output-dir "$RUN_DIR"
