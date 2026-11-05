#!/usr/bin/env bash
set -euo pipefail
: "${MANIFEST:?set MANIFEST}" "${LLM:?set LLM}" "${ENCODER_DIM:?set ENCODER_DIM}" "${STAGE2_ADAPTER:?set STAGE2_ADAPTER}" "${STAGE2_LORA:?set STAGE2_LORA}"
STACK_FACTOR="${STACK_FACTOR:-2}"; RUN_DIR="${RUN_DIR:-runs/stage3}"
python scripts/training/train_stage.py --manifest "$MANIFEST" --llm "$LLM" --encoder-dim "$ENCODER_DIM" --stack-factor "$STACK_FACTOR" --stage joint --tasks BARGE,TURN_END --init-adapter "$STAGE2_ADAPTER" --init-lora "$STAGE2_LORA" --epochs "${EPOCHS:-2}" --batch-size "${BATCH_SIZE:-2}" --lr "${LR:-2e-5}" --lora-lr "${LORA_LR:-1e-5}" --precision "${PRECISION:-fp16}" --output-dir "$RUN_DIR"
