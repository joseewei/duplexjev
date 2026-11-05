#!/usr/bin/env bash
set -euo pipefail
: "${ROBUST_MANIFEST:?set ROBUST_MANIFEST}" "${LLM:?set LLM}" "${ENCODER_DIM:?set ENCODER_DIM}" "${STAGE3_ADAPTER:?set STAGE3_ADAPTER}" "${STAGE3_LORA:?set STAGE3_LORA}"
STACK_FACTOR="${STACK_FACTOR:-2}"; RUN_DIR="${RUN_DIR:-runs/stage4}"
python scripts/training/train_stage.py --manifest "$ROBUST_MANIFEST" --llm "$LLM" --encoder-dim "$ENCODER_DIM" --stack-factor "$STACK_FACTOR" --stage robustness --tasks BARGE,TURN_END --init-adapter "$STAGE3_ADAPTER" --init-lora "$STAGE3_LORA" --epochs "${EPOCHS:-1}" --batch-size "${BATCH_SIZE:-2}" --lr "${LR:-1e-5}" --lora-lr "${LORA_LR:-5e-6}" --bf16 --output-dir "$RUN_DIR"
