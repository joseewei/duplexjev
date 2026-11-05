#!/usr/bin/env bash
set -euo pipefail
: "${ROBUST_MANIFEST:?set ROBUST_MANIFEST}" "${LLM:?set LLM}" "${ENCODER_DIM:?set ENCODER_DIM}" "${STAGE3_ADAPTER:?set STAGE3_ADAPTER}" "${STAGE3_LORA:?set STAGE3_LORA}"
STACK_FACTOR="${STACK_FACTOR:-2}"; LM_FACTORY_ARGS=(); if [[ -n "${LM_FACTORY:-}" ]]; then LM_FACTORY_ARGS+=(--lm-factory "$LM_FACTORY"); fi; RUN_DIR="${RUN_DIR:-runs/stage4}"
python scripts/training/train_stage.py "${LM_FACTORY_ARGS[@]}" --manifest "$ROBUST_MANIFEST" --llm "$LLM" --encoder-dim "$ENCODER_DIM" --stack-factor "$STACK_FACTOR" --stage robustness --tasks BARGE,TURN_END --init-adapter "$STAGE3_ADAPTER" --init-lora "$STAGE3_LORA" --epochs "${EPOCHS:-1}" --batch-size "${BATCH_SIZE:-2}" --lr "${LR:-1e-5}" --lora-lr "${LORA_LR:-5e-6}" --precision "${PRECISION:-fp16}" --output-dir "$RUN_DIR"
