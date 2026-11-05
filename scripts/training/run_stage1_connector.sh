#!/usr/bin/env bash
set -euo pipefail
: "${MANIFEST:?set MANIFEST}" "${LLM:?set LLM}" "${ENCODER_DIM:?set ENCODER_DIM}"
STACK_FACTOR="${STACK_FACTOR:-2}"; LM_FACTORY_ARGS=(); if [[ -n "${LM_FACTORY:-}" ]]; then LM_FACTORY_ARGS+=(--lm-factory "$LM_FACTORY"); fi; RUN_DIR="${RUN_DIR:-runs/stage1}"
python scripts/training/train_stage.py "${LM_FACTORY_ARGS[@]}" --manifest "$MANIFEST" --llm "$LLM" --encoder-dim "$ENCODER_DIM" --stack-factor "$STACK_FACTOR" --stage connector --tasks BARGE,TURN_END --epochs "${EPOCHS:-3}" --batch-size "${BATCH_SIZE:-2}" --lr "${LR:-1e-4}" --precision "${PRECISION:-fp16}" --output-dir "$RUN_DIR"
