# Training stage launchers

All launchers are executed from the repository root. Set environment variables before running them.

Common:
~~~bash
export MANIFEST=data/manifest/events.jsonl
export LLM=/models/the-exact-serving-base-llm
export ENCODER_DIM=YOUR_CHECKPOINT_HIDDEN_SIZE
export STACK_FACTOR=2
~~~

Stage 1, train connector and learned Query vectors with the LLM frozen:
~~~bash
MANIFEST=$MANIFEST LLM=$LLM ENCODER_DIM=$ENCODER_DIM bash scripts/training/run_stage1_connector.sh
~~~

Stage 2, adapt BARGE with LoRA, initialized from Stage 1:
~~~bash
export STAGE1_ADAPTER=runs/stage1/best/streaming_adapter.pt
MANIFEST=$MANIFEST LLM=$LLM ENCODER_DIM=$ENCODER_DIM STAGE1_ADAPTER=$STAGE1_ADAPTER bash scripts/training/run_stage2_barge.sh
~~~

Stage 3, jointly refine BARGE and TURN_END, resuming both connector/query and LoRA:
~~~bash
export STAGE2_ADAPTER=runs/stage2/best/streaming_adapter.pt
export STAGE2_LORA=runs/stage2/best/lora
MANIFEST=$MANIFEST LLM=$LLM ENCODER_DIM=$ENCODER_DIM STAGE2_ADAPTER=$STAGE2_ADAPTER STAGE2_LORA=$STAGE2_LORA bash scripts/training/run_stage3_joint.sh
~~~

Stage 4, robustness fine-tuning on a separately curated hard-case manifest:
~~~bash
export ROBUST_MANIFEST=data/manifest/robustness_events.jsonl
export STAGE3_ADAPTER=runs/stage3/best/streaming_adapter.pt
export STAGE3_LORA=runs/stage3/best/lora
ROBUST_MANIFEST=$ROBUST_MANIFEST LLM=$LLM ENCODER_DIM=$ENCODER_DIM STAGE3_ADAPTER=$STAGE3_ADAPTER STAGE3_LORA=$STAGE3_LORA bash scripts/training/run_stage4_robustness.sh
~~~

For serving, restore the connector/query and LoRA using duplexjev.streaming.load_streaming_adapter. The projector passed to it must be the same MLP instance architecture used by StreamingConnector. See docs/training_pipeline.md for data schema, split rules and event-time leakage constraints.
