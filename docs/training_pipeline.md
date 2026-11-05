# Streaming DuplexJev training pipeline

This pipeline assumes a trained streaming audio encoder is already available. The encoder is frozen; this work trains the connector and task decision readout, then optionally adapts the LLM with LoRA. It does not retrain or replace the audio encoder.

## 1. Data model: one recording, many decision events

The manifest is JSONL with one row per decision event. Multiple rows can reference the same complete recording and feature cache. Split assignment must be by recording_id, never by event, to prevent leakage between train/valid/test.

Example event row:

~~~json
{
  "recording_id": "call_0001",
  "event_id": "call_0001_t1250",
  "split": "train",
  "time_ms": 1250,
  "query": "BARGE",
  "label": 1,
  "feature_file": "../features/call_0001.pt",
  "segments": [
    {"type": "text", "played_at_ms": 300, "token_ids": [151644, 1234, 151645]},
    {"type": "audio", "start_frame": 0, "end_frame": 28, "available_at_ms": 1200}
  ]
}
~~~

Query values are BARGE or TURN_END. Label 1 means positive (BARGE_IN or TURN_END); label 0 means negative (NO_BARGE_IN or TURN_CONTINUE). The training readout orders answer tokens as B then A: B is positive, A is negative. At inference the options remain ordered positive then negative, and the session reads answer_letters=(B,A).

Segments are in the exact append order observed by Runtime. Text segments must already be tokenized using the target LLM's exact chat template and must represent text confirmed played by the TTS playback cursor. Audio segments refer to cached streaming-encoder output frames. Audio segment boundaries must be connector emission boundaries, and frame count must be divisible by stack_factor. available_at_ms and played_at_ms are checked against time_ms to reject future leakage.

A feature cache is a PyTorch file containing a dictionary with a features tensor shaped [frames, encoder_dim]. Do not create it with a full-utterance bidirectional encoder pass. Call the trained streaming encoder chunk by chunk and concatenate emitted frame outputs in order. Preserve the production frontend, CMVN, sample rate, chunk size, right context and cache policy.

## 2. Data preparation

First create audio_list.jsonl with recording_id and audio_path per recording. Implement an adapter factory module:function(config_dict) that returns an object with encode_file(audio_path, chunk_ms) -> Tensor[T,D]. The adapter must run your real trained streaming encoder and exact frontend.

~~~bash
pip install -e '.[speech]'

python scripts/training/cache_streaming_features.py \
  --audio-list data/audio_list.jsonl \
  --adapter my_project.duplexjev_adapter:build \
  --config '{"checkpoint":"/models/streaming_encoder"}' \
  --out-dir data/features --chunk-ms 160
~~~

Prepare annotations JSONL with one event per row. Each event contains recording_id, event_id, time_ms, query, label and ordered segments. Then join the cached feature index and assign splits by recording:

~~~bash
python scripts/training/build_manifest.py \
  --annotations data/annotations/events.jsonl \
  --feature-index data/features/index.json \
  --out data/manifest/events.jsonl \
  --train-ratio 0.90 --valid-ratio 0.05 --seed 42

python scripts/training/prepare_manifest.py \
  --manifest data/manifest/events.jsonl --out data/manifest/report.json
~~~

If annotations already contain split values, build_manifest preserves them. Ensure every split has enough events from both tasks and both classes.

## 3. Training stages

Stage 1 — connector warm-up. Freeze the LLM and train the MLP connector plus two learned query vectors using BARGE/TURN_END task labels. This is supervised decision training, not an assumed feature-alignment loss.

~~~bash
python scripts/training/train_stage.py --manifest data/manifest/events.jsonl \
  --llm /models/base-llm --encoder-dim ACTUAL_DIM --stack-factor 2 \
  --stage connector --epochs 3 --batch-size 2 --lr 1e-4 --precision fp16 \
  --output-dir runs/stage1
~~~

Stage 2 — task adaptation. Initialize from Stage 1; train connector, query vectors and rank-16 LLM LoRA. Use a lower LR for LoRA than for the connector.

~~~bash
python scripts/training/train_stage.py --manifest data/manifest/events.jsonl \
  --llm /models/base-llm --encoder-dim ACTUAL_DIM --stack-factor 2 \
  --stage lora --tasks BARGE --init-adapter runs/stage1/best/streaming_adapter.pt \
  --epochs 2 --batch-size 2 --lr 5e-5 --lora-lr 2e-5 --precision fp16 \
  --output-dir runs/stage2
~~~

Stage 3 — joint multi-task refinement. Resume both the Stage 2 connector/query vectors and its LoRA adapter, and train BARGE and TURN_END together.

~~~bash
python scripts/training/train_stage.py --manifest data/manifest/events.jsonl \
  --llm /models/base-llm --encoder-dim ACTUAL_DIM --stack-factor 2 \
  --stage joint --tasks BARGE,TURN_END --init-adapter runs/stage2/best/streaming_adapter.pt \
  --init-lora runs/stage2/best/lora \
  --epochs 2 --batch-size 2 --lr 2e-5 --lora-lr 1e-5 --precision fp16 \
  --output-dir runs/stage3
~~~

Stage 4 — streaming robustness. Build a separate manifest with additional real decision timestamps, chunk-boundary cases, short pauses, long pauses, false barge-ins, background speakers, backchannels, and examples where TTS text is queued but not yet played. Preserve the true event labels and event-time visibility constraints. Continue from Stage 3 using --stage robustness, --init-adapter and --init-lora. Do not simulate future audio/text by simply truncating a prefix while keeping a label that may no longer be valid.

~~~bash
python scripts/training/train_stage.py --manifest data/manifest/robustness_events.jsonl \
  --llm /models/base-llm --encoder-dim ACTUAL_DIM --stack-factor 2 \
  --stage robustness --tasks BARGE,TURN_END --init-adapter runs/stage3/best/streaming_adapter.pt \
  --init-lora runs/stage3/best/lora --epochs 1 --batch-size 2 \
  --lr 1e-5 --lora-lr 5e-6 --precision fp16 --output-dir runs/stage4
~~~

The trainer saves connector/query weights in streaming_adapter.pt and LoRA weights in the lora subdirectory. Best and last checkpoints are saved separately. Validation currently reports per-task accuracy; before production, add per-class precision/recall/F1, false interruption rate, missed interruption rate, and latency.

## 4. Runtime loading

Load the adapter checkpoint and apply the connector state to the same projector architecture used by StreamingConnector. Load query vectors into StreamingUltravoxSession:

~~~python
from duplexjev.streaming import load_streaming_adapter
load_streaming_adapter(session, projector,
    "runs/stage4/best/streaming_adapter.pt",
    lora_path="runs/stage4/best/lora")
~~~

Then bind the runtime with positive-first options:

~~~python
barge_question = Question(id="barge", text="barge-in", options=["BARGE_IN", "NO_BARGE_IN"], lang="en")
turn_question = Question(id="turn", text="turn-end", options=["TURN_END", "TURN_CONTINUE"], lang="en")
decision_fn = make_duplexjev_decision_fn(session, barge_question=barge_question,
                                         turn_question=turn_question)
~~~

With trained query vectors, the runtime uses one learned query embedding per decision and reads the B/A answer-token distribution. The natural-language question body is not recomputed at each query.

## 5. Model/checkpoint compatibility gates

- The example dimension in any config is not a default. Read encoder_dim from the trained streaming encoder checkpoint.
- The trainer currently loads a causal LM through AutoModelForCausalLM. Verify that the exact serving language model accepts inputs_embeds, attention_mask, position_ids, output_hidden_states and returns hidden_states[-1]. For a custom Ultravox/VL checkpoint, adapt the loader to use the exact model.language_model.
- The query vector is trained against the exact base LLM, tokenizer, text prompt tokens, and audio connector layout used by serving. Changing the base LLM or chat-template tokens invalidates the learned connector/query vectors.
- Audio and played-text segments must preserve the actual append order. Audio frames are included only once they have been emitted by the streaming encoder/connector; AI text is included only after playback confirmation.
- This implementation uses event-level examples that reference complete recording feature caches. Do not randomly split events from the same recording across datasets.
- No claim of end-to-end GPU training success is made until smoke tests run against the actual checkpoint, adapter, tokenizer and target hardware.

For NVIDIA V100, use FP16 (the stage launchers default to fp16 and enable GradScaler). Use BF16 only on hardware that supports it; use fp32 for a numerical baseline.
