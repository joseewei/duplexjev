# WeNet streaming runtime integration

This branch adds the first independently testable streaming layer around the
existing DuplexJev package. It does **not** silently replace the released
Qwen3-ASR checkpoint or claim that the current one-shot `Decider` is already
an incremental LLM implementation.

## Selected architecture

```
8 kHz/16 kHz call audio
  -> checkpoint-matched Fbank + CMVN (integration supplied by deployment)
  -> WeNet TransformerEncoder.forward_chunk
       per-session att_cache / cnn_cache / offset / Fbank overlap
  -> StreamingConnector (frame stack across chunk boundaries)
  -> append-only audio embedding prefix
  -> DuplexJev LLM + typed single-token readout
       LISTEN -> QUERY_TURN_END
       DUAL   -> QUERY_BARGE
  -> runtime threshold / state_version validation
```

## Files

- `duplexjev/streaming/wenet_encoder.py`: WeNet `BaseEncoder.forward_chunk`
  driver. Uses the encoder's actual `embed.subsampling_rate` and
  `embed.right_context`; does not hardcode 4x subsampling.
- `duplexjev/streaming/connector.py`: carries incomplete frame groups across
  encoder chunks. Supply the trained checkpoint-compatible projector.
- `duplexjev/streaming/context.py`: only commits assistant text after the TTS
  playback cursor confirms it has been played; append-only audio event log.
- `duplexjev/streaming/runtime.py`: routes query type from authoritative
  runtime state and discards stale results after a state transition.
- `tests/test_streaming_runtime.py`, `tests/test_wenet_streaming.py`.

## Important model integration boundary

The package's existing `Decider` currently builds an audio prompt through its
Ultravox processor and calls `self.model(...)` for a one-shot audio forward.
Its packed mode prefills and reads a temporary suffix; it does not expose a
stable streaming audio-embedding KV API. Therefore this branch intentionally
exposes `DuplexStreamingRuntime.decision_fn` as an explicit seam rather than
pretending that appending embeddings to the one-shot `Decider` is correct.

The next model-specific integration must add a separate streaming session API
to the Ultravox model:
1. Encode WeNet chunks and project each complete stack exactly once.
2. Append projected audio embeddings to a dedicated audio-prefix KV cache.
3. Append only stable tokenizer token boundaries from assistant text actually
   played. Keep this text KV segment separate from audio KV.
4. For each query, append the query suffix to a temporary branch/cache, obtain
   the restricted option-letter distribution, then discard that suffix cache.
5. Never append query tokens to the shared prefix. Never mutate committed
   prefix KVs. Keep per-session caches isolated.
6. Capture `state_version` before inference and validate it again before
   dispatching a barge/response action.

The exact KV API depends on the selected Transformers/LLM architecture and
attention backend. It must be implemented and tested against the actual
checkpoint, not guessed from the generic `past_key_values` interface.

## WeNet checkpoint requirements

`WenetStreamingEncoder` accepts Fbank tensors [T,F], not waveform samples.
Instantiate WeNet's TransformerEncoder from the exact model config/checkpoint.
The feature extractor, sample rate, mel dimension, CMVN, encoder dimension,
chunk size and left-context policy must come from that same checkpoint recipe.
The driver derives subsampling rate and right context from
`encoder.embed`. Each `push_features` call can emit zero or more chunks.

The WeNet `forward_chunk` contract requires the input window length
`(chunk_size - 1) * subsampling_rate + right_context + 1`; the next window
starts `chunk_size * subsampling_rate` feature frames later. This overlap is
needed because WeNet does not cache its subsampling convolution. The encoder
itself is bounded-lookahead, not strict zero-lookahead causal.

## Training data contract

Keep each complete call as one recording/sample. Attach multiple query events
to that recording. At each event timestamp reconstruct only the prefix visible
at that time:
- current user audio prefix;
- assistant text confirmed played by the playback cursor;
- runtime state and state version;
- query kind selected from state (DUAL/BARGE, LISTEN/TURN_END);
- label and confidence/ignore mask.

Never expose future user audio, future assistant generation, TTS-enqueued but
unplayed text, or future transcript tokens. Train with a causal/chunk-aware
WeNet configuration matching online chunk size and right-context settings.
A full-utterance WeNet forward is not automatically equivalent to repeated
`forward_chunk` calls.

## Validation status

The tests in this branch exercise state routing, playback-cursor monotonicity,
frame stacking across chunk boundaries, and WeNet chunk-window scheduling
using a fake encoder. They do not certify a real WeNet checkpoint, actual
DuplexJev projector compatibility, LLM cache correctness, model accuracy,
streaming latency, or end-to-end inference. Those require the target checkpoint
and GPU runtime.


## Incremental Ultravox LLM session

The second-stage implementation adds `duplexjev/streaming/ultravox_session.py`.

```python
from duplexjev import Decider
from duplexjev.question import Question
from duplexjev.streaming import (
    DuplexStreamingRuntime, RuntimeState, make_duplexjev_decision_fn,
)

decider = Decider.from_pretrained("YOUR_CHECKPOINT", device="cuda")
session = decider.create_streaming_session(prefix_text=STATIC_PROMPT_PREFIX)

barge_q = Question(
    id="barge", lang="zh", text="用户当前是否在打断AI播报？",
    options=["BARGE_IN", "NO_BARGE_IN"],
)
turn_q = Question(
    id="turn", lang="zh", text="用户当前这轮话是否已经表达完整？",
    options=["TURN_END", "TURN_CONTINUE"],
)
decision_fn = make_duplexjev_decision_fn(
    session, barge_question=barge_q, turn_question=turn_q,
    suffix=MODEL_CHAT_TEMPLATE_SUFFIX,
    # Optional: return token IDs including assistant role/turn markers.
    played_text_encoder=render_played_assistant_span,
)
runtime = DuplexStreamingRuntime(decision_fn)

runtime.set_state(RuntimeState.DUAL)
# For every newly emitted WeNet+trained-projector embedding chunk:
runtime.append_audio_event(audio_embedding_chunk, start_ms=t0, end_ms=t1)
# Only when the playback cursor confirms text has been heard:
runtime.commit_ai_playback(played_text, playback_offset_ms=played_until_ms)
result = runtime.query(audio_prefix=None)
```

The session's language-model cache is a shared append-only prefix. Audio
embeddings and played text are appended by the runtime callbacks. A typed query
runs against a temporary suffix: for Transformers `DynamicCache`, the session
appends the query tokens under a lock and calls `crop(prefix_length)` in a
`finally` block. This avoids copying the entire long KV prefix on every
decision tick. If a cache implementation lacks `crop()`, a deepcopy fallback
is used. The cache length is checked by the session's own committed-token count.

Important constraints:
- This API consumes **projected embeddings**, not raw WeNet hidden states. The
  projector must be the one trained with this exact LLM and its expected stack
  factor/dimension.
- `prefix_text` and `suffix` must reproduce the model's trained chat template.
  They are not guessed automatically; use the same system/user/assistant prompt
  conventions as the checkpoint.
- Append only stable text spans that have actually played. Prefer
  `append_text_tokens()` with token IDs rendered using the checkpoint's exact
  chat template (including assistant role/turn markers); use `played_text_encoder`
  in the runtime adapter for this. Do not retokenize committed text or append
  generated-but-unplayed TTS text.
- Audio and played-text events must be serialized by the Voice Runtime in their
  actual event-time order before they are appended to the shared causal prefix.
  Do not append late-arriving events out of order.
- The session serializes append/query operations with a lock. This is correct
  but means a long query temporarily blocks appends; measure this under the
  target GPU load and consider a cache-native fork implementation if concurrent
  appends become a latency bottleneck.
- DiffusionGemma and recurrent/SSM hybrid LMs are rejected until their cache
  semantics are implemented and validated.
- Query options must be single-token letters under the checkpoint tokenizer.
  The readout returns a softmax restricted to those option-token logits, exactly
  like the existing Jev-style closed-set readout.

The session is an LLM-side streaming primitive. It does not by itself make the
WeNet-to-projector path checkpoint-compatible; that requires training or
adapting the projector for WeNet's encoder dimension and output distribution.
The existing DuplexJev projector was trained for its original audio encoder and
must not be connected to WeNet merely because tensor dimensions happen to match.

## Second-stage validation status

Added `tests/test_streaming_ultravox_session.py` to verify that query suffix KV
is rolled back and that subsequent prefix appends remain possible. It uses a
small fake decoder with Transformers DynamicCache. Run the test suite in the
project's speech/dev environment. No target checkpoint/GPU end-to-end benchmark
has been run in this environment.
