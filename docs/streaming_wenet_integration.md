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
