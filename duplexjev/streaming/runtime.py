"""State-routed query runtime for streaming DuplexJev decisions."""
from enum import Enum
from typing import Callable, Any
from .context import StreamingContext

class RuntimeState(str, Enum):
    IDLE = "IDLE"
    LISTEN = "LISTEN"
    SPEAK = "SPEAK"
    DUAL = "DUAL"
    THINK = "THINK"

class QueryKind(str, Enum):
    BARGE = "QUERY_BARGE"
    TURN_END = "QUERY_TURN_END"

class DuplexStreamingRuntime:
    """Route readout queries from authoritative Voice Runtime state.

    decision_fn(kind, audio_prefix, context_snapshot, state_version) returns
    a closed-set decision dict. The callback is the seam where a model-specific
    cached LLM readout is connected; the one-shot Decider API is not misused as
    if it already supported streaming KV cache.
    """
    def __init__(self, decision_fn: Callable[..., dict], *, barge_threshold: float = 0.75,
                 turn_threshold: float = 0.75):
        self.decision_fn = decision_fn
        self.barge_threshold = float(barge_threshold)
        self.turn_threshold = float(turn_threshold)
        self.context = StreamingContext()
        self.last_result: dict[str, Any] | None = None

    def set_state(self, state: RuntimeState | str) -> int:
        return self.context.set_state(RuntimeState(state).value)

    def commit_ai_playback(self, text: str, *, playback_offset_ms: int) -> int:
        return self.context.commit_played_text(text, playback_offset_ms=playback_offset_ms)

    def append_audio_event(self, audio_embedding_chunk: Any, *, start_ms: int, end_ms: int) -> int:
        return self.context.append_user_audio(audio_embedding_chunk, start_ms=start_ms, end_ms=end_ms)

    def query(self, audio_prefix: Any) -> dict[str, Any] | None:
        state = self.context.runtime_state
        if state == RuntimeState.DUAL.value:
            kind = QueryKind.BARGE
        elif state == RuntimeState.LISTEN.value:
            kind = QueryKind.TURN_END
        else:
            return None
        version = self.context.state_version
        result = self.decision_fn(kind.value, audio_prefix, self.context.snapshot(), version)
        if version != self.context.state_version:
            return None
        result = dict(result)
        result.update(query_kind=kind.value, state_version=version)
        self.last_result = result
        return result

    def should_interrupt(self, result: dict[str, Any]) -> bool:
        return result.get("query_kind") == QueryKind.BARGE.value and result.get("answer") == "BARGE_IN" and float(result.get("confidence", 0)) >= self.barge_threshold

    def should_respond(self, result: dict[str, Any]) -> bool:
        return result.get("query_kind") == QueryKind.TURN_END.value and result.get("answer") == "TURN_END" and float(result.get("confidence", 0)) >= self.turn_threshold
