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
                 turn_threshold: float = 0.75, on_ai_played_text=None, on_audio_embeddings=None):
        self.decision_fn = decision_fn
        self.on_ai_played_text = on_ai_played_text or getattr(decision_fn, "on_ai_played_text", None)
        self.on_audio_embeddings = on_audio_embeddings or getattr(decision_fn, "on_audio_embeddings", None)
        self.barge_threshold = float(barge_threshold)
        self.turn_threshold = float(turn_threshold)
        self.context = StreamingContext()
        self.last_result: dict[str, Any] | None = None

    def set_state(self, state: RuntimeState | str) -> int:
        return self.context.set_state(RuntimeState(state).value)

    def commit_ai_playback(self, text: str, *, playback_offset_ms: int) -> int:
        seq = self.context.commit_played_text(text, playback_offset_ms=playback_offset_ms)
        if text and self.on_ai_played_text is not None:
            self.on_ai_played_text(text)
        return seq

    def append_audio_event(self, audio_embedding_chunk: Any, *, start_ms: int, end_ms: int) -> int:
        seq = self.context.append_user_audio(audio_embedding_chunk, start_ms=start_ms, end_ms=end_ms)
        if self.on_audio_embeddings is not None:
            self.on_audio_embeddings(audio_embedding_chunk)
        return seq

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

def make_duplexjev_decision_fn(session, *, barge_question, turn_question, suffix: str = ""):
    """Bind a StreamingUltravoxSession to the Runtime's decision_fn contract.

    Configure barge_question options exactly as ("BARGE_IN", "NO_BARGE_IN")
    and turn_question options as ("TURN_END", "TURN_CONTINUE"). The audio prefix
    is already appended to the session incrementally and is intentionally not
    replayed in this callback.
    """
    expected_barge = ("BARGE_IN", "NO_BARGE_IN")
    expected_turn = ("TURN_END", "TURN_CONTINUE")
    if tuple(barge_question.options) != expected_barge:
        raise ValueError(f"barge_question.options must be {expected_barge}")
    if tuple(turn_question.options) != expected_turn:
        raise ValueError(f"turn_question.options must be {expected_turn}")

    def decide(query_kind, audio_prefix, context_snapshot, state_version):
        from .ultravox_session import StreamingUltravoxSession

        session.set_state_version(state_version)
        question = barge_question if query_kind == QueryKind.BARGE.value else turn_question
        result = session.query(question, suffix=suffix, expected_state_version=state_version)
        if result.get("stale"):
            return {"answer": "STALE", "confidence": 0.0}
        return {"answer": result["best_option"], "confidence": result["confidence"],
                "probabilities": result["probabilities"]}

    decide.on_ai_played_text = session.append_text
    decide.on_audio_embeddings = session.append_audio_embeddings
    return decide
