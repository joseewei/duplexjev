"""Causally committed runtime context for streaming voice interaction."""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class StreamingContext:
    """Only commits AI text that has actually reached the user's playback cursor."""
    runtime_state: str = "IDLE"
    state_version: int = 0
    ai_played_text: str = ""
    user_audio_events: list[dict[str, Any]] = field(default_factory=list)
    event_sequence: int = 0
    last_playback_offset_ms: int = 0

    def set_state(self, state: str) -> int:
        if state != self.runtime_state:
            self.runtime_state = state
            self.state_version += 1
        return self.state_version

    def commit_played_text(self, text: str, *, playback_offset_ms: int) -> int:
        if playback_offset_ms < self.last_playback_offset_ms:
            raise ValueError("playback cursor must be monotonic")
        if text:
            self.last_playback_offset_ms = int(playback_offset_ms)
            self.ai_played_text += text
            self.event_sequence += 1
        return self.event_sequence

    def append_user_audio(self, payload: Any, *, start_ms: int, end_ms: int) -> int:
        if end_ms < start_ms:
            raise ValueError("audio end_ms must be >= start_ms")
        self.event_sequence += 1
        self.user_audio_events.append({"sequence_id": self.event_sequence,
            "start_ms": int(start_ms), "end_ms": int(end_ms), "payload": payload})
        return self.event_sequence

    def snapshot(self) -> dict[str, Any]:
        return {"runtime_state": self.runtime_state, "state_version": self.state_version,
            "ai_played_text": self.ai_played_text, "event_sequence": self.event_sequence,
            "last_playback_offset_ms": self.last_playback_offset_ms}
