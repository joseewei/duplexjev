"""Append-only frame-stacking connector for WeNet encoder outputs."""
import torch

class StreamingConnector:
    """Apply a checkpoint-compatible projector to stacked encoder frames.

    The projector maps [B,T,stack_factor*encoder_dim] to [B,T,llm_dim].
    Incomplete frame groups are retained across chunks, never dropped at a
    chunk boundary.
    """
    def __init__(self, projector: torch.nn.Module, *, stack_factor: int = 2):
        if stack_factor <= 0:
            raise ValueError("stack_factor must be > 0")
        self.projector = projector
        self.stack_factor = int(stack_factor)

    def new_state(self):
        return {"pending": None, "frames_seen": 0, "tokens_emitted": 0}

    def push(self, state, encoder_output: torch.Tensor) -> torch.Tensor:
        if encoder_output.ndim != 3 or encoder_output.size(0) != 1:
            raise ValueError("encoder_output must have shape [1,T,D]")
        x = encoder_output
        if state["pending"] is not None:
            x = torch.cat((state["pending"], x), dim=1)
        state["frames_seen"] += int(encoder_output.size(1))
        n_full = (x.size(1) // self.stack_factor) * self.stack_factor
        if n_full == 0:
            state["pending"] = x
            return x.new_empty((1, 0, 0))
        ready, tail = x[:, :n_full], x[:, n_full:]
        b, t, d = ready.shape
        stacked = ready.reshape(b, t // self.stack_factor, d * self.stack_factor)
        projected = self.projector(stacked)
        if isinstance(projected, tuple):
            projected = projected[0]
        if projected.ndim != 3:
            raise RuntimeError("projector must return [B,T,D] embeddings")
        state["pending"] = tail if tail.size(1) else None
        state["tokens_emitted"] += int(projected.size(1))
        return projected

    def flush(self, state, *, pad: bool = False):
        """Optionally pad the final incomplete stack; default preserves exact frames."""
        pending = state["pending"]
        if pending is None or pending.size(1) == 0 or not pad:
            return None
        missing = self.stack_factor - pending.size(1)
        padded = torch.cat((pending, pending.new_zeros((1, missing, pending.size(-1)))), dim=1)
        out = self.projector(padded.reshape(1, 1, -1))
        if isinstance(out, tuple):
            out = out[0]
        state["pending"] = None
        state["tokens_emitted"] += int(out.size(1))
        return out
