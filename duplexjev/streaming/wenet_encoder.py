"""Chunked inference adapter for the official WeNet TransformerEncoder.

Consumes Fbank feature frames, not waveforms. Feature extraction (sample rate,
mel bins and CMVN) must match the selected WeNet checkpoint. Each stream owns
independent WeNet attention and CNN caches.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class WenetEncoderState:
    feature_buffer: torch.Tensor
    att_cache: torch.Tensor
    cnn_cache: torch.Tensor
    encoder_offset: int = 0
    input_frames_seen: int = 0
    output_frames_emitted: int = 0

class WenetStreamingEncoder:
    """Driver around a configured WeNet BaseEncoder/TransformerEncoder.

    Chunk size is measured in encoder output frames. This is bounded-lookahead
    streaming: the complete WeNet right-context window must arrive first.
    """
    def __init__(self, encoder, *, chunk_size: int, left_chunks: int = -1,
                 device: Optional[object] = None):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not hasattr(encoder, "forward_chunk") or not hasattr(encoder, "embed"):
            raise TypeError("encoder must expose WeNet BaseEncoder.forward_chunk and embed")
        self.encoder = encoder.eval()
        self.chunk_size = int(chunk_size)
        self.left_chunks = int(left_chunks)
        self.device = torch.device(device) if device is not None else next(encoder.parameters()).device
        self.subsampling_rate = int(encoder.embed.subsampling_rate)
        self.right_context = int(encoder.embed.right_context)
        self.input_window = (self.chunk_size - 1) * self.subsampling_rate + self.right_context + 1
        self.input_stride = self.chunk_size * self.subsampling_rate
        self.required_cache_size = -1 if self.left_chunks < 0 else self.chunk_size * self.left_chunks

    def new_state(self, *, feature_dim: int = 80) -> WenetEncoderState:
        p = next(self.encoder.parameters())
        empty = torch.empty((0, feature_dim), device=self.device, dtype=p.dtype)
        cache = torch.zeros((0, 0, 0, 0), device=self.device, dtype=p.dtype)
        return WenetEncoderState(empty, cache, cache.clone())

    @torch.inference_mode()
    def push_features(self, state: WenetEncoderState, features: torch.Tensor) -> list[torch.Tensor]:
        if features.ndim == 3:
            if features.size(0) != 1:
                raise ValueError("one WeNet streaming session must have batch size 1")
            features = features[0]
        if features.ndim != 2:
            raise ValueError(f"features must be [T,F], got {tuple(features.shape)}")
        if features.size(0) == 0:
            return []
        p = next(self.encoder.parameters())
        features = features.to(device=self.device, dtype=p.dtype)
        if state.feature_buffer.size(-1) != features.size(-1):
            if state.input_frames_seen:
                raise ValueError("Fbank dimension changed within a stream")
            state.feature_buffer = features[:0]
        state.feature_buffer = torch.cat((state.feature_buffer, features), dim=0)
        state.input_frames_seen += int(features.size(0))
        outputs = []
        while state.feature_buffer.size(0) >= self.input_window:
            window = state.feature_buffer[:self.input_window].unsqueeze(0)
            y, state.att_cache, state.cnn_cache = self.encoder.forward_chunk(
                window, offset=state.encoder_offset,
                required_cache_size=self.required_cache_size,
                att_cache=state.att_cache, cnn_cache=state.cnn_cache)
            if y.ndim != 3 or y.size(0) != 1:
                raise RuntimeError(f"unexpected WeNet output shape: {tuple(y.shape)}")
            outputs.append(y)
            state.encoder_offset += int(y.size(1))
            state.output_frames_emitted += int(y.size(1))
            state.feature_buffer = state.feature_buffer[self.input_stride:]
        return outputs

    def pending_feature_frames(self, state: WenetEncoderState) -> int:
        return int(state.feature_buffer.size(0))
