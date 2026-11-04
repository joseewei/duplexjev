"""Append-only Ultravox/causal-LM session with isolated typed-query branches."""
from __future__ import annotations
import copy
import threading
from dataclasses import dataclass
from typing import Any, Optional
import torch
from ..question import Question

@dataclass
class StreamingLMState:
    cache: Any
    sequence_length: int = 0
    state_version: int = 0
    closed: bool = False

class StreamingUltravoxSession:
    """One-call streaming context backed by the Ultravox language model.

    Consumes already-projected audio embeddings (WeNet -> trained projector) and
    tokenizer-stable text. The caller supplies static prompt text at creation.
    Shared-prefix appends mutate only state.cache. Query inference deep-copies
    that cache, so query tokens cannot leak into later decisions.
    """
    def __init__(self, model, tokenizer, *, prefix_text: str = "",
                 device: Optional[object] = None):
        from transformers.cache_utils import DynamicCache
        self.model = model.eval()
        self.lm = model.language_model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else self.lm.get_input_embeddings().weight.device
        self._lock = threading.RLock()
        self.state = StreamingLMState(cache=DynamicCache())
        self._letter_ids: dict[str, int] = {}
        if prefix_text:
            self.append_text(prefix_text)

    @property
    def cache_length(self) -> int:
        return self.state.sequence_length

    def _token_ids(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        return ids.to(self.device)

    def _forward_append(self, embeds: torch.Tensor) -> None:
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        if embeds.ndim != 3 or embeds.size(0) != 1:
            raise ValueError("stream append expects [T,D] or [1,T,D]")
        if embeds.size(1) == 0:
            return
        embeds = embeds.to(device=self.device, dtype=self.lm.get_input_embeddings().weight.dtype)
        past = self.state.sequence_length
        total = past + embeds.size(1)
        attention_mask = torch.ones((1, total), dtype=torch.long, device=self.device)
        cache_position = torch.arange(past, total, dtype=torch.long, device=self.device)
        self.lm(inputs_embeds=embeds, attention_mask=attention_mask,
                past_key_values=self.state.cache, cache_position=cache_position,
                use_cache=True, return_dict=True)
        self.state.sequence_length = total

    @torch.inference_mode()
    def append_text(self, text: str) -> int:
        """Append stable text tokens. Caller must enforce played-text-only policy."""
        if not text:
            return self.cache_length
        with self._lock:
            if self.state.closed:
                raise RuntimeError("session is closed")
            ids = self._token_ids(text)
            self._forward_append(self.lm.get_input_embeddings()(ids))
            return self.cache_length

    @torch.inference_mode()
    def append_audio_embeddings(self, embeddings: torch.Tensor) -> int:
        """Append newly emitted WeNet+projector embeddings exactly once."""
        with self._lock:
            if self.state.closed:
                raise RuntimeError("session is closed")
            self._forward_append(embeddings.to(self.device))
            return self.cache_length

    def _letter_id(self, letter: str) -> int:
        if letter not in self._letter_ids:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"option {letter!r} is not one tokenizer token: {ids}")
            self._letter_ids[letter] = int(ids[0])
        return self._letter_ids[letter]

    @torch.inference_mode()
    def query(self, question: Question, *, suffix: str = "",
              expected_state_version: Optional[int] = None) -> dict[str, Any]:
        """Fork shared cache, read option-letter logits, then discard the fork."""
        with self._lock:
            if self.state.closed:
                raise RuntimeError("session is closed")
            version = self.state.state_version
            if expected_state_version is not None and version != expected_state_version:
                return {"stale": True, "state_version": version}
            base_len = self.state.sequence_length
            try:
                query_cache = copy.deepcopy(self.state.cache)
            except Exception as exc:
                raise RuntimeError("Transformers Cache is not deepcopyable; implement cache-specific fork.") from exc

        prompt = question.render(list(range(len(question.options))))
        ids = self._token_ids(prompt + suffix)
        if ids.size(1) == 0:
            raise ValueError("query prompt tokenized to an empty sequence")
        q_len = int(ids.size(1))
        q_embeds = self.lm.get_input_embeddings()(ids)
        attention_mask = torch.ones((1, base_len + q_len), dtype=torch.long, device=self.device)
        cache_position = torch.arange(base_len, base_len + q_len, dtype=torch.long, device=self.device)
        output = self.lm(inputs_embeds=q_embeds, attention_mask=attention_mask,
                         past_key_values=query_cache, cache_position=cache_position,
                         use_cache=True, return_dict=True)
        logits = output.logits[0, -1].float()
        letter_ids = torch.tensor([self._letter_id(x) for x in question.letters()],
                                  dtype=torch.long, device=logits.device)
        probs = torch.softmax(logits.index_select(0, letter_ids), dim=-1)
        with self._lock:
            if self.state.state_version != version:
                return {"stale": True, "state_version": self.state.state_version}
        best = int(probs.argmax().item())
        return {"stale": False, "state_version": version,
                "query_cache_length": base_len + q_len,
                "shared_cache_length": self.state.sequence_length,
                "options": list(question.options),
                "probabilities": probs.detach().cpu().tolist(),
                "best_index": best, "best_option": question.options[best],
                "confidence": float(probs.max().item())}

    def set_state_version(self, version: int) -> None:
        with self._lock:
            if version < self.state.state_version:
                raise ValueError("state_version cannot move backwards")
            self.state.state_version = int(version)

    def close(self) -> None:
        with self._lock:
            self.state.closed = True
            self.state.cache = None
