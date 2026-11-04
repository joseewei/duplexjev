import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
from transformers.cache_utils import DynamicCache

from duplexjev.question import Question
from duplexjev.streaming.ultravox_session import StreamingUltravoxSession


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = [ord(c) % 128 for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 128 for c in text]


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "llama"})()
        self.embed = nn.Embedding(128, 4)
        self.head = nn.Linear(4, 128)
    def get_input_embeddings(self):
        return self.embed
    def forward(self, inputs_embeds, attention_mask, past_key_values,
                cache_position, use_cache, return_dict):
        # Simulate a decoder-only transformer appending K/V to DynamicCache.
        k = inputs_embeds[:, None, :, :]
        v = inputs_embeds[:, None, :, :]
        past_key_values.update(k, v, 0)
        return type("Out", (), {"logits": self.head(inputs_embeds)})()


class FakeUltravox(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeLM()
        self.is_diffusion_gemma = False


def test_query_cache_is_rolled_back_and_prefix_is_append_only():
    model = FakeUltravox()
    session = StreamingUltravoxSession(model, CharTokenizer())
    session.append_audio_embeddings(torch.randn(1, 3, 4))
    prefix_len = session.cache_length
    q = Question(id="barge", text="是否打断", options=["打断", "不打断"], lang="zh")
    result = session.query(q)
    assert not result["stale"]
    assert result["shared_cache_length"] == prefix_len
    assert session.cache_length == prefix_len
    assert session.state.cache.get_seq_length() == prefix_len
    session.append_text("你好")
    assert session.cache_length > prefix_len
