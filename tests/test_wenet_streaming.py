import torch
from torch import nn
from duplexjev.streaming.wenet_encoder import WenetStreamingEncoder

class FakeEmbed:
    subsampling_rate = 4
    right_context = 6

class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(()))
        self.embed = FakeEmbed()
        self.calls = []
    def forward_chunk(self, xs, offset, required_cache_size, att_cache, cnn_cache):
        self.calls.append((xs.clone(), offset, required_cache_size))
        return xs[:, :9:4, :2], att_cache, cnn_cache

def test_wenet_driver_respects_right_context_and_stride():
    enc = FakeEncoder()
    driver = WenetStreamingEncoder(enc, chunk_size=3, left_chunks=2)
    state = driver.new_state()
    assert driver.input_window == 15
    assert driver.input_stride == 12
    assert driver.push_features(state, torch.randn(1, 10, 80)) == []
    outputs = driver.push_features(state, torch.randn(5, 80))
    assert len(outputs) == 1
    assert enc.calls[0][0].shape[1] == 15
    assert enc.calls[0][1:] == (0, 6)
    assert driver.pending_feature_frames(state) == 3
