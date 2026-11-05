import torch
from torch import nn
from duplexjev.streaming import DuplexStreamingRuntime, QueryKind, RuntimeState, StreamingConnector, StreamingContext

def test_playback_context_only_commits_monotonic_played_text():
    c = StreamingContext()
    c.commit_played_text("您好", playback_offset_ms=120)
    assert c.ai_played_text == "您好"
    try:
        c.commit_played_text("未来文本", playback_offset_ms=100)
    except ValueError:
        pass
    else:
        raise AssertionError("non-monotonic playback cursor must fail")
    assert c.ai_played_text == "您好"

def test_runtime_routes_query_by_authoritative_state():
    seen = []
    def decide(kind, audio, context, version):
        seen.append((kind, context["ai_played_text"], version))
        return {"answer": "BARGE_IN", "confidence": 0.91}
    r = DuplexStreamingRuntime(decide)
    assert r.query(torch.zeros(1, 2, 8)) is None
    r.set_state(RuntimeState.DUAL)
    r.commit_ai_playback("我来帮您", playback_offset_ms=320)
    out = r.query(torch.zeros(1, 2, 8))
    assert out["query_kind"] == QueryKind.BARGE.value
    assert seen[-1][1] == "我来帮您"
    assert r.should_interrupt(out)
    r.set_state(RuntimeState.LISTEN)
    out = r.query(torch.zeros(1, 2, 8))
    assert out["query_kind"] == QueryKind.TURN_END.value

def test_connector_keeps_partial_stack_across_chunks():
    projector = nn.Linear(6, 4, bias=False)
    c = StreamingConnector(projector, stack_factor=2)
    s = c.new_state()
    a, b = torch.randn(1, 3, 3), torch.randn(1, 1, 3)
    y1, y2 = c.push(s, a), c.push(s, b)
    expected = projector(torch.cat([a, b], dim=1).reshape(1, 2, 6))
    assert y1.shape[1] == y2.shape[1] == 1
    assert torch.allclose(torch.cat([y1, y2], dim=1), expected)
