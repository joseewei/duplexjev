import json
import torch
import pytest
from duplexjev.training.data import StreamingEventDataset

def _write(tmp_path, available=90):
    torch.save({"features":torch.randn(8,4)},tmp_path/"feat.pt")
    row={"recording_id":"r1","event_id":"e1","split":"train","time_ms":100,"query":"BARGE","label":1,
         "feature_file":"feat.pt","segments":[{"type":"audio","start_frame":0,"end_frame":4,"available_at_ms":available}]}
    p=tmp_path/"events.jsonl"; p.write_text(json.dumps(row)+"\n",encoding="utf-8"); return p

def test_event_dataset_loads_prefix_features(tmp_path):
    p=_write(tmp_path)
    ds=StreamingEventDataset(str(p),"train")
    item=ds[0]
    assert item["features"].shape==(8,4)
    assert item["label"]==1

def test_event_dataset_rejects_future_audio(tmp_path):
    p=_write(tmp_path,available=101)
    with pytest.raises(ValueError,match="future audio"):
        StreamingEventDataset(str(p),"train")
