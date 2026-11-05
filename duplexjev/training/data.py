from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset

TASKS = {"BARGE": {"positive": "BARGE_IN", "negative": "NO_BARGE_IN"},
         "TURN_END": {"positive": "TURN_END", "negative": "TURN_CONTINUE"}}

class StreamingEventDataset(Dataset):
    """One item is one decision event; many events may reference the same full recording.

    JSONL event schema is documented in docs/training_pipeline.md. Segments are in
    append order, not reconstructed at training time: this is the prefix visible to Runtime.
    """
    def __init__(self, manifest: str, split: str, *, max_events: int = 0):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.rows: list[dict[str, Any]] = []
        with self.manifest.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip(): continue
                row = json.loads(line)
                if row.get("split") != split: continue
                self._validate(row, line_no)
                self.rows.append(row)
                if max_events and len(self.rows) >= max_events: break
        if not self.rows: raise ValueError(f"No events for split={split!r} in {manifest}")

    def _resolve(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.root / q

    def _validate(self, r: dict, line_no: int):
        required = ("recording_id", "event_id", "time_ms", "query", "label", "feature_file", "segments")
        missing = [k for k in required if k not in r]
        if missing: raise ValueError(f"manifest line {line_no}: missing {missing}")
        if r["query"] not in TASKS: raise ValueError(f"manifest line {line_no}: query must be one of {list(TASKS)}")
        if r["label"] not in (0, 1): raise ValueError(f"manifest line {line_no}: label must be 0 or 1")
        if not isinstance(r["segments"], list) or not r["segments"]:
            raise ValueError(f"manifest line {line_no}: segments must be a non-empty ordered list")
        for s in r["segments"]:
            if s.get("type") == "audio":
                if not (0 <= int(s["start_frame"]) < int(s["end_frame"])): raise ValueError(f"manifest line {line_no}: invalid audio frame range")
                if int(s.get("available_at_ms", r["time_ms"])) > int(r["time_ms"]): raise ValueError(f"manifest line {line_no}: future audio leaks into query prefix")\n                if int(s["start_frame"]) < last_frame: raise ValueError(f"manifest line {line_no}: audio feature frames are duplicated or out of order")\n                last_frame=int(s["end_frame"])
            elif s.get("type") == "text":
                if int(s.get("played_at_ms", r["time_ms"])) > int(r["time_ms"]): raise ValueError(f"manifest line {line_no}: future/unplayed assistant text leaks into prefix")
                if not s.get("token_ids"): raise ValueError(f"manifest line {line_no}: text segment needs token_ids")
            else: raise ValueError(f"manifest line {line_no}: segment type must be audio or text")\n        if not has_audio: raise ValueError(f"manifest line {line_no}: every decision event must include visible user audio")

    def __len__(self): return len(self.rows)

    def __getitem__(self, i: int):
        r = dict(self.rows[i])
        path = self._resolve(r["feature_file"])
        if not path.exists(): raise FileNotFoundError(path)
        obj = torch.load(path, map_location="cpu", weights_only=True)
        features = obj["features"] if isinstance(obj, dict) else obj
        if features.ndim != 2: raise ValueError(f"{path}: expected [frames, encoder_dim], got {tuple(features.shape)}")
        for s in r["segments"]:
            if s["type"] == "audio" and int(s["end_frame"]) > features.size(0): raise ValueError(f"{r['event_id']}: audio segment exceeds cached feature length")
        r["features"] = features.float()
        return r

def collate_events(rows):
    # Variable-length ordered prefixes are assembled individually by the model.
    return rows
