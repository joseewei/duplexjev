from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

TASKS = {"BARGE": {"positive": "BARGE_IN", "negative": "NO_BARGE_IN"},
         "TURN_END": {"positive": "TURN_END", "negative": "TURN_CONTINUE"}}

class StreamingEventDataset(Dataset):
    """One decision event per item; multiple events may reference one complete recording."""
    def __init__(self, manifest: str, split: str, *, max_events: int = 0):
        self.manifest=Path(manifest).resolve()
        self.root=self.manifest.parent
        self.rows=[]
        with self.manifest.open(encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                row=json.loads(line)
                if row.get("split")!=split: continue
                self._validate(row,line_no)
                self.rows.append(row)
                if max_events and len(self.rows)>=max_events: break
        if not self.rows: raise ValueError(f"No events for split={split!r} in {manifest}")

    def _resolve(self,p):
        q=Path(p)
        return q if q.is_absolute() else self.root/q

    def _validate(self,r,line_no):
        required=("recording_id","event_id","time_ms","query","label","feature_file","segments")
        missing=[k for k in required if k not in r]
        if missing: raise ValueError(f"manifest line {line_no}: missing {missing}")
        if r["query"] not in TASKS: raise ValueError(f"manifest line {line_no}: invalid query {r['query']}")
        if int(r["label"]) not in (0,1): raise ValueError(f"manifest line {line_no}: label must be 0 or 1")
        if not isinstance(r["segments"],list) or not r["segments"]: raise ValueError(f"manifest line {line_no}: segments must be a non-empty list")
        has_audio=False
        last_append_ms=-1
        last_frame=0
        for s in r["segments"]:
            append_ms=int(s.get("append_at_ms",s.get("available_at_ms",s.get("played_at_ms",r["time_ms"]))))
            if append_ms<last_append_ms: raise ValueError(f"manifest line {line_no}: segments are not in append-time order")
            last_append_ms=append_ms
            if s.get("type")=="audio":
                has_audio=True
                if "available_at_ms" not in s: raise ValueError(f"manifest line {line_no}: audio segment requires available_at_ms")
                start,end=int(s["start_frame"]),int(s["end_frame"])
                if not (0<=start<end): raise ValueError(f"manifest line {line_no}: invalid audio frame range")
                if int(s["available_at_ms"])>int(r["time_ms"]): raise ValueError(f"manifest line {line_no}: future audio leaks into query prefix")
                if start<last_frame: raise ValueError(f"manifest line {line_no}: audio frames are duplicated or out of order")
                last_frame=end
            elif s.get("type")=="text":
                if s.get("source")=="static":
                    if "append_at_ms" not in s: raise ValueError(f"manifest line {line_no}: static text requires append_at_ms")
                    if append_ms>int(r["time_ms"]): raise ValueError(f"manifest line {line_no}: future static prompt leaks into prefix")
                else:
                    if "played_at_ms" not in s: raise ValueError(f"manifest line {line_no}: assistant text requires played_at_ms")
                    if int(s["played_at_ms"])>int(r["time_ms"]): raise ValueError(f"manifest line {line_no}: future/unplayed assistant text leaks into prefix")
                if not s.get("token_ids"): raise ValueError(f"manifest line {line_no}: text segment needs token_ids")
            else:
                raise ValueError(f"manifest line {line_no}: segment type must be audio or text")
        if not has_audio: raise ValueError(f"manifest line {line_no}: every decision event must include visible user audio")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        r=dict(self.rows[i])
        path=self._resolve(r["feature_file"])
        if not path.exists(): raise FileNotFoundError(path)
        obj=torch.load(path,map_location="cpu",weights_only=True)
        features=obj["features"] if isinstance(obj,dict) else obj
        if features.ndim!=2: raise ValueError(f"{path}: expected [frames, encoder_dim], got {tuple(features.shape)}")
        for s in r["segments"]:
            if s["type"]=="audio" and int(s["end_frame"])>features.size(0):
                raise ValueError(f"{r['event_id']}: audio segment exceeds cached feature length")
        r["features"]=features.float()
        return r

def collate_events(rows):
    return rows
