#!/usr/bin/env python3
"""Validate cached feature references, event prefixes and recording-level splits."""
import argparse,json
from collections import Counter
from pathlib import Path
import torch
from duplexjev.training.data import StreamingEventDataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--out",default="manifest_report.json"); a=ap.parse_args()
    rows=[json.loads(x) for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    report={}; split_by_recording={}
    for row in rows:
        split=row.get("split")
        if split not in ("train","valid","test"): raise ValueError(f"unknown/missing split {split!r}")
        rid=row["recording_id"]
        old=split_by_recording.setdefault(rid,split)
        if old!=split: raise ValueError(f"recording leakage: {rid} appears in {old} and {split}")
    for split in ("train","valid","test"):
        if not any(r.get("split")==split for r in rows):
            report[split]={"events":0}; continue
        ds=StreamingEventDataset(a.manifest,split)
        counts=Counter((r["query"],int(r["label"])) for r in ds.rows)
        unique_features={ds._resolve(r["feature_file"]) for r in ds.rows}
        for path in unique_features:
            if not path.exists(): raise FileNotFoundError(path)
            obj=torch.load(path,map_location="cpu",weights_only=True)
            x=obj["features"] if isinstance(obj,dict) else obj
            if x.ndim!=2: raise ValueError(f"{path}: expected [T,D] feature tensor")
        report[split]={"events":len(ds),"recordings":len({r["recording_id"] for r in ds.rows}),
            "feature_files":len(unique_features),
            "counts":{f"{task}:label_{label}":n for (task,label),n in counts.items()}}
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
