#!/usr/bin/env python3
"""Validate all event rows and summarize split/task/class balance."""
import argparse,json
from collections import Counter
from pathlib import Path
from duplexjev.training.data import StreamingEventDataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--out",default="manifest_report.json"); a=ap.parse_args()
    report={}
    for split in ("train","valid","test"):
        try: ds=StreamingEventDataset(a.manifest,split)
        except ValueError: report[split]={"events":0}; continue
        counts=Counter((r["query"],int(r["label"])) for r in ds.rows)
        unique_features={ds._resolve(r["feature_file"]) for r in ds.rows}
        for path in unique_features:
            if not path.exists(): raise FileNotFoundError(path)
            obj=__import__("torch").load(path,map_location="cpu",weights_only=True)
            x=obj["features"] if isinstance(obj,dict) else obj
            if x.ndim!=2: raise ValueError(f"{path}: expected [T,D] feature tensor")
        report[split]={"events":len(ds),"recordings":len({r["recording_id"] for r in ds.rows}),
            "feature_files":len(unique_features),
            "counts":{f"{task}:label_{label}":n for (task,label),n in counts.items()}}
    # Hard leakage guard: a recording may occur in exactly one split.
    split_by_recording={}
    for split in ("train","valid","test"):
        try: ds=StreamingEventDataset(a.manifest,split)
        except ValueError: continue
        for row in ds.rows:
            old=split_by_recording.setdefault(row["recording_id"],split)
            if old!=split: raise ValueError(f"recording leakage: {row['recording_id']} appears in {old} and {split}")
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
