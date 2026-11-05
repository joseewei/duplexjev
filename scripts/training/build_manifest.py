#!/usr/bin/env python3
"""Join event annotations to cached streaming features and split by recording_id.

Input annotations JSONL: recording_id,event_id,time_ms,query,label,segments.
Feature index: output of cache_streaming_features.py (recording_id,feature_file,...).
Optional split in annotation is respected; otherwise a deterministic recording-level split is assigned.
"""
import argparse,json,random
from collections import defaultdict
from pathlib import Path

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--annotations",required=True); ap.add_argument("--feature-index",required=True); ap.add_argument("--out",required=True)
 ap.add_argument("--train-ratio",type=float,default=.9); ap.add_argument("--valid-ratio",type=float,default=.05); ap.add_argument("--seed",type=int,default=42)
 a=ap.parse_args()
 if a.train_ratio<=0 or a.valid_ratio<0 or a.train_ratio+a.valid_ratio>=1: raise ValueError("ratios must satisfy train>0, valid>=0, train+valid<1")
 idx={x["recording_id"]:x["feature_file"] for x in json.loads(Path(a.feature_index).read_text(encoding="utf-8"))}
 rows=[json.loads(x) for x in Path(a.annotations).read_text(encoding="utf-8").splitlines() if x.strip()]
 ids=sorted({r["recording_id"] for r in rows}); rng=random.Random(a.seed); rng.shuffle(ids)
 ntr=round(len(ids)*a.train_ratio); nva=round(len(ids)*a.valid_ratio)
 split={rid:("train" if i<ntr else "valid" if i<ntr+nva else "test") for i,rid in enumerate(ids)}
 out=[]
 for r in rows:
  rid=r["recording_id"]
  if rid not in idx: raise ValueError(f"missing cached encoder features for recording_id={rid}")
  r["feature_file"]=idx[rid]
  r.setdefault("split",split[rid])
  out.append(r)
 out.sort(key=lambda x:(x["recording_id"],int(x["time_ms"]),x["event_id"]))
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 with Path(a.out).open("w",encoding="utf-8") as f:
  for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
 print(json.dumps({"recordings":len(ids),"events":len(out),"splits":{s:sum(v==s for v in split.values()) for s in ("train","valid","test")}},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
