#!/usr/bin/env python3
"""Join timestamped event annotations to cached features; split by recording_id."""
import argparse,json,random
from collections import Counter
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--annotations",required=True); ap.add_argument("--feature-index",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--train-ratio",type=float,default=.9); ap.add_argument("--valid-ratio",type=float,default=.05); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    if a.train_ratio<=0 or a.valid_ratio<=0 or a.train_ratio+a.valid_ratio>=1:
        raise ValueError("ratios must satisfy train>0, valid>0, train+valid<1")
    idx={x["recording_id"]:x["feature_file"] for x in json.loads(Path(a.feature_index).read_text(encoding="utf-8"))}
    rows=[json.loads(x) for x in Path(a.annotations).read_text(encoding="utf-8").splitlines() if x.strip()]
    ids=sorted({r["recording_id"] for r in rows})
    if len(ids)<3 and not all("split" in r for r in rows):
        raise ValueError("Need at least 3 recordings for automatic train/valid/test splitting; or provide explicit split on every event")
    rng=random.Random(a.seed); rng.shuffle(ids); n=len(ids)
    ntr=max(1,int(n*a.train_ratio)); nva=max(1,int(n*a.valid_ratio))
    if ntr+nva>=n: ntr=max(1,n-2); nva=1
    assigned={rid:("train" if i<ntr else "valid" if i<ntr+nva else "test") for i,rid in enumerate(ids)}
    out=[]; split_by_recording={}
    for r in rows:
        rid=r["recording_id"]
        if rid not in idx: raise ValueError(f"missing cached encoder features for recording_id={rid}")
        r["feature_file"]=idx[rid]; r.setdefault("split",assigned[rid])
        if r["split"] not in ("train","valid","test"): raise ValueError(f"invalid split {r['split']!r}")
        old=split_by_recording.setdefault(rid,r["split"])
        if old!=r["split"]: raise ValueError(f"recording {rid} assigned to multiple splits")
        out.append(r)
    out.sort(key=lambda x:(x["recording_id"],int(x["time_ms"]),x["event_id"]))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with Path(a.out).open("w",encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    counts=Counter(split_by_recording.values())
    print(json.dumps({"recordings":len(ids),"events":len(out),"splits":dict(counts)},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
