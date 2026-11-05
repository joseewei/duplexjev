#!/usr/bin/env python3
"""Run a user-supplied trained streaming encoder adapter and cache frame features.
Adapter factory module:function(config_dict) returns an object with encode_file(path, chunk_ms)->Tensor[T,D].
The adapter must use the actual streaming encoder, frontend and CMVN from the trained checkpoint.
"""
import argparse,importlib,json
from pathlib import Path
import torch

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--audio-list",required=True); ap.add_argument("--adapter",required=True); ap.add_argument("--config",default="{}"); ap.add_argument("--out-dir",required=True); ap.add_argument("--chunk-ms",type=int,default=160); a=ap.parse_args()
 mod,name=a.adapter.split(":",1); factory=getattr(importlib.import_module(mod),name); adapter=factory(json.loads(a.config)); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); index=[]
 for line in Path(a.audio_list).read_text(encoding="utf-8").splitlines():
  if not line.strip(): continue
  r=json.loads(line); x=adapter.encode_file(r["audio_path"],chunk_ms=a.chunk_ms)
  if not torch.is_tensor(x) or x.ndim!=2: raise ValueError(f"{r['recording_id']}: adapter must return Tensor[T,D]")
  name=f"{r['recording_id']}.pt"; torch.save({"features":x.detach().cpu().float()},out/name); index.append({"recording_id":r["recording_id"],"feature_file":str((out/name).resolve()),"num_frames":x.size(0),"dim":x.size(1)}); print(f"cached {r['recording_id']}: {tuple(x.shape)}")
 (out/"index.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
if __name__=="__main__": main()
