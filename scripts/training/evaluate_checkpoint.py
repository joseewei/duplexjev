#!/usr/bin/env python3
"""Evaluate a saved streaming connector/query adapter on the held-out test split."""
import argparse,importlib,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from duplexjev.training.data import StreamingEventDataset
from duplexjev.training.model import StreamingDecisionModel

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--manifest",required=True); ap.add_argument("--llm",required=True); ap.add_argument("--encoder-dim",type=int,required=True)
 ap.add_argument("--stack-factor",type=int,default=2); ap.add_argument("--adapter",required=True); ap.add_argument("--lora")
 ap.add_argument("--lm-factory"); ap.add_argument("--tasks",default="BARGE,TURN_END"); ap.add_argument("--batch-size",type=int,default=1)
 ap.add_argument("--precision",choices=["fp32","fp16","bf16"],default="fp16"); ap.add_argument("--output",default="test_metrics.json")
 a=ap.parse_args(); device="cuda" if torch.cuda.is_available() else "cpu"
 dtype={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[a.precision]
 if a.lm_factory:
  mod,name=a.lm_factory.split(":",1); lm,tok=getattr(importlib.import_module(mod),name)(a.llm,dtype=dtype,device=device)
 else:
  tok=AutoTokenizer.from_pretrained(a.llm,trust_remote_code=True)
  lm=AutoModelForCausalLM.from_pretrained(a.llm,trust_remote_code=True,torch_dtype=dtype)
 if a.lora: lm=PeftModel.from_pretrained(lm,a.lora,is_trainable=False)
 lm.to(device)
 model=StreamingDecisionModel(lm,tok,encoder_dim=a.encoder_dim,stack_factor=a.stack_factor)
 ck=torch.load(a.adapter,map_location="cpu",weights_only=False)
 model.connector.load_state_dict(ck["connector"]); model.query_embeddings.data.copy_(ck["query_embeddings"])
 model.to(device).eval()
 selected={x.strip() for x in a.tasks.split(",") if x.strip()}
 ds=StreamingEventDataset(a.manifest,"test"); ds.rows=[r for r in ds.rows if r["query"] in selected]
 if not ds.rows: raise ValueError(f"No test events for tasks {selected}")
 loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,collate_fn=lambda x:x)
 stats={}; lat=[]
 with torch.no_grad():
  for rows in loader:
   if device=="cuda": torch.cuda.synchronize()
   t=time.perf_counter(); pred=model(rows).argmax(-1).cpu().tolist()
   if device=="cuda": torch.cuda.synchronize()
   lat.append((time.perf_counter()-t)*1000)
   for r,p in zip(rows,pred):
    s=stats.setdefault(r["query"],{"tp":0,"fp":0,"tn":0,"fn":0})
    actual=int(r["label"])==1; positive=(p==0)
    if actual and positive:s["tp"]+=1
    elif not actual and positive:s["fp"]+=1
    elif not actual and not positive:s["tn"]+=1
    else:s["fn"]+=1
 result={}
 for task,s in stats.items():
  tp,fp,tn,fn=(s[k] for k in ("tp","fp","tn","fn")); pr=tp/max(1,tp+fp); re=tp/max(1,tp+fn)
  result[task]={**s,"accuracy":(tp+tn)/max(1,tp+fp+tn+fn),"precision":pr,"recall":re,"f1":2*pr*re/max(1e-12,pr+re)}
 result["latency_ms"]={"mean":sum(lat)/len(lat),"p50":sorted(lat)[len(lat)//2],"p95":sorted(lat)[min(len(lat)-1,int(.95*len(lat)))]}
 Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
