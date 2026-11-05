#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer
from duplexjev.training.data import StreamingEventDataset
from duplexjev.training.model import StreamingDecisionModel

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def evaluate(model,loader):
 model.eval(); n={}; ok={}
 with torch.no_grad():
  for rows in loader:
   pred=model(rows).argmax(-1).cpu().tolist()
   for r,p in zip(rows,pred):
    k=r["query"]; n[k]=n.get(k,0)+1; ok[k]=ok.get(k,0)+int(p==int(r["label"]))
 return {k:{"n":n[k],"accuracy":ok[k]/n[k]} for k in n}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True); ap.add_argument("--llm",required=True); ap.add_argument("--encoder-dim",type=int,required=True); ap.add_argument("--stack-factor",type=int,default=2); ap.add_argument("--stage",choices=["connector","lora","joint"],required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--init-adapter"); ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--batch-size",type=int,default=2); ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--lora-lr",type=float,default=2e-5); ap.add_argument("--weight-decay",type=float,default=.01); ap.add_argument("--max-grad-norm",type=float,default=1.0); ap.add_argument("--lora-rank",type=int,default=16); ap.add_argument("--lora-alpha",type=int,default=32); ap.add_argument("--target-modules",default="q_proj,k_proj,v_proj,o_proj"); ap.add_argument("--num-workers",type=int,default=0); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--bf16",action="store_true"); ap.add_argument("--gradient-checkpointing",action="store_true"); a=ap.parse_args()
 seed_all(a.seed); device="cuda" if torch.cuda.is_available() else "cpu"; dtype=torch.bfloat16 if a.bf16 else torch.float32
 tok=AutoTokenizer.from_pretrained(a.llm,trust_remote_code=True); lm=AutoModelForCausalLM.from_pretrained(a.llm,trust_remote_code=True,torch_dtype=dtype); lm.to(device)
 model=StreamingDecisionModel(lm,tok,encoder_dim=a.encoder_dim,stack_factor=a.stack_factor)
 if a.init_adapter:
  ck=torch.load(a.init_adapter,map_location="cpu",weights_only=False); model.connector.load_state_dict(ck["connector"]); model.query_embeddings.data.copy_(ck["query_embeddings"])
 trainable=model.configure_stage(a.stage,lora_rank=a.lora_rank,lora_alpha=a.lora_alpha,target_modules=[x.strip() for x in a.target_modules.split(",") if x.strip()])
 if a.gradient_checkpointing and hasattr(model.lm,"gradient_checkpointing_enable"): model.lm.gradient_checkpointing_enable()
 model.to(device); tr=StreamingEventDataset(a.manifest,"train"); va=StreamingEventDataset(a.manifest,"valid")
 trl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers,collate_fn=lambda x:x); val=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,collate_fn=lambda x:x)
 lp=[p for n,p in model.named_parameters() if p.requires_grad and "lora_" in n]; cp=[p for n,p in model.named_parameters() if p.requires_grad and "lora_" not in n]; groups=[]
 if cp: groups.append({"params":cp,"lr":a.lr})
 if lp: groups.append({"params":lp,"lr":a.lora_lr})
 opt=torch.optim.AdamW(groups,weight_decay=a.weight_decay); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); best=-1
 for ep in range(a.epochs):
  model.train(); total=0.; steps=0
  for rows in trl:
   labels=torch.tensor([int(r["label"]) for r in rows],device=device); opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=(device=="cuda" and a.bf16)): logits=model(rows); loss=torch.nn.functional.cross_entropy(logits,labels)
   loss.backward(); torch.nn.utils.clip_grad_norm_(trainable,a.max_grad_norm); opt.step(); total+=float(loss.detach()); steps+=1
  metrics=evaluate(model,val); score=sum(v["accuracy"] for v in metrics.values())/max(1,len(metrics)); report={"epoch":ep+1,"train_loss":total/max(1,steps),"valid":metrics,"macro_accuracy":score}; print(json.dumps(report,ensure_ascii=False)); (out/f"epoch_{ep+1}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
  meta={"stage":a.stage,"epoch":ep+1,"encoder_dim":a.encoder_dim,"stack_factor":a.stack_factor,"llm":a.llm,"valid":metrics}; model.save_trainable(str(out/"last"),meta)
  if score>best: best=score; model.save_trainable(str(out/"best"),meta)
if __name__=="__main__": main()
