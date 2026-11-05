from __future__ import annotations
from typing import Any
import torch
from torch import nn
from ..streaming.connector import build_streaming_projector

class StreamingDecisionModel(nn.Module):
    """Frozen causal LM + trainable streaming connector + task-specific query vectors.

    Logit order is [B, A]: B denotes the positive decision (BARGE_IN / TURN_END),
    A denotes the negative decision. Manifest label 1=positive, 0=negative.
    """
    def __init__(self, lm: nn.Module, tokenizer, *, encoder_dim: int, stack_factor: int = 2, connector_hidden_dim: int | None = None):
        super().__init__()
        self.lm, self.tokenizer = lm, tokenizer
        self.stack_factor, self.encoder_dim = int(stack_factor), int(encoder_dim)
        self.hidden_size = int(getattr(lm.config, "hidden_size", getattr(getattr(lm.config, "text_config", None), "hidden_size", 0)))
        if not self.hidden_size: raise ValueError("Cannot infer LM hidden_size from checkpoint config")
        self.connector = build_streaming_projector(self.encoder_dim, self.hidden_size,
            stack_factor=self.stack_factor, hidden_dim=connector_hidden_dim)
        self.input_embeddings = lm.get_input_embeddings()
        self.query_embeddings = nn.Parameter(torch.empty(2, self.hidden_size))
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)
        # Initialize virtual Query vectors from compact natural-language task descriptions;
        # they are still trained as one embedding each and do not add prompt tokens at runtime.
        init_texts = ("判断用户当前是否有效打断AI播报", "判断用户当前是否已经完成本轮表达")
        with torch.no_grad():
            for i, text in enumerate(init_texts):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if ids:
                    token_ids = torch.tensor(ids, dtype=torch.long, device=self.input_embeddings.weight.device)
                    self.query_embeddings[i].copy_(self.input_embeddings(token_ids).mean(dim=0).to(self.query_embeddings.dtype))
        self.output_head = lm.get_output_embeddings()
        self.answer_token_ids = self._answer_ids()

    def _answer_ids(self):
        # Positive class is read from B; negative class from A.
        ids=[]
        for x in ("B","A"):
            t=self.tokenizer.encode(x,add_special_tokens=False)
            if len(t)!=1: raise ValueError(f"Answer token {x!r} must be one token, got {t}")
            ids.append(t[0])
        return ids

    def _sequence(self, row: dict, device):
        chunks=[]
        target_dtype=self.input_embeddings.weight.dtype
        for seg in row["segments"]:
            if seg["type"]=="text":
                ids=torch.tensor(seg["token_ids"],dtype=torch.long,device=device)
                chunks.append(self.input_embeddings(ids).to(dtype=target_dtype))
            else:
                x=row["features"][int(seg["start_frame"]):int(seg["end_frame"])].to(device=device,dtype=next(self.connector.parameters()).dtype)
                if x.size(0)%self.stack_factor: raise ValueError(f"{row['event_id']}: audio segment length must be divisible by stack_factor={self.stack_factor}; split at connector emission boundaries")
                x=x.reshape(-1,self.stack_factor*self.encoder_dim)
                chunks.append(self.connector(x).to(dtype=target_dtype))
        qi=0 if row["query"]=="BARGE" else 1
        chunks.append(self.query_embeddings[qi].to(dtype=target_dtype).view(1,-1))
        return torch.cat(chunks,dim=0)

    def forward(self, rows: list[dict]):
        device=self.query_embeddings.device
        seqs=[self._sequence(r,device) for r in rows]
        lengths=torch.tensor([x.size(0) for x in seqs],device=device)
        max_len=int(lengths.max().item()); dim=seqs[0].size(-1)
        embeds=seqs[0].new_zeros((len(seqs),max_len,dim)); mask=torch.zeros((len(seqs),max_len),dtype=torch.long,device=device)
        for i,x in enumerate(seqs): embeds[i,:x.size(0)]=x; mask[i,:x.size(0)]=1
        pos=(mask.cumsum(-1)-1).clamp_min(0)
        out=self.lm(inputs_embeds=embeds,attention_mask=mask,position_ids=pos,use_cache=False,output_hidden_states=True,return_dict=True)
        hidden=out.hidden_states[-1]
        h=hidden[torch.arange(len(rows),device=device),lengths-1]
        head=self.lm.get_output_embeddings()
        w=head.weight[self.answer_token_ids]
        logits=h.to(w.dtype)@w.T
        if getattr(head,"bias",None) is not None: logits=logits+head.bias[self.answer_token_ids]
        return logits.float()

    def configure_stage(self, stage: str, *, lora_rank: int=16, lora_alpha: int=32, lora_dropout: float=0.05, target_modules: list[str]|None=None):
        for p in self.lm.parameters(): p.requires_grad_(False)
        for p in self.connector.parameters(): p.requires_grad_(True)
        self.query_embeddings.requires_grad_(True)
        if stage in ("lora","joint","robustness") and not getattr(self.lm,"peft_config",None):
            try: from peft import LoraConfig,get_peft_model
            except ImportError as e: raise RuntimeError("Install peft for LoRA stages: pip install peft") from e
            cfg=LoraConfig(r=lora_rank,lora_alpha=lora_alpha,lora_dropout=lora_dropout,target_modules=target_modules or ["q_proj","k_proj","v_proj","o_proj"],bias="none",task_type="CAUSAL_LM")
            self.lm=get_peft_model(self.lm,cfg)
            self.input_embeddings=self.lm.get_input_embeddings()
        elif stage in ("lora","joint","robustness") and getattr(self.lm,"peft_config",None):
            for name,p in self.lm.named_parameters():
                if "lora_" in name: p.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

    def save_trainable(self, output_dir: str, metadata: dict[str,Any]):
        import json
        from pathlib import Path
        p=Path(output_dir); p.mkdir(parents=True,exist_ok=True)
        torch.save({"connector":self.connector.state_dict(),"query_embeddings":self.query_embeddings.detach().cpu(),"metadata":metadata},p/"streaming_adapter.pt")
        if hasattr(self.lm,"save_pretrained") and getattr(self.lm,"peft_config",None): self.lm.save_pretrained(p/"lora")
        (p/"training_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
