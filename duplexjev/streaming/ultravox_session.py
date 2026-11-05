"""Append-only Ultravox/causal-LM session with isolated typed-query branches."""
from __future__ import annotations
import copy
import threading
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import torch
from ..question import Question

@dataclass
class StreamingLMState:
    cache: Any
    sequence_length: int = 0
    state_version: int = 0
    closed: bool = False

class StreamingUltravoxSession:
    """One-call streaming context backed by the Ultravox language model.

    Audio/text prefixes are committed once. A learned one-vector query is appended
    temporarily for each decision and rolled back after reading answer-letter logits.
    """
    def __init__(self, model, tokenizer, *, prefix_text: str = "", device: Optional[object] = None):
        from transformers.cache_utils import DynamicCache
        self.model=model.eval(); self.lm=model.language_model; self.tokenizer=tokenizer
        self.device=torch.device(device) if device is not None else self.lm.get_input_embeddings().weight.device
        lm_type=str(getattr(getattr(self.lm,"config",None),"model_type",""))
        unsupported={"mamba","mamba2","jamba","lfm2","lfm2_moe","rwkv","falcon_h1","zamba","zamba2","bamba","nemotron_h","granitemoehybrid","recurrent_gemma"}
        if getattr(model,"is_diffusion_gemma",False) or lm_type in unsupported: raise NotImplementedError(f"Streaming cache rollback is not validated for LM type {lm_type!r}")
        self._lock=threading.RLock(); self.state=StreamingLMState(cache=DynamicCache())
        self._letter_ids: dict[str,int]={}; self.query_embeddings: Optional[torch.Tensor]=None
        if prefix_text: self.append_text(prefix_text)

    @property
    def cache_length(self): return self.state.sequence_length

    def _token_ids(self,text):
        encoded=self.tokenizer(text,add_special_tokens=False,return_tensors="pt")
        ids=encoded["input_ids"] if isinstance(encoded,dict) else encoded.input_ids
        return ids.to(self.device)

    def set_query_embeddings(self, embeddings: torch.Tensor):
        """Load the trained [2, hidden] query vectors; index 0=BARGE, 1=TURN_END."""
        if embeddings.ndim!=2 or embeddings.size(0)!=2: raise ValueError("query embeddings must have shape [2, hidden]")
        expected=self.lm.get_input_embeddings().weight.size(-1)
        if embeddings.size(1)!=expected: raise ValueError(f"query hidden size {embeddings.size(1)} != LM hidden size {expected}")
        self.query_embeddings=embeddings.detach().to(device=self.device,dtype=self.lm.get_input_embeddings().weight.dtype)

    def _forward_append(self,embeds):
        if embeds.ndim==2: embeds=embeds.unsqueeze(0)
        if embeds.ndim!=3 or embeds.size(0)!=1: raise ValueError("stream append expects [T,D] or [1,T,D]")
        if embeds.size(1)==0: return
        embeds=embeds.to(device=self.device,dtype=self.lm.get_input_embeddings().weight.dtype)
        past=self.state.sequence_length; total=past+embeds.size(1)
        self.lm(inputs_embeds=embeds,attention_mask=torch.ones((1,total),dtype=torch.long,device=self.device),
                past_key_values=self.state.cache,cache_position=torch.arange(past,total,dtype=torch.long,device=self.device),
                use_cache=True,return_dict=True)
        self.state.sequence_length=total

    @torch.inference_mode()
    def append_text_tokens(self,token_ids):
        if token_ids.ndim==1: token_ids=token_ids.unsqueeze(0)
        if token_ids.ndim!=2 or token_ids.size(0)!=1: raise ValueError("token_ids must have shape [T] or [1,T]")
        with self._lock:
            if self.state.closed: raise RuntimeError("session is closed")
            token_ids=token_ids.to(self.device); self._forward_append(self.lm.get_input_embeddings()(token_ids))
            return self.cache_length

    @torch.inference_mode()
    def append_text(self,text):
        if not text: return self.cache_length
        return self.append_text_tokens(self._token_ids(text))

    @torch.inference_mode()
    def append_audio_embeddings(self,embeddings):
        with self._lock:
            if self.state.closed: raise RuntimeError("session is closed")
            self._forward_append(embeddings.to(self.device))
            return self.cache_length

    def _letter_id(self,letter):
        if letter not in self._letter_ids:
            ids=self.tokenizer.encode(letter,add_special_tokens=False)
            if len(ids)!=1: raise ValueError(f"option {letter!r} is not one tokenizer token: {ids}")
            self._letter_ids[letter]=int(ids[0])
        return self._letter_ids[letter]

    @torch.inference_mode()
    def query(self,question: Question,*,suffix: str="",expected_state_version: Optional[int]=None,
              query_embedding: Optional[torch.Tensor]=None,answer_letters: Sequence[str]=("A","B")) -> dict[str,Any]:
        """Read option logits on a temporary cache branch, then rollback.

        If query_embedding is supplied, it is the trained dedicated Query vector and
        the natural-language Question body is not tokenized or recomputed.
        """
        with self._lock:
            if self.state.closed: raise RuntimeError("session is closed")
            version=self.state.state_version
            if expected_state_version is not None and version!=expected_state_version: return {"stale":True,"state_version":version}
            if len(answer_letters)!=len(question.options): raise ValueError("answer_letters and question.options must have equal length")
            base_len=self.state.sequence_length
            if query_embedding is None:
                prompt=question.render(list(range(len(question.options))))
                ids=self._token_ids(prompt+suffix)
                if ids.size(1)==0: raise ValueError("query prompt tokenized to an empty sequence")
                q_embeds=self.lm.get_input_embeddings()(ids)
            else:
                q=torch.as_tensor(query_embedding,device=self.device,dtype=self.lm.get_input_embeddings().weight.dtype)
                if q.ndim==1: q=q.unsqueeze(0)
                if q.ndim!=2 or q.size(0)!=1: raise ValueError("query_embedding must have shape [hidden] or [1,hidden]")
                q_embeds=q.unsqueeze(0)
            q_len=int(q_embeds.size(1)); cache=self.state.cache; crop=getattr(cache,"crop",None)
            if not callable(crop):
                try: cache=copy.deepcopy(cache)
                except Exception as exc: raise RuntimeError("Cache supports neither crop() nor deepcopy().") from exc
            try:
                output=self.lm(inputs_embeds=q_embeds,attention_mask=torch.ones((1,base_len+q_len),dtype=torch.long,device=self.device),
                    past_key_values=cache,cache_position=torch.arange(base_len,base_len+q_len,dtype=torch.long,device=self.device),
                    use_cache=True,return_dict=True)
                logits=output.logits[0,-1].float()
                ids=torch.tensor([self._letter_id(x) for x in answer_letters],dtype=torch.long,device=logits.device)
                probs=torch.softmax(logits.index_select(0,ids),dim=-1)
            finally:
                if callable(crop): cache.crop(base_len)
            best=int(probs.argmax().item())
            return {"stale":False,"state_version":version,"query_cache_length":base_len+q_len,
                    "shared_cache_length":self.state.sequence_length,"options":list(question.options),
                    "probabilities":probs.detach().cpu().tolist(),"best_index":best,
                    "best_option":question.options[best],"confidence":float(probs.max().item())}

    def set_state_version(self,version):
        with self._lock:
            if version<self.state.state_version: raise ValueError("state_version cannot move backwards")
            self.state.state_version=int(version)

    def close(self):
        with self._lock: self.state.closed=True; self.state.cache=None
