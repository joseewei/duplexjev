from types import SimpleNamespace
import torch
from torch import nn
from duplexjev.training.model import StreamingDecisionModel

class Tok:
    def encode(self,s,add_special_tokens=False): return [1 if s=="B" else 2]
class FakeLM(nn.Module):
    def __init__(self):
        super().__init__(); self.config=SimpleNamespace(hidden_size=8)
        self.emb=nn.Embedding(16,8); self.head=nn.Linear(8,16)
    def get_input_embeddings(self): return self.emb
    def get_output_embeddings(self): return self.head
    def forward(self,inputs_embeds,attention_mask,position_ids,use_cache,output_hidden_states,return_dict):
        h=inputs_embeds.cumsum(dim=1)
        return SimpleNamespace(hidden_states=(h,))

def test_connector_and_query_vectors_receive_gradients():
    model=StreamingDecisionModel(FakeLM(),Tok(),encoder_dim=4,stack_factor=2)
    model.configure_stage("connector")
    row={"event_id":"x","query":"BARGE","features":torch.randn(8,4),
         "segments":[{"type":"text","token_ids":[3,4]},
                     {"type":"audio","start_frame":0,"end_frame":4}]}
    logits=model([row])
    assert logits.shape==(1,2)
    torch.nn.functional.cross_entropy(logits,torch.tensor([0])).backward()
    assert model.connector[1].weight.grad is not None
    assert model.query_embeddings.grad is not None
