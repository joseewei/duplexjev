"""Load trained streaming connector/query vectors and optional LLM LoRA weights."""
from __future__ import annotations
import torch

def load_streaming_adapter(session, projector, adapter_path: str, *, lora_path: str | None = None):
    """Load a training output into the runtime session.

    projector must be the same MLP module used by StreamingConnector (not the wrapper).
    lora_path points to the PEFT adapter directory saved beside streaming_adapter.pt.
    """
    ckpt=torch.load(adapter_path,map_location="cpu",weights_only=False)
    projector.load_state_dict(ckpt["connector"])
    session.set_query_embeddings(ckpt["query_embeddings"])
    if lora_path:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise RuntimeError("Install peft to load the trained LoRA adapter") from e
        session.lm=PeftModel.from_pretrained(session.lm,lora_path,is_trainable=False)
        session.lm.eval()
    return ckpt.get("metadata",{})
