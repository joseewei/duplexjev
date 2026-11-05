"""State-routed query runtime for streaming DuplexJev decisions."""
from enum import Enum
from typing import Callable, Any
from .context import StreamingContext

class RuntimeState(str,Enum):
    IDLE="IDLE"; LISTEN="LISTEN"; SPEAK="SPEAK"; DUAL="DUAL"; THINK="THINK"
class QueryKind(str,Enum):
    BARGE="QUERY_BARGE"; TURN_END="QUERY_TURN_END"

class DuplexStreamingRuntime:
    """Route decisions from authoritative Voice Runtime state."""
    def __init__(self,decision_fn:Callable[...,dict],*,barge_threshold:float=.75,turn_threshold:float=.75,on_ai_played_text=None,on_audio_embeddings=None):
        self.decision_fn=decision_fn
        self.on_ai_played_text=on_ai_played_text or getattr(decision_fn,"on_ai_played_text",None)
        self.on_audio_embeddings=on_audio_embeddings or getattr(decision_fn,"on_audio_embeddings",None)
        self.barge_threshold=float(barge_threshold); self.turn_threshold=float(turn_threshold)
        self.context=StreamingContext(); self.last_result:dict[str,Any]|None=None
    def set_state(self,state:RuntimeState|str)->int: return self.context.set_state(RuntimeState(state).value)
    def commit_ai_playback(self,text:str,*,playback_offset_ms:int)->int:
        seq=self.context.commit_played_text(text,playback_offset_ms=playback_offset_ms)
        if text and self.on_ai_played_text is not None: self.on_ai_played_text(text)
        return seq
    def append_audio_event(self,audio_embedding_chunk:Any,*,start_ms:int,end_ms:int)->int:
        seq=self.context.append_user_audio(audio_embedding_chunk,start_ms=start_ms,end_ms=end_ms)
        if self.on_audio_embeddings is not None: self.on_audio_embeddings(audio_embedding_chunk)
        return seq
    def query(self,audio_prefix:Any)->dict[str,Any]|None:
        state=self.context.runtime_state
        if state==RuntimeState.DUAL.value: kind=QueryKind.BARGE
        elif state==RuntimeState.LISTEN.value: kind=QueryKind.TURN_END
        else: return None
        version=self.context.state_version
        result=self.decision_fn(kind.value,audio_prefix,self.context.snapshot(),version)
        if version!=self.context.state_version: return None
        result=dict(result); result.update(query_kind=kind.value,state_version=version); self.last_result=result
        return result
    def should_interrupt(self,result):
        return result.get("query_kind")==QueryKind.BARGE.value and result.get("answer")=="BARGE_IN" and float(result.get("confidence",0))>=self.barge_threshold
    def should_respond(self,result):
        return result.get("query_kind")==QueryKind.TURN_END.value and result.get("answer")=="TURN_END" and float(result.get("confidence",0))>=self.turn_threshold

def make_duplexjev_decision_fn(session,*,barge_question,turn_question,suffix:str="",played_text_encoder=None,query_embeddings=None):
    """Bind a streaming session to Runtime.

    With trained query_embeddings [2,H], index 0 is BARGE and 1 is TURN_END.
    The model was trained with answer-token order [B,A], where B is positive;
    therefore options remain (positive, negative) and answer_letters=("B","A").
    Without trained vectors, the legacy natural-language Question path remains available.
    """
    expected_barge=("BARGE_IN","NO_BARGE_IN"); expected_turn=("TURN_END","TURN_CONTINUE")
    if tuple(barge_question.options)!=expected_barge: raise ValueError(f"barge_question.options must be {expected_barge}")
    if tuple(turn_question.options)!=expected_turn: raise ValueError(f"turn_question.options must be {expected_turn}")
    if query_embeddings is not None: session.set_query_embeddings(query_embeddings)

    def decide(query_kind,audio_prefix,context_snapshot,state_version):
        session.set_state_version(state_version)
        is_barge=query_kind==QueryKind.BARGE.value
        question=barge_question if is_barge else turn_question
        if session.query_embeddings is not None:
            qvec=session.query_embeddings[0 if is_barge else 1]
            result=session.query(question,suffix="",expected_state_version=state_version,
                                 query_embedding=qvec,answer_letters=("B","A"))
        else:
            result=session.query(question,suffix=suffix,expected_state_version=state_version)
        if result.get("stale"): return {"answer":"STALE","confidence":0.0}
        return {"answer":result["best_option"],"confidence":result["confidence"],"probabilities":result["probabilities"]}
    decide.on_ai_played_text=(lambda text:session.append_text_tokens(played_text_encoder(text))) if played_text_encoder is not None else session.append_text
    decide.on_audio_embeddings=session.append_audio_embeddings
    return decide
