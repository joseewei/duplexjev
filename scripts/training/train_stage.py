#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from duplexjev.training.data import StreamingEventDataset
from duplexjev.training.model import StreamingDecisionModel

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_mode(model):
    model.train()
    # Keep the frozen base LM deterministic; only LoRA dropout stays in train mode.
    model.lm.eval()
    for name, module in model.lm.named_modules():
        if "lora_dropout" in name:
            module.train()
    model.connector.train()

def evaluate(model, loader):
    model.eval()
    stats = {}
    with torch.no_grad():
        for rows in loader:
            pred = model(rows).argmax(-1).cpu().tolist()
            for row, p in zip(rows, pred):
                task = row["query"]
                s = stats.setdefault(task, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
                actual = int(row["label"]) == 1
                predicted = p == 0  # output index 0 is positive class B
                if actual and predicted: s["tp"] += 1
                elif not actual and predicted: s["fp"] += 1
                elif not actual and not predicted: s["tn"] += 1
                else: s["fn"] += 1
    out = {}
    for task, s in stats.items():
        tp, fp, tn, fn = (s[k] for k in ("tp", "fp", "tn", "fn"))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        out[task] = {**s, "accuracy": (tp + tn) / max(1, tp + fp + tn + fn),
                     "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / max(1e-12, precision + recall)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--lm-factory", help="optional module:function returning (language_model, tokenizer)")
    ap.add_argument("--encoder-dim", type=int, required=True)
    ap.add_argument("--stack-factor", type=int, default=2)
    ap.add_argument("--stage", choices=["connector", "lora", "joint", "robustness"], required=True)
    ap.add_argument("--tasks", default="BARGE,TURN_END")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--init-adapter")
    ap.add_argument("--init-lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    a = ap.parse_args()
    seed_all(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and a.precision != "fp32":
        raise ValueError("CPU training requires --precision fp32")
    if device == "cuda" and a.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use --precision fp16")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.precision]
    if a.lm_factory:
        module, name = a.lm_factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        lm, tok = factory(a.llm, dtype=dtype, device=device)
    else:
        tok = AutoTokenizer.from_pretrained(a.llm, trust_remote_code=True)
        lm = AutoModelForCausalLM.from_pretrained(a.llm, trust_remote_code=True, torch_dtype=dtype)
    if a.init_lora:
        lm = PeftModel.from_pretrained(lm, a.init_lora, is_trainable=True)
    lm.to(device)
    model = StreamingDecisionModel(lm, tok, encoder_dim=a.encoder_dim, stack_factor=a.stack_factor)
    if a.init_adapter:
        ck = torch.load(a.init_adapter, map_location="cpu", weights_only=False)
        model.connector.load_state_dict(ck["connector"])
        model.query_embeddings.data.copy_(ck["query_embeddings"])
    trainable = model.configure_stage(a.stage, lora_rank=a.lora_rank, lora_alpha=a.lora_alpha,
        target_modules=[x.strip() for x in a.target_modules.split(",") if x.strip()])
    if a.gradient_checkpointing and hasattr(model.lm, "gradient_checkpointing_enable"):
        model.lm.gradient_checkpointing_enable()
    model.to(device)

    train_ds = StreamingEventDataset(a.manifest, "train")
    valid_ds = StreamingEventDataset(a.manifest, "valid")
    selected = {x.strip() for x in a.tasks.split(",") if x.strip()}
    if not selected or not selected.issubset({"BARGE", "TURN_END"}):
        raise ValueError("--tasks must be BARGE,TURN_END or a subset")
    train_ds.rows = [r for r in train_ds.rows if r["query"] in selected]
    valid_ds.rows = [r for r in valid_ds.rows if r["query"] in selected]
    if not train_ds.rows or not valid_ds.rows:
        raise ValueError(f"No train/valid events after task filter: {selected}")
    class_counts = Counter((r["query"], int(r["label"])) for r in train_ds.rows)
    missing = [(task, label) for task in selected for label in (0, 1) if class_counts[(task, label)] == 0]
    if missing:
        raise ValueError(f"Training split is missing task/class groups: {missing}")
    group_weights = {key: len(train_ds) / (len(class_counts) * count) for key, count in class_counts.items()}

    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, collate_fn=lambda x: x)
    valid_loader = DataLoader(valid_ds, batch_size=a.batch_size, shuffle=False,
                              num_workers=a.num_workers, collate_fn=lambda x: x)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    connector_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    groups = []
    if connector_params: groups.append({"params": connector_params, "lr": a.lr})
    if lora_params: groups.append({"params": lora_params, "lr": a.lora_lr})
    if not groups:
        raise RuntimeError("No trainable parameters; check stage/LoRA configuration")
    optimizer = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and a.precision == "fp16"))
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(a.precision, torch.float32)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    for epoch in range(a.epochs):
        train_mode(model)
        total_loss = 0.0
        steps = 0
        for rows in train_loader:
            # Logit order is [B=positive, A=negative]; label 1=positive -> target index 0.
            targets = torch.tensor([1 - int(r["label"]) for r in rows], device=device)
            weights = torch.tensor([group_weights[(r["query"], int(r["label"]))] for r in rows],
                                   device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(device == "cuda" and a.precision != "fp32")):
                logits = model(rows)
                per_item = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                loss = (per_item * weights).mean()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
                optimizer.step()
            total_loss += float(loss.detach())
            steps += 1

        metrics = evaluate(model, valid_loader)
        score = sum(v["f1"] for v in metrics.values()) / max(1, len(metrics))
        report = {"epoch": epoch + 1, "train_loss": total_loss / max(1, steps),
                  "valid": metrics, "macro_f1": score}
        print(json.dumps(report, ensure_ascii=False))
        (out_dir / f"epoch_{epoch + 1}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        metadata = {"stage": a.stage, "epoch": epoch + 1, "encoder_dim": a.encoder_dim,
                    "stack_factor": a.stack_factor, "llm": a.llm, "valid": metrics}
        model.save_trainable(str(out_dir / "last"), metadata)
        if score > best:
            best = score
            model.save_trainable(str(out_dir / "best"), metadata)

if __name__ == "__main__":
    main()
