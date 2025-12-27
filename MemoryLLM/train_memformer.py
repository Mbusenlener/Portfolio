import os
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset
from huggingface_hub import login

from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from nltk.tokenize import sent_tokenize
import numpy as np 

# Config
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    K: int = 16  # number of memory tokens per context per layer

    # LoRA hyperparams
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_accum_steps: int = 4
    max_epochs: int = 5
    patience: int = 1
    min_delta: float = 1e-4

    max_new_tokens: int = 5
    eos_token_id: Optional[int] = None

    val_size: float = 0.1
    test_size: float = 0.1

    ckpt_dir: str = "dual_lora_ckpt"

    pretrain_epochs: int = 1  # epoch 0 only by default




# Helpers
def make_splits(df: pd.DataFrame, cfg: TrainConfig):
    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True
    )
    train_df, val_df = train_test_split(
        train_df, test_size=cfg.val_size/(1-cfg.test_size),
        random_state=cfg.seed, shuffle=True
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_qa(tokenizer, question: str, answer: str):
    prompt = f"Question: {question}\nAnswer:"
    full_text = prompt + " " + answer + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]

    labels = full_ids.clone()
    labels[: len(prompt_ids)] = -100

    prompt_len = len(prompt_ids)
    return full_ids.unsqueeze(0), labels.unsqueeze(0), prompt_len

def tokenize_reconstruction(tokenizer, full_context: str):
    prompt = "Answer:"
    full_text = prompt + " " + full_context + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]

    labels = full_ids.clone()
    labels[: len(prompt_ids)] = -100

    prompt_len = len(prompt_ids)
    return full_ids.unsqueeze(0), labels.unsqueeze(0), prompt_len


def pad_labels_for_soft_prompt(labels: torch.Tensor):
    pad = torch.full((labels.size(0), 1), -100, device=labels.device, dtype=labels.dtype)
    return torch.cat([pad, labels], dim=1)


def compute_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )



# Latent Memory Model
class DualLoraMemoryQA(nn.Module):
    def __init__(self, peft_lm: PeftModelForCausalLM, tokenizer, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.lm = peft_lm
        self.tokenizer = tokenizer

        if hasattr(peft_lm, "get_base_model"):
            core = peft_lm.get_base_model()
        else:
            core = peft_lm.base_model

        for _ in range(3):
            if hasattr(core, "model") and not hasattr(core, "layers") and hasattr(core.model, "layers"):
                break
            if hasattr(core, "model") and hasattr(core.model, "model"):
                core = core.model
            else:
                break

        self.config = core.config

        if hasattr(core, "model") and hasattr(core.model, "layers"):
            self.blocks = core.model.layers
            self.embed_tokens = core.model.embed_tokens
            self.norm = getattr(core.model, "norm", None)
            self.rotary = getattr(core.model, "rotary_emb", None)

        elif hasattr(core, "layers"):
            self.blocks = core.layers
            self.embed_tokens = core.embed_tokens
            self.norm = getattr(core, "norm", None)
            self.rotary = getattr(core, "rotary_emb", None)

        elif hasattr(core, "transformer") and hasattr(core.transformer, "h"):
            self.blocks = core.transformer.h
            self.embed_tokens = core.transformer.wte
            self.norm = getattr(core.transformer, "ln_f", None)
            self.rotary = getattr(core.transformer, "rotary_emb", None)

        else:
            raise ValueError(
                f"Unsupported base model architecture for manual stepping. "
                f"Core type={type(core)}; attrs={dir(core)[:50]}"
            )

        self.n_layers = len(self.blocks)
        self.hidden_size = self.config.hidden_size

        self.lm_head = getattr(core, "lm_head", None)
        if self.lm_head is None:
            raise ValueError("Missing lm_head on base model.")

        self.soft_prompt_ae = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.soft_prompt_ft = nn.Parameter(torch.zeros(1, 1, self.hidden_size))


    def set_adapter(self, name: str):
        self.lm.set_adapter(name)


    def _build_mem_cross_attn_mask(
        self,
        mem_len: int,
        base_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        total = mem_len + base_len
        min_val = torch.finfo(dtype).min
        mask = torch.full((total, total), min_val, device=device, dtype=dtype)

        if mem_len > 0:
            idx = torch.arange(mem_len, device=device)
            mask[idx, idx] = 0

        if mem_len > 0:
            mask[mem_len:, :mem_len] = 0

        real_causal = torch.tril(torch.ones((base_len, base_len), device=device, dtype=torch.bool))
        mask[mem_len:, mem_len:] = torch.where(
            real_causal,
            torch.zeros_like(mask[mem_len:, mem_len:]),
            mask[mem_len:, mem_len:]
        )
        return mask.unsqueeze(0).unsqueeze(1)


    def encode_contexts(self, contexts: List[str]):
        self.set_adapter("encoder")
        memory_per_layer = [[] for _ in range(self.n_layers)]

        for ctx in contexts:
            ids = self.tokenizer(ctx, return_tensors="pt").input_ids.to(self.cfg.device)
            out = self.lm(
                input_ids=ids,
                output_hidden_states=True,
                use_cache=False
            )
            hs = out.hidden_states  # len = n_layers+1

            for l in range(self.n_layers):
                layer_h = hs[l + 1][0]      # (seq, h)
                k_states = layer_h[-self.cfg.K:]  # (K, h)
                memory_per_layer[l].append(k_states)


        memory_pools = []
        for l in range(self.n_layers):
            if len(memory_per_layer[l]) == 0:
                memory_pools.append(None)
            else:
                memory_pools.append(torch.cat(memory_per_layer[l], dim=0))  # (num_ctx*K, h)

        return memory_pools


    def forward_with_memory(
        self,
        input_ids: torch.Tensor,
        memory_pools: List[torch.Tensor],
        soft_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.set_adapter("decoder")
        device = self.cfg.device

        def _ensure_batched(x):
            return x.unsqueeze(0) if x.dim() == 2 else x

        tok_emb = _ensure_batched(self.embed_tokens(input_ids))  # (1, base_len, h)
        base_dtype = tok_emb.dtype

        if soft_prompt is not None:
            soft = soft_prompt.to(device=device, dtype=base_dtype)
            tok_emb = torch.cat([soft, tok_emb], dim=1)

        base_len = tok_emb.size(1)
        keep_len = base_len
        hidden_states = tok_emb

        for l, block in enumerate(self.blocks):
            hidden_states = _ensure_batched(hidden_states)
            current_states = hidden_states[:, -keep_len:, :]

            mem_l = memory_pools[l]
            if mem_l is not None and mem_l.numel() > 0:
                mem_l = mem_l.to(device=device, dtype=base_dtype).unsqueeze(0)
                hidden_states = torch.cat([mem_l, current_states], dim=1)
                mem_len = mem_l.size(1)
            else:
                hidden_states = current_states
                mem_len = 0

            seq_len = hidden_states.size(1)
            attention_mask = self._build_mem_cross_attn_mask(
                mem_len=mem_len,
                base_len=base_len,
                device=device,
                dtype=base_dtype
            )

            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            position_embeddings = None
            if self.rotary is not None:
                try:
                    position_embeddings = self.rotary(hidden_states, position_ids)
                except TypeError:
                    position_embeddings = self.rotary(position_ids, seq_len=seq_len)

            block_out = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False
            )

            hidden_states = _ensure_batched(block_out[0]).to(dtype=base_dtype)

        if self.norm is not None:
            hidden_states = _ensure_batched(self.norm(hidden_states)).to(dtype=base_dtype)
        final_states = hidden_states[:, -base_len:, :]
        logits = self.lm_head(final_states)
        return logits




def build_dual_lora_model(base_lm, cfg: TrainConfig):
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    peft_model = get_peft_model(base_lm, lora_cfg, adapter_name="encoder")
    peft_model.add_adapter("decoder", lora_cfg)

    for _, p in peft_model.base_model.named_parameters():
        p.requires_grad = False

    peft_model.set_adapter("decoder")
    return peft_model



def save_dual_lora(peft_model, mem_model: DualLoraMemoryQA, cfg):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    peft_model.save_pretrained(
        cfg.ckpt_dir,
        selected_adapters=["encoder", "decoder"]
    )

    with open(os.path.join(cfg.ckpt_dir, "base_model.txt"), "w") as f:
        f.write(cfg.model_name)

    torch.save(
        {
            "soft_prompt_ae": mem_model.soft_prompt_ae.detach().cpu(),
            "soft_prompt_ft": mem_model.soft_prompt_ft.detach().cpu(),
        },
        os.path.join(cfg.ckpt_dir, "soft_prompts.pt")
    )


def load_soft_prompts(mem_model: DualLoraMemoryQA, cfg):
    path = os.path.join(cfg.ckpt_dir, "soft_prompts.pt")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        mem_model.soft_prompt_ae.data.copy_(ckpt["soft_prompt_ae"])
        mem_model.soft_prompt_ft.data.copy_(ckpt["soft_prompt_ft"])
    else:
        print("[warn] soft_prompts.pt not found; using fresh soft prompts.")


from peft import PeftModelForCausalLM

def load_dual_lora(cfg, device=None):
    device = device or cfg.device

    base_lm = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None
    ).to(device)

    peft_model = PeftModelForCausalLM.from_pretrained(
        base_lm,
        os.path.join(cfg.ckpt_dir, "encoder"),
        adapter_name="encoder"
    )
    peft_model.load_adapter(
        os.path.join(cfg.ckpt_dir, "decoder"),
        adapter_name="decoder"
    )

    peft_model.to(device)
    return peft_model





# Train / Eval
@torch.no_grad()
def generate_answer(model: DualLoraMemoryQA, contexts, question, cfg):
    model.eval()
    memory_pools = model.encode_contexts(contexts)

    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(cfg.device)

    input_ids = prompt_ids
    for _ in range(cfg.max_new_tokens):
        logits = model.forward_with_memory(
            input_ids,
            memory_pools,
            soft_prompt=model.soft_prompt_ft
        )
        next_id = int(torch.argmax(logits[0, -1]).item())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=cfg.device)], dim=1)
        if cfg.eos_token_id is not None and next_id == cfg.eos_token_id:
            break

    gen_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    gen = gen_text.split("Answer:")[-1].strip() if "Answer:" in gen_text else gen_text.strip()
    return gen, memory_pools[0].size(0)


def evaluate(model: DualLoraMemoryQA, df, tokenizer, cfg):
    model.eval()
    losses = []

    for i in range(len(df)):
        row = df.iloc[i]
        contexts, q, a = row["context"], row["question"], row["answer"]

        memory_pools = model.encode_contexts(contexts)

        input_ids, labels, _ = tokenize_qa(tokenizer, q, a)
        input_ids, labels = input_ids.to(cfg.device), labels.to(cfg.device)

        labels = pad_labels_for_soft_prompt(labels)
        logits = model.forward_with_memory(input_ids, memory_pools, soft_prompt=model.soft_prompt_ft)

        loss = compute_loss(logits, labels)
        losses.append(loss.item())

    return float(sum(losses) / max(1, len(losses)))


def train_and_test(df: pd.DataFrame, cfg: TrainConfig):
    set_seed(cfg.seed)
    random.seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.eos_token_id = tokenizer.eos_token_id

    base_lm = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None
    ).to(cfg.device)
    base_lm.eval()

    peft_lm = build_dual_lora_model(base_lm, cfg).to(cfg.device)
    peft_lm = peft_lm.to(dtype=torch.bfloat16)

    mem_model = DualLoraMemoryQA(peft_lm, tokenizer, cfg).to(cfg.device)

    train_df, val_df, test_df = make_splits(df, cfg)

    opt = torch.optim.AdamW(
        [p for p in mem_model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    all_contexts = []
    for i in range(len(train_df)):
        all_contexts.extend(train_df.iloc[i]["context"])



    best_val = float("inf")
    patience_ctr = 0
    global_step = 0

    for epoch in range(cfg.max_epochs):
        is_pretrain = epoch < cfg.pretrain_epochs
        stage_name = "RECON-AE" if is_pretrain else "QA-FT"
        print(f"\nEpoch {epoch+1}/{cfg.max_epochs} [{stage_name}]")

        mem_model.train()
        opt.zero_grad()
        epoch_losses = []
        window_losses = []
        all_ctx_lens = []
        all_sample_lens = []
        for i in tqdm(range(len(train_df))):
            row = train_df.iloc[i]
            contexts, q, a = row["context"], row["question"], row["answer"]
            ctx_len = []
            for ctx in contexts:
                len_ids = len(tokenizer(ctx)['input_ids'])
                ctx_len.append(len_ids)
            all_ctx_lens.extend(ctx_len)
            all_sample_lens.append(np.sum(ctx_len))
            if is_pretrain:
                # AE stage: one context at a time
                per_ctx_losses = []

                for ctx in contexts:
                    memory_pools = mem_model.encode_contexts([ctx])

                    input_ids, labels, _ = tokenize_reconstruction(tokenizer, ctx)
                    input_ids, labels = input_ids.to(cfg.device), labels.to(cfg.device)

                    labels = pad_labels_for_soft_prompt(labels)
                    logits = mem_model.forward_with_memory(
                        input_ids,
                        memory_pools,
                        soft_prompt=mem_model.soft_prompt_ae
                    )

                    loss_ctx = compute_loss(logits, labels)
                    per_ctx_losses.append(loss_ctx)

                unscaled_loss = torch.stack(per_ctx_losses).mean()

            else:
                # FT stage: normal QA from full memories
                memory_pools = mem_model.encode_contexts(contexts)

                input_ids, labels, _ = tokenize_qa(tokenizer, q, a)
                input_ids, labels = input_ids.to(cfg.device), labels.to(cfg.device)

                labels = pad_labels_for_soft_prompt(labels)
                logits = mem_model.forward_with_memory(
                    input_ids,
                    memory_pools,
                    soft_prompt=mem_model.soft_prompt_ft
                )

                unscaled_loss = compute_loss(logits, labels)

            (unscaled_loss / cfg.grad_accum_steps).backward()

            epoch_losses.append(float(unscaled_loss.item()))
            window_losses.append(float(unscaled_loss.item()))
            global_step += 1

            if global_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(peft_lm.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

                print(f"  step={global_step} loss={sum(window_losses)/len(window_losses):.4f}")
                window_losses = []

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))


        val_loss = evaluate(mem_model, val_df, tokenizer, cfg)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            patience_ctr = 0
            print("  New best val. Saving adapters + soft prompts.")
            save_dual_lora(peft_lm, mem_model, cfg)
        else:
            patience_ctr += 1
            print(f"  No improvement. Patience {patience_ctr}/{cfg.patience}")
            if patience_ctr >= cfg.patience:
                print("Early stopping.")
                break
    return mem_model



def build_ctx(example):
    context = example["context"]
    valid_titles = example["supporting_facts"]["title"]
    outs = []
    sents = []
    for title, sentence_list in zip(context["title"], context["sentences"]):
        sents.extend(sentence_list)
    for i in range(0, len(sents), 4):
        outs.append(" ".join(sents[i:i+4]))
    return outs


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        login(token=hf_token)

    cfg = TrainConfig()

    ds = load_dataset("hotpot_qa", "fullwiki")
    df = ds["train"].to_pandas()
    df = df[:7500]

    df["context"] = df.apply(lambda row: build_ctx(row), axis=1)


    df = df[["question", "answer", "context"]].reset_index(drop=True)

    train_and_test(df, cfg)
