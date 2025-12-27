from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset
import torch
import random
import pandas as pd
import numpy as np
import re, string
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt

from train_memformer import build_ctx, generate_answer, load_dual_lora, DualLoraMemoryQA, load_soft_prompts


# Evaluation metrics/helpers
def normalize_text(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(pred, truth):
    return normalize_text(pred) == normalize_text(truth)

def f1_score(pred, truth):
    pred_tokens = normalize_text(pred).split()
    gt_tokens = normalize_text(truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / max(1, len(pred_tokens))
    recall = len(common) / max(1, len(gt_tokens))
    return 2 * precision * recall / (precision + recall)


# Config
@dataclass
class TestConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    K: int = 16
    max_new_tokens: int = 5
    eos_token_id: Optional[int] = None
    ckpt_dir: str = "dual_lora_ckpt"


# Create groupings of samples across different context-length windows
def make_clusters_banded(
    df,
    k=10,
    max_cluster_size=10,
    trim_pct=0.05,
):
    df = df.copy()

    def combined_text(row):
        ctx = " ".join(row["context"]) if isinstance(row["context"], list) else str(row["context"])
        return (row["question"] or "") + " " + ctx

    df["combined_text"] = df.apply(combined_text, axis=1)
    df["str_len"] = df["combined_text"].apply(len)

    df_sorted = df.sort_values("str_len").reset_index(drop=True)

    n = len(df_sorted)
    lo_i = int(np.floor(trim_pct * n))
    hi_i = int(np.ceil((1.0 - trim_pct) * n))
    df_filt = df_sorted.iloc[lo_i:hi_i].reset_index(drop=True)

    lengths = df_filt["str_len"].to_numpy()
    L_min, L_max = lengths.min(), lengths.max()

    step = (L_max - L_min) / k if k > 0 else 1.0
    centroids = [L_min + (j + 0.5) * step for j in range(k)]

    bounds = [L_min] + [(centroids[j] + centroids[j+1]) / 2 for j in range(k-1)] + [L_max + 1]

    clusters = []
    for j in range(k):
        lo, hi = bounds[j], bounds[j+1]
        band_idx = np.where((lengths >= lo) & (lengths < hi))[0].tolist()
        band_idx.sort(key=lambda i: abs(lengths[i] - centroids[j]))
        band_idx = band_idx[:max_cluster_size]
        clusters.append(sorted(band_idx))

    return df_filt, clusters, centroids


# Load data
cfg = TestConfig()
out_csv_path = "test_eval_outputs.csv"
plot_path = "f1_vs_length_compare.png"

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
if cfg.eos_token_id is None:
    cfg.eos_token_id = tokenizer.eos_token_id

ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
df = ds.to_pandas()

df["context"] = df.apply(lambda row: build_ctx(row), axis=1)
df = df[["question", "answer", "context"]].reset_index(drop=True)


df_sorted, clusters, centroids = make_clusters_banded(
    df, k=8, max_cluster_size=150, trim_pct=0.01
)

selected_positions = [p for cluster in clusters for p in cluster]
eval_df = df_sorted.iloc[selected_positions].reset_index(drop=True)

def tok_len(text):
    return len(tokenizer(text, return_tensors="pt").input_ids[0])

eval_df["tok_len"] = eval_df["combined_text"].apply(tok_len)
df_sorted.loc[selected_positions, "tok_len"] = eval_df["tok_len"].values

print("Centroids / bin stats:")
for gi, cluster in enumerate(clusters):
    sub = df_sorted.iloc[cluster]
    if len(sub) == 0:
        print(f"group {gi}: EMPTY bin near centroid {centroids[gi]:.1f}")
        continue
    print(
        f"group {gi}: centroid={centroids[gi]:.1f}, "
        f"n={len(sub)}, str_avg={sub['str_len'].mean():.1f}, "
        f"str_min={sub['str_len'].min()}, str_max={sub['str_len'].max()}, "
        f"tok_avg={sub['tok_len'].mean():.1f}"
    )


### Load models ###
# Base LM baseline
base_lm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=None
).to(cfg.device)
base_lm.eval()

# Memory model
peft_lm = load_dual_lora(cfg)
mem_model = DualLoraMemoryQA(peft_lm, tokenizer, cfg).to(cfg.device)
load_soft_prompts(mem_model, cfg)
mem_model.eval()

set_seed(cfg.seed)
rng = random.Random(cfg.seed)


# Run eval
outputs = []

group_f1s_mem = []
group_f1s_base = []
group_tok_avgs = []
group_mem_tok = []
all_f1_mem, all_f1_base = [], []
all_em_mem, all_em_base = [], []

for g, cluster in enumerate(clusters):
    if len(cluster) == 0:
        group_f1s_mem.append(np.nan)
        group_f1s_base.append(np.nan)
        group_tok_avgs.append(np.nan)
        continue

    group_rows = df_sorted.iloc[cluster]

    f1_list_mem, f1_list_base = [], []
    em_list_mem, em_list_base = [], []
    tok_lens = []
    mem_tok_lens = []
    for i in range(len(group_rows)):
        row = group_rows.iloc[i]
        contexts = row["context"]
        q = row["question"]
        a = row["answer"]

        if q is None or a is None or not contexts:
            continue

        try:
            pred_mem, len_mem = generate_answer(mem_model, contexts, q, cfg)
        except:
            continue
        em_mem = exact_match(pred_mem, a)
        f1_mem = f1_score(pred_mem, a)
        
        full_ctx = " ".join(contexts)
        full_prompt = f"Respond to the question in only one or two words using the following context:\n{full_ctx}\nQuestion: {q}\nAnswer:"
        model_inputs = tokenizer([full_prompt], return_tensors="pt").to(cfg.device)
        generated_ids = base_lm.generate(
            model_inputs.input_ids,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
        )
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        pred_base = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        em_base = exact_match(pred_base, a)
        f1_base = f1_score(pred_base, a)

        f1_list_mem.append(f1_mem); f1_list_base.append(f1_base)
        em_list_mem.append(em_mem); em_list_base.append(em_base)

        all_f1_mem.append(f1_mem); all_f1_base.append(f1_base)
        all_em_mem.append(em_mem); all_em_base.append(em_base)

        tok_lens.append(float(row["tok_len"]))
        mem_tok_lens.append(float(len_mem))
        outputs.append({
            "group": g,
            "question": q,
            "answer": a,
            "pred_memformer": pred_mem,
            "pred_base": pred_base,
            "f1_memformer": f1_mem,
            "f1_base": f1_base,
            "em_memformer": em_mem,
            "em_base": em_base,
            "token_len": float(row["tok_len"]),
            "str_len": int(row["str_len"]),
        })

    group_f1s_mem.append(float(np.mean(f1_list_mem)) if f1_list_mem else np.nan)
    group_f1s_base.append(float(np.mean(f1_list_base)) if f1_list_base else np.nan)
    group_tok_avgs.append(float(np.mean(tok_lens)) if tok_lens else np.nan)
    group_mem_tok.append(float(np.mean(mem_tok_lens)) if mem_tok_lens else np.nan)
    print(
        f"group {g}: n={len(f1_list_mem)}, "
        f"avg_tok_len={group_tok_avgs[-1]:.1f}, "
        f"avg_tok_len_compressed={group_mem_tok[-1]:.1f}, "
        f"mem_f1={group_f1s_mem[-1]*100:.2f}%, "
        f"base_f1={group_f1s_base[-1]*100:.2f}%"
    )


# Save outputs
out_df = pd.DataFrame(outputs)
out_df.to_csv(out_csv_path, index=False)
print(f"Saved predictions to {out_csv_path}")

overall_em_mem = 100 * (sum(all_em_mem) / max(1, len(all_em_mem)))
overall_f1_mem = 100 * (sum(all_f1_mem) / max(1, len(all_f1_mem)))

overall_em_base = 100 * (sum(all_em_base) / max(1, len(all_em_base)))
overall_f1_base = 100 * (sum(all_f1_base) / max(1, len(all_f1_base)))
overall_tok_len = np.mean(group_tok_avgs)
overall_mem_len = np.mean(group_mem_tok)
print("\n===============================")
print("Evaluation Results")
print("===============================")
print(f"Latent Memory LM EM: {overall_em_mem:.2f}% | F1: {overall_f1_mem:.2f}%")
print(f"Base LM EM:   {overall_em_base:.2f}% | F1: {overall_f1_base:.2f}%")
print(f"Latent Memory LM Average CTX size: {overall_mem_len:.2f}")
print(f"Baseline Average CTX size: {overall_tok_len:.2f}")
print(f"Compression Rate: {(overall_tok_len/overall_mem_len):.2f}")

# Plot avg F1 vs avg token length (both models)
xs, ys_mem, ys_base = [], [], []
for x, y_m, y_b in zip(group_tok_avgs, group_f1s_mem, group_f1s_base):
    if np.isnan(x) or np.isnan(y_m) or np.isnan(y_b):
        continue
    xs.append(x)
    ys_mem.append(y_m * 100)
    ys_base.append(y_b * 100)

plt.figure()
plt.plot(xs, ys_mem, marker="o", label="Memory-Augmented LLM")
plt.plot(xs, ys_base, marker="o", label="Baseline LLM In-Context")
plt.xlabel("Context Token Length")
plt.ylabel("F1 Score")
plt.title("F1 vs Token Length")
plt.grid(True)
plt.legend()
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Saved plot to {plot_path}")
