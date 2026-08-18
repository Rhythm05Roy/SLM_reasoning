"""Generate curricsym_week1.ipynb — the Week-1 Kaggle pipeline.

Kept as a generator script so the notebook source stays reviewable in git
and diffs stay readable (ipynb JSON does not diff well).

Run:  python scripts/build_notebook.py
"""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# CurricSym — Week 1 Pipeline (Kaggle T4)

**Purpose.** Get a *correct* pipeline running on free Kaggle GPU, and run the one
experiment that decides what the paper can claim.

**Order matters.** Section 4 (pass@k) is the decisive experiment and needs **no
training**. Run it before investing quota in GRPO. If the base model's coverage at
large *k* dominates the RL model, the Yue et al. (arXiv 2504.13837, NeurIPS 2025
Oral) critique applies directly and the paper must be framed as measurement, not
as a performance gain.

**Hardware.** Select **GPU T4 × 2** in the Kaggle sidebar. **Not P100** — Pascal is
sm_60, below both Unsloth's sm_70 and vLLM's sm_75 floors.

### What this notebook deliberately does *not* contain

The prior codebase claimed four mechanisms that were absent or inert. They are
omitted here on purpose, not forgotten:

| Omitted | Why |
|---|---|
| Adaptive difficulty curriculum | Occupied by AdaRFT (2504.05520, TMLR); the old implementation saturated on step 1 and never selected data |
| Process Reward Model | Occupied by FoVer (2505.15960) for label provenance and VeriGate (2605.30451) for GRPO integration |
| Internalization *loss* | Was never a loss — only a reward. Dead config (`internalization_weight`) removed |
| Z3 "symbolic verification" | Was tautological: asserted `x==pred ∧ x!=gold` and reported unsat, i.e. `pred==gold` via a solver. Replaced with an honest exact-match checker |

Real symbolic step-level verification belongs to the benchmark contribution (C1),
which is CPU work and lives outside this notebook.
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## 1. Environment check\n\nRun this first. If capability is `(6, 0)` you are on a P100 — switch to T4 ×2 and restart.")

code(r"""
import subprocess, sys, torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Enable an accelerator in the Kaggle sidebar.")

n = torch.cuda.device_count()
print("device count:", n)
for i in range(n):
    cap = torch.cuda.get_device_capability(i)
    print(f"  [{i}] {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}  "
          f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

cap0 = torch.cuda.get_device_capability(0)
print("bf16 supported:", torch.cuda.is_bf16_supported(), "(expect False on T4)")

if cap0 < (7, 5):
    raise SystemExit(
        f"sm_{cap0[0]}{cap0[1]} is below vLLM's sm_75 floor. "
        "Select 'GPU T4 x2' in the sidebar, not P100."
    )
print("\nOK — T4-class or better.")
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
""")

md("""
**Also note your remaining weekly GPU quota** from the sidebar now — the four-week
schedule is built against that number, and it is the one figure the feasibility
research could not verify from outside.
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 2. Install

Kaggle's image ships `transformers` but **no** `trl`, `vllm`, `bitsandbytes`, or
`peft`. Versions are pinned deliberately:

- `vllm==0.26.0` — TRL 1.10 hard-pins `vllm>=0.17.0,<=0.26.0`; PyPI's latest (0.27.1) is **unsupported**
- **No Unsloth** — it is a Triton library (Triton needs sm_80+), it patches
  transformers/TRL aggressively, and it imposes an sm_70 floor for ~no gain on Turing
""")

code(r"""
%pip install -q "trl==1.10.0" "vllm==0.26.0" "peft>=0.20.0" "bitsandbytes>=0.50.1" datasets
print("done — if vllm pulled a different torch, restart the kernel before continuing")
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## 3. Config, seeding, verifier, data\n\nAll experiment knobs in one place.")

code(r"""
import gc, hashlib, json, math, os, random, re, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch

SEED = 42

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed()

@dataclass
class Cfg:
    # model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_prompt_tokens: int = 256      # pre-truncated: GRPOConfig.max_prompt_length is GONE
    max_completion_length: int = 512

    # GRPO (T4-safe)
    num_generations: int = 8
    per_device_train_batch_size: int = 2   # logits dominate memory (~1.45 GB / 1024 tok)
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    max_steps: int = 150
    lora_r: int = 16
    lora_alpha: int = 32

    # pass@k
    passk_n_problems: int = 100
    passk_n_samples: int = 64
    passk_temperature: float = 1.0
    passk_ks: tuple = (1, 2, 4, 8, 16, 32, 64)

    # data
    n_train: int = 2000
    n_eval: int = 300
    include_fol: bool = False   # enable only after inspecting the schema (Section 3c)

    # vllm
    vllm_gpu_util: float = 0.60   # standalone pass@k; GRPO colocate uses 0.30

    out_dir: str = "/kaggle/working/results"

cfg = Cfg()
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
print(json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2))
""")

md(r"""
### 3a. Verifier

Honest scope: this is an **exact-match answer checker** with numeric tolerance. It is
*not* symbolic verification and is not described as such. The tautological Z3 wrapper
from the previous codebase (`assert x==pred ∧ x!=gold; check unsat`) is removed — it
computed `pred == gold` at the cost of a solver call.
""")

code(r"""
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

def extract_gsm_answer(ans: str) -> Optional[str]:
    "GSM8K/GSM-Symbolic gold answers end with '#### <value>'."
    if "####" not in ans:
        return None
    return ans.split("####")[-1].strip().replace(",", "")

def extract_pred(completion: str) -> Optional[str]:
    "Prefer an explicit <answer> tag; else fall back to the last number."
    m = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if m:
        cand = m.group(1).strip()
        nums = NUM_RE.findall(cand)
        if nums:
            return nums[-1].replace(",", "")
        return cand
    nums = NUM_RE.findall(completion)
    return nums[-1].replace(",", "") if nums else None

def check_numeric(pred: Optional[str], gold: Optional[str], tol=1e-4) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) <= tol * max(1.0, abs(float(gold)))
    except ValueError:
        return pred.strip().lower() == gold.strip().lower()

def is_correct(completion: str, gold: str) -> bool:
    return check_numeric(extract_pred(completion), gold)

# sanity
assert is_correct("<answer>72</answer>", "72")
assert is_correct("... so the answer is 1,024", "1024")
assert not is_correct("<answer>71</answer>", "72")
print("verifier OK")
""")

md(r"""
### 3b. Data — and the contamination fixes

Three correctness bugs from the previous pipeline are fixed here:

1. **Do not train on GSM-Symbolic.** It is Apple's *evaluation* benchmark and ships
   only a `test` split; the old loader trained on it. We train on **GSM8K `train`**
   and reserve GSM-Symbolic for evaluation.
2. **OOD must be disjoint.** The old OOD set was 100% contained in training data.
   Here GSM-Symbolic `p1`/`p2` are eval-only, so disjointness is structural.
3. **Group by template, not by row.** GSM-Symbolic is ~100 templates × 50
   instantiations; a random row split leaks templates across train/eval.
""")

code(r"""
from datasets import load_dataset

SYS = (
    "You are a careful reasoner. Think step by step inside <thinking></thinking>, "
    "then give only the final answer inside <answer></answer>."
)

def build_prompt(question: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": question}]

# ---- TRAIN: GSM8K train split ------------------------------------------------
gsm8k = load_dataset("openai/gsm8k", "main", split="train")
train_rows = []
for ex in gsm8k.select(range(min(cfg.n_train, len(gsm8k)))):
    gold = extract_gsm_answer(ex["answer"])
    if gold is None:
        continue
    train_rows.append({"question": ex["question"], "gold": gold, "src": "gsm8k"})
print(f"train (GSM8K): {len(train_rows)}")

# ---- EVAL: GSM-Symbolic, template-grouped ------------------------------------
def load_gsm_symbolic(variant: str, limit: int):
    ds = load_dataset("apple/GSM-Symbolic", variant, split="test")
    rows = []
    for ex in ds:
        gold = extract_gsm_answer(ex["answer"])
        if gold is None:
            continue
        # template id: prefer an explicit field, else derive from the number-masked
        # question. Uses blake2b, NOT hash() — Python string hashing is salted per
        # process, so hash() would give different splits on every session.
        tid = ex.get("template_id", ex.get("id", None))
        if tid is None:
            masked = re.sub(r"-?\d[\d,]*\.?\d*", "#", ex["question"])
            tid = hashlib.blake2b(masked.encode(), digest_size=8).hexdigest()
        rows.append({"question": ex["question"], "gold": gold,
                     "src": f"gsm_symbolic_{variant}", "template": str(tid)})
    # keep at most one instantiation per template -> no near-duplicates in eval
    seen, dedup = set(), []
    for r in rows:
        if r["template"] in seen:
            continue
        seen.add(r["template"]); dedup.append(r)
    random.Random(SEED).shuffle(dedup)
    return dedup[:limit]

eval_id  = load_gsm_symbolic("main", cfg.n_eval)     # in-distribution difficulty
eval_ood = load_gsm_symbolic("p1",   cfg.n_eval)     # harder variant = OOD shift
print(f"eval  ID (main): {len(eval_id)} unique templates")
print(f"eval OOD (p1)  : {len(eval_ood)} unique templates")

overlap = {r["question"] for r in train_rows} & {r["question"] for r in eval_id + eval_ood}
assert not overlap, f"train/eval overlap: {len(overlap)}"
print("no train/eval overlap")
""")

md(r"""
### 3c. FOL (optional, off by default)

The previous loader silently fell back to **5 hardcoded templates, 80% majority
class**, on any load exception — which quietly produces a meaningless dataset. That
fallback is removed. This cell only *inspects* the schema and fails loudly. Set
`cfg.include_fol = True` after you have seen the columns and written a real adapter.
""")

code(r"""
if cfg.include_fol:
    ds = load_dataset("tasksource/proofwriter", split="train")
    print("columns:", ds.column_names)
    print("first row:", {k: str(v)[:200] for k, v in ds[0].items()})
    raise NotImplementedError(
        "Inspect the columns above, then write an explicit adapter. "
        "Do NOT add a synthetic fallback."
    )
else:
    print("FOL disabled. Math-only for Week 1 (see PAPER_PLAN.md §7 decision 3).")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 4. Experiment 1 — pass@k on the base model  ⭐ **run this first**

This is the experiment that decides the paper, and it requires **no training**.

We sample *n* completions per problem at temperature 1.0 and compute the unbiased
pass@k estimator from Chen et al. (2107.03374):

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

**How to read the result.** After GRPO (Section 6) you re-run this on the trained
adapter and overlay the curves. If the base model catches up or overtakes at large
*k*, then RL sharpened the distribution without expanding capability — the central
claim of Yue et al. That is a publishable finding, but it forces a measurement
framing rather than a performance claim.
""")

code(r"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(cfg.model_name)

def render(rows):
    return [tok.apply_chat_template(build_prompt(r["question"]),
                                    tokenize=False, add_generation_prompt=True)
            for r in rows]

def pass_at_k(n: int, c: int, k: int) -> float:
    "Unbiased estimator; 1 - C(n-c,k)/C(n,k)."
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

MAX_LEN = cfg.max_prompt_tokens + cfg.max_completion_length + 64   # chat-template margin

def make_llm(model_path, lora_path=None, gpu_util=None):
    "Only pass LoRA kwargs when LoRA is actually used — vLLM rejects max_lora_rank otherwise."
    kw = {}
    if lora_path is not None:
        kw.update(enable_lora=True, max_lora_rank=cfg.lora_r)
    return LLM(model=model_path, dtype="float16", max_model_len=MAX_LEN,
               gpu_memory_utilization=gpu_util or cfg.vllm_gpu_util, seed=SEED, **kw)

def lora_kwargs(lora_path):
    if lora_path is None:
        return {}
    from vllm.lora.request import LoRARequest
    return {"lora_request": LoRARequest("adapter", 1, lora_path)}

def run_passk(model_path: str, rows, tag: str, lora_path: Optional[str] = None):
    llm = make_llm(model_path, lora_path)
    # NO seed here. With n>1 a fixed seed can collapse the samples toward each other,
    # which would silently flatten the pass@k curve — the one number this experiment
    # exists to measure. Diversity across the n samples is the point.
    sp = SamplingParams(n=cfg.passk_n_samples, temperature=cfg.passk_temperature,
                        top_p=1.0, max_tokens=cfg.max_completion_length)
    kw = lora_kwargs(lora_path)

    t0 = time.time()
    outs = llm.generate(render(rows), sp, **kw)
    print(f"  generated in {time.time()-t0:.0f}s")

    correct_counts, n_eff = [], []
    for row, out in zip(rows, outs):
        correct_counts.append(sum(is_correct(o.text, row["gold"]) for o in out.outputs))
        n_eff.append(len(out.outputs))       # guard: vLLM may return fewer than n

    n = min(n_eff)
    if n < cfg.passk_n_samples:
        print(f"  WARNING: got only {n} samples for some prompts "
              f"(asked {cfg.passk_n_samples}); capping k at {n}")
    curve = {k: float(np.mean([pass_at_k(ne, c, k)
                               for c, ne in zip(correct_counts, n_eff)]))
             for k in cfg.passk_ks if k <= n}
    solved_any = sum(c > 0 for c in correct_counts)

    del llm; gc.collect(); torch.cuda.empty_cache()
    return {"tag": tag, "n_samples": n, "n_problems": len(rows),
            "curve": {str(k): v for k, v in curve.items()},
            "solved_at_least_once": solved_any,
            "correct_counts": correct_counts,
            "questions": [r["question"] for r in rows]}

subset = eval_id[:cfg.passk_n_problems]
base_passk = run_passk(cfg.model_name, subset, "base")

print("\npass@k — base model")
for k in sorted(int(x) for x in base_passk["curve"]):
    print(f"  k={k:>3}: {base_passk['curve'][str(k)]:.3f}")
print(f"solved at least once: {base_passk['solved_at_least_once']}/{len(subset)}")

with open(f"{cfg.out_dir}/passk_base.json", "w") as f:
    json.dump(base_passk, f, indent=2)
""")

code(r"""
import matplotlib.pyplot as plt

def plot_passk(results, fname="passk.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        ks = sorted(int(k) for k in r["curve"])      # int sort, not lexicographic
        ax.plot(ks, [r["curve"][str(k)] for k in ks], marker="o", label=r["tag"])
    ax.set_xscale("log", base=2); ax.set_xlabel("k (samples)")
    ax.set_ylabel("pass@k"); ax.set_title("Coverage vs. sampling budget")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(f"{cfg.out_dir}/{fname}", dpi=150)
    plt.show()

plot_passk([base_passk])
""")

md("""
> **Decision gate.** Record `pass@1` and `pass@64` for the base model. You will
> compare these against the trained model in Section 7. Do not skip writing them down.
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 5. Rewards

Two rewards only, both honest about what they measure:

- **outcome** — verifier-checked correctness
- **format** — structural compliance with the `<thinking>`/`<answer>` contract

No "process reward." The previous implementation scored
`0.3 if 50 < len(thinking) < 600`, `+0.1` for a word like "therefore", `+0.2` for a
`<verify>` tag — a bag-of-words length heuristic, not process supervision. Real
step-level reward requires the symbolic step checker from contribution C1.
""")

code(r"""
def reward_outcome(completions, gold, **kw):
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [1.0 if is_correct(t, g) else -1.0 for t, g in zip(texts, gold)]

def reward_format(completions, **kw):
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    out = []
    for t in texts:
        s = 0.0
        has_think = "<thinking>" in t and "</thinking>" in t
        has_ans   = "<answer>"   in t and "</answer>"   in t
        s += 0.4 * has_think + 0.4 * has_ans
        if has_think and has_ans and t.index("<thinking>") < t.index("<answer>"):
            s += 0.2
        out.append(s)
    return out

print(reward_outcome([["x"]], ["1"]) if False else "rewards defined")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 6. Experiment 2 — GRPO

### ⚠️ Restart the kernel first

vLLM does not release all device memory when its `LLM` object is deleted, and GRPO
starts its own colocated engine. If Section 4 ran in this kernel, you will OOM here.

**Do this:** *Run → Restart session*, then re-run **cells 1, 3 (config), 3a, 3b** —
not Section 4 — and continue from here. Section 4's results are already on disk in
`results/passk_base.json`, and Section 7 reloads them, so you do not lose them.

Config rationale (each of these is load-bearing on a T4):

| Setting | Why |
|---|---|
| `fp16=True, bf16=False` | Turing has no bf16 |
| `beta=0.0` | Reference model not loaded: −2.9 GB **and** removes the KL term, the worst fp16 numerics hazard in GRPO |
| `num_iterations=1` | Fully on-policy → importance ratio ≡ 1, so fp16 log-prob noise cannot inject spurious ratio gradients |
| `loss_type="dr_grpo"`, `scale_rewards="none"` | Drops division by `std(r)`: a documented difficulty bias *and* a divide-by-small-number fp16 hazard |
| `attn_implementation="sdpa"` | FlashAttention-2 needs sm_80+ |
| `use_liger_kernel=False` | Triton is sm_80+; on Turing it falls back to unoptimized FMA |
| `per_device_train_batch_size=2` | Logits dominate memory at ~1.45 GB / 1024 tokens |
| `save_steps=25` | Sessions get reclaimed; a kill should cost ≤15 min |

**`max_prompt_length` is not passed** — it was removed from `GRPOConfig` in current
TRL, and the old code crashes on it. Prompts are pre-truncated instead.
""")

code(r"""
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# pre-truncate prompts (replaces the removed GRPOConfig.max_prompt_length)
def to_grpo_rows(rows):
    keep = []
    for r in rows:
        ids = tok.apply_chat_template(build_prompt(r["question"]),
                                      tokenize=True, add_generation_prompt=True)
        if len(ids) <= cfg.max_prompt_tokens:
            keep.append({"prompt": build_prompt(r["question"]), "gold": r["gold"]})
    return Dataset.from_list(keep)

train_ds = to_grpo_rows(train_rows)
print(f"train after prompt-length filter: {len(train_ds)} / {len(train_rows)}")

peft_cfg = LoraConfig(
    r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

args = GRPOConfig(
    output_dir="/kaggle/working/grpo",
    fp16=True, bf16=False,
    # transformers 5.x renamed torch_dtype -> dtype (Kaggle ships 5.15).
    # If you see an unexpected-kwarg error, swap "dtype" back to "torch_dtype".
    model_init_kwargs={"attn_implementation": "sdpa", "dtype": "float16"},

    num_generations=cfg.num_generations,
    max_completion_length=cfg.max_completion_length,
    temperature=1.0,

    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,

    beta=0.0,
    num_iterations=1,
    loss_type="dr_grpo",
    scale_rewards="none",
    mask_truncated_completions=True,
    max_grad_norm=1.0,

    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.30,

    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",   # HF's name; "adamw_8bit" is Unsloth-only and will raise
    use_liger_kernel=False,

    learning_rate=cfg.learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.1,
    max_steps=cfg.max_steps,
    logging_steps=5, save_steps=25, save_total_limit=2,
    report_to="none", seed=SEED,
)

trainer = GRPOTrainer(
    model=cfg.model_name,
    args=args,
    train_dataset=train_ds,
    reward_funcs=[reward_outcome, reward_format],
    peft_config=peft_cfg,
)
""")

code(r"""
t0 = time.time()
result = trainer.train()
mins = (time.time() - t0) / 60
print(f"\ntrained {cfg.max_steps} steps in {mins:.1f} min "
      f"({mins*60/cfg.max_steps:.1f} s/step)")

ADAPTER = "/kaggle/working/grpo_adapter"
trainer.save_model(ADAPTER)

hist = [h for h in trainer.state.log_history if "reward" in h]
with open(f"{cfg.out_dir}/grpo_log.json", "w") as f:
    json.dump(hist, f, indent=2)

if hist:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h["step"] for h in hist], [h["reward"] for h in hist])
    ax.set_xlabel("step"); ax.set_ylabel("mean reward")
    ax.set_title("GRPO reward"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{cfg.out_dir}/grpo_reward.png", dpi=150)
    plt.show()
""")

md(r"""
> **Gate.** Does mean reward actually rise? If it is flat after 150 steps, stop and
> diagnose — no amount of additional quota rescues a training signal that never
> moved. Check first: are completions hitting `max_completion_length` (truncation),
> and is `reward_format` saturating while `reward_outcome` stays at −1?
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 7. Experiment 3 — batched evaluation + pass@k comparison

The previous evaluator made ~700 **sequential batch-size-1** `generate` calls, costing
2.2–4.3 h — longer than the training it evaluated. vLLM batches this to a few minutes.

Reported with **Wilson confidence intervals**, not bare point estimates. The old n=20
run reported 4/20 vs 2/20 on *identical* prompts and rows — a 2× spread that was pure
sampling noise. Small evals are fine; uncertainty-free small evals are not.
""")

code(r"""
def wilson(k, n, z=1.96):
    "Wilson score interval — behaves sensibly at small n and near 0/1."
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def evaluate(model_path, rows, tag, lora_path=None, temperature=0.0):
    # temperature=0 (greedy) so accuracy is deterministic. The old pipeline sampled
    # at 0.7 with n=1 and reported 4/20 vs 2/20 on identical rows.
    llm = make_llm(model_path, lora_path)
    sp = SamplingParams(n=1, temperature=temperature, top_p=1.0,
                        max_tokens=cfg.max_completion_length)
    kw = lora_kwargs(lora_path)

    t0 = time.time()
    outs = llm.generate(render(rows), sp, **kw)
    secs = time.time() - t0

    hits = sum(is_correct(o.outputs[0].text, r["gold"]) for o, r in zip(outs, rows))
    fmt  = float(np.mean(reward_format([o.outputs[0].text for o in outs])))
    lo, hi = wilson(hits, len(rows))

    del llm; gc.collect(); torch.cuda.empty_cache()
    res = {"tag": tag, "n": len(rows), "correct": hits,
           "accuracy": hits / len(rows), "ci95": [lo, hi],
           "format_score": fmt, "eval_seconds": round(secs, 1)}
    print(f"{tag:28s} acc {res['accuracy']:.3f}  "
          f"[{lo:.3f},{hi:.3f}]  n={len(rows)}  ({secs:.0f}s)")
    return res

rows_eval = []
for tag, rows, lora in [
    ("base / ID",     eval_id,  None),
    ("base / OOD-p1", eval_ood, None),
    ("grpo / ID",     eval_id,  ADAPTER),
    ("grpo / OOD-p1", eval_ood, ADAPTER),
]:
    rows_eval.append(evaluate(cfg.model_name, rows, tag, lora_path=lora))

with open(f"{cfg.out_dir}/eval.json", "w") as f:
    json.dump(rows_eval, f, indent=2)
""")

code(r"""
# Reload the base run from disk so this cell survives a kernel restart between
# Section 4 and Section 6 (which the Section 6 header tells you to do).
with open(f"{cfg.out_dir}/passk_base.json") as f:
    base_passk = json.load(f)

# Re-derive the SAME problem subset the base run used — comparing pass@k across
# different problem sets would be meaningless.
subset = [{"question": q, "gold": g} for q, g in
          zip(base_passk["questions"],
              [r["gold"] for r in eval_id[:len(base_passk["questions"])]])]
assert [s["question"] for s in subset] == base_passk["questions"], \
    "subset mismatch — re-run Section 3b with the same SEED before comparing"

grpo_passk = run_passk(cfg.model_name, subset, "grpo", lora_path=ADAPTER)
with open(f"{cfg.out_dir}/passk_grpo.json", "w") as f:
    json.dump(grpo_passk, f, indent=2)

plot_passk([base_passk, grpo_passk], fname="passk_base_vs_grpo.png")

b, g = base_passk["curve"], grpo_passk["curve"]
shared = sorted(int(k) for k in set(b) & set(g))
print(f"\n{'k':>4}  {'base':>7}  {'grpo':>7}  {'delta':>7}")
for k in shared:
    bk, gk = b[str(k)], g[str(k)]
    print(f"{k:>4}  {bk:>7.3f}  {gk:>7.3f}  {gk-bk:>+7.3f}")

kmax = max(shared)
bk, gk = b[str(kmax)], g[str(kmax)]
if gk <= bk:
    print(f"\n>>> At k={kmax} the base model matches or beats GRPO ({bk:.3f} vs {gk:.3f}).")
    print(">>> Consistent with Yue et al. (2504.13837): RL sharpened rather than expanded.")
    print(">>> Frame the paper as measurement, not as a performance gain.")
else:
    print(f"\n>>> GRPO retains an advantage at k={kmax} (+{gk-bk:.3f}).")
    print(">>> Coverage expansion is defensible — but confirm with a second seed.")
print("\nNOTE: one seed. Nothing here is publishable until Section 8 items 1-5 are done.")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 8. What to run next

Everything above is one seed. Nothing here is yet a publishable number.

**Required before any claim (all cheap, see PAPER_PLAN.md §4):**

1. **3 seeds** of Sections 6–7, reporting CIs across seeds — reviewers penalize
   single-run results far more than small eval sets
2. **Non-Qwen control** (Llama-3.2-1B). Mandatory after Spurious Rewards
   (2506.10947): random rewards gave +21.4 on Qwen vs +29.1 for real rewards, and
   the effect did not transfer to Llama. Any Qwen-only gain is presumed spurious
3. **Random-reward arm** — replace `reward_outcome` with `lambda *a, **k: [random.choice([-1.,1.]) ...]`
4. **One-example arm** — `train_ds.select(range(1))`, after 1-shot RLVR (2504.20571)
   took Qwen2.5-Math-1.5B from 36.0 → 73.6 on MATH500
5. **Distillation baseline** — Yue et al. explicitly names distillation as the thing
   that expands capability; omitting it is a rejection

**In parallel, on CPU — the primary contribution (C1):** the procedurally generated,
verifier-labeled process-faithfulness benchmark. It needs no GPU, does not compete
for quota, and is the one contribution with no incumbent.

**Housekeeping:** commit `/kaggle/working/results/*.json` back to git with the commit
hash of the code that produced them. Do not reuse anything from `thesis_results/` —
those narrative strings contradict their own numbers.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "curricsym_week1.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
