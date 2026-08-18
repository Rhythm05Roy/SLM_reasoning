# CurricSym → ICLR 2027: Research Plan

**Date:** 2026-08-18 · **Target:** ICLR 2027 (abstract ~Sept 2026) · **Budget:** $0 (Kaggle free tier)

---

## 0. The two findings that determine everything

**Finding 1 — Compute is not the constraint.** Kaggle's free T4×2 runs Qwen2.5-1.5B GRPO at ~11 GB of 16 GB and ~21–42 s/step. A 450-step run finishes in one 12 h session; the 30 h/week quota buys 5–10 runs/week, so a ~22-run matrix fits in four weeks with slack. The original premise of this project — "I need GPUs I don't have" — is false at this scale.

**Finding 2 — Novelty is the constraint.** Three of the proposal's four levers are occupied, two of them by papers published in the last five months:

| Lever | Status | Owner |
|---|---|---|
| Tool-fading / internalization **mechanism** | **Taken** | SKILL0 (2604.02268, **verified**): GRPO, 3 stages, linear decay budget `[6,3,0]`→0, explicitly "internalization." Also SKILLC (2605.27899); Backward Hint Annealing (2604.07747, at 1B–1.7B); TInR (2604.10788, ACL 2026) |
| …the same mechanism applied to a **symbolic verifier** in math/FOL at ≤2B | **OPEN (narrow)** | SKILL0 uses **no** verifier and evaluates **no** math/logic — it is agentic (ALFWorld/Search-QA/WebShop) on Qwen2.5-**VL** 3B/7B. This is a domain transfer, not a duplicate |
| Symbolic-verifier-derived **step labels** | **Taken** | FoVer (2505.15960, ACL 2026 Findings, **verified**): Z3 on FLDx2 + Isabelle on GSM8K/MetaMathQA/Big-Math, 40K steps, plus formal→informal transfer over 12 benchmarks |
| Those labels used as an **RL process reward** at ≤1.5B on logic | **OPEN (narrow, crowded)** | FoVer stops at **Best-of-K reranking** — no policy optimization, and 7–8B models. VeriGate (2605.30451, **verified**) does step→token-level advantages in GRPO at 1.5B/7B but from a **learned neural PRM**, not a symbolic verifier. Neither closes it alone |
| Adaptive difficulty curriculum for RLVR | **Taken** | AdaRFT (2504.05520, TMLR); SEELE (2509.06923); Scaf-GRPO (ICLR 2026) |
| Process-faithfulness evaluation artifact | **OPEN (no incumbent)** | — |

**Caution on the two "narrow open" rows.** They are a two-paper composition (FoVer's labels + VeriGate's integration) in a fast-closing neighbourhood: VPR (2605.10325) converts symbolic verifiers into process-reward oracles for agentic RL; 2607.02869 already ran process-vs-outcome GRPO at 0.5B; "Verifiable PRMs for Structured Reasoning" is in ACL 2026 Findings; VeriBound (2606.20740) is already writing PAC-Bayesian theory for verifier-trained PRMs. Building the paper's *primary* claim here invites exactly the "incremental integration of known techniques" rejection the original proposal warned itself about. Use it as a **secondary** contribution, not the headline.

Plus two structural attacks on the premise: **"GRPO is Secretly a Process Reward Model"** (2509.21154, ICML 2026) proves outcome-reward GRPO is already equivalent to a PRM-aware objective, and **Yue et al.** (2504.13837, NeurIPS 2025 **Oral**) argues RLVR adds no reasoning capacity beyond the base model and that distillation, not RL, is what expands it.

**Conclusion: keep the infrastructure, change the paper type.** Compute abundance means we can afford a real training study — but it must serve a claim that isn't already taken.

---

## 1. The paper

> **Right answers, wrong reasons: a procedurally generated, verifier-labeled benchmark shows RLVR improves small-LM answers without improving step validity.**

Three contributions, in descending order of defensibility:

**C1 — The benchmark (primary).** No existing resource combines all four of:
1. **procedural generation** — unbounded items, difficulty as a dial (PRMBench is static and finite)
2. **machine-checkable ground-truth intermediate steps from a symbolic verifier** — not human annotation (ProcessBench/PRMBench are human/model-labeled)
3. **step-level counterfactual perturbations** in the Lanham (2307.13702) style (GSM-Symbolic perturbs *problems*, not steps)
4. **a step-level faithfulness-to-verifier metric** — does the asserted step actually follow under the formal semantics?

Nearest neighbours are PRMBench (2501.03124) and Reasoning Gym (2505.24760). The one-sentence defense must be: *unbounded generation + verifier-derived (non-human) step labels + step-level counterfactual perturbations.* If that sentence isn't crisp, this reads as "PRMBench with a generator" and dies.

**C2 — The measurement.** Train GRPO properly, then show outcome accuracy rises while step validity does not. This is the finding that gives the benchmark teeth — an artifact paper without a result is a workshop paper.

**C3 — The honest negative on internalization.** Fading verifier access does not transfer the internalization benefit at ≤1.5B, with a mechanism. Two papers (2507.05065, 2606.27281) independently argue internalization is the wrong goal at this scale, so this is likely the true result. The existing n=20 smoke run already showed a *negative* delta.

**Explicitly not claimed:** SOTA accuracy, curriculum efficiency, a novel PRM architecture, 1B–13B scaling, human evaluation.

---

## 2. What must be rebuilt

The audit found the three core mechanisms absent or inert. Since we are no longer claiming them, most need not be built — but the honest ones do.

| Component | Current state | Action |
|---|---|---|
| Curriculum scheduler | Inert: saturates on step 1; difficulty never selects data | **Delete.** Not a claim anymore. |
| Tool fading | Never reaches the prompt (`curriculum_stage` never written non-zero) | **Fix properly** — it's C3's independent variable |
| PRM | Does not exist | **Do not build.** Occupied by FoVer/VeriGate |
| Internalization loss | Reward only; `internalization_weight` is dead config | **Delete the dead config.** Keep as reward for the C3 arm |
| Z3 verifier | Tautological (`x==pred ∧ x!=gold` → unsat) | **Rewrite** — must produce real step-level judgments |
| SFT targets | Fixed string, no reasoning, no `<verify>` tag | **Rewrite** — use the real CoT already extracted into `full_answer` |
| Evaluator | ~700 sequential batch-1 generates (2.2–4.3 h) | **Batch it** (bs=32 → ~10 min) |
| Data splits | Trains on GSM-Symbolic `test`; OOD ⊂ train; templates leak | **Rebuild** — group-split by template |

### Data integrity — non-negotiable
- Stop training on GSM-Symbolic's `test` split. It is an evaluation benchmark.
- Group-split by template, not by row. GSM-Symbolic is ~100 templates × 50 instantiations; a random row split leaks.
- Make the OOD set actually disjoint (currently 100% contained in training data).
- Remove the silent ProofWriter synthetic fallback (5 templates, 80% majority class) — fail loudly instead.

### Delete before writing
`thesis_results/` narrative strings and `generate_tables.py` hardcoded "Finding" text contradict their own numbers. None of it may be reused.

---

## 3. Engineering config (Kaggle T4×2)

```python
# Qwen2.5-1.5B-Instruct · fp16 · TRL 1.10 · ~11 GB / 16 GB · ~21-42 s/step
# pip install "trl==1.10.0" "vllm==0.26.0" peft bitsandbytes
# DROP unsloth (Triton is sm_80+; imposes an sm_70 floor for no gain on Turing)

GRPOConfig(
    fp16=True, bf16=False,                        # T4 has no bf16
    model_init_kwargs={"attn_implementation": "sdpa"},   # FA2 needs sm_80+
    num_generations=8, max_completion_length=512, temperature=1.0,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    beta=0.0,               # no ref model: -2.9 GB AND removes the fp16 KL hazard
    num_iterations=1,       # on-policy; ratio == 1, no fp16 IS noise
    loss_type="dr_grpo", scale_rewards="none",
    mask_truncated_completions=True, max_grad_norm=1.0,
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.30,
    gradient_checkpointing=True, optim="adamw_8bit",
    use_liger_kernel=False,
    learning_rate=5e-6, lr_scheduler_type="cosine", warmup_ratio=0.1,
    max_steps=450, save_steps=25, save_total_limit=2,
)
```

**Known traps:** `GRPOConfig(max_prompt_length=...)` was **removed** — the repo passes it at `grpo_trainer.py:47` and will crash; pre-truncate prompts to 256 tokens instead. `cast_lm_head_to_fp32` is unavailable (all candidate models have `tie_word_embeddings: true`). Always select **T4×2, never P100** (sm_60 is below both Unsloth's and vLLM's floors). Add a `z3.Solver()` timeout.

**Model choice:** stay at Qwen2.5-1.5B. Unsloth documents 1.5B as the GRPO floor, so a null result at 0.5B is uninterpretable — reviewers read it as "too small," not as evidence. Use 0.5B only as a one-seed appendix trend point. Prefer Qwen2.5 over Qwen3 (4× smaller KV cache).

---

## 4. Mandatory controls

These are not optional; each is demanded by a specific paper, and all are cheap.

| Control | Demanded by | Cost |
|---|---|---|
| **pass@k out to k≥64** vs base model | Yue et al. 2504.13837 (NeurIPS Oral) | Sampling only, no training |
| Base-model coverage: is the RL model's correct set a subset? | Yue et al. | Sampling only |
| **Non-Qwen control** (Llama-3.2-1B or OLMo) | Spurious Rewards 2506.10947 | 1 run |
| **Random-reward arm** | Spurious Rewards | 1 run |
| **One-example arm** | 1-shot RLVR 2504.20571 | 1 run |
| **Distillation baseline** | Yue et al. names distillation as what works | 1 run |
| 3 seeds + CIs on every headline number | Standard | 3× |

Run pass@k **first**, in week 1. If base-model coverage dominates, that *is* the paper (C2/C3 become the story) — better to learn it in week 1 than week 4.

---

## 5. Four-week schedule

**Week 1 — Unblock and de-risk.**
Port to TRL 1.10 (drop `max_prompt_length`, pre-truncate); make Unsloth optional, default off; batch the evaluator; wire `beta=0` / `dr_grpo` / vLLM colocate; pin `vllm==0.26.0`. Confirm real Kaggle quota in-notebook. Rebuild data splits with template grouping. 20-step smoke run.
*In parallel (CPU, no GPU needed):* start the benchmark generator — this is the primary contribution and does not compete for GPU time.
**Gate:** does reward move at all on a 150-step run? If not, stop and diagnose.

**Week 2 — Benchmark + baseline.**
Finish the generator, the verifier-derived step labels, the perturbation families, and the faithfulness metric. Run pass@k and the base-model coverage analysis. 3 seeds of the main GRPO arm.
**Gate:** does the benchmark discriminate between models? If every model scores the same, the metric is broken.

**Week 3 — Matrix.**
Controls (non-Qwen, random-reward, one-example, distillation) + the tool-fading arms for C3. Checkpoint every 25 steps.

**Week 4 — Measure and write.**
OOD eval, CIs over seeds, figures, abstract. Reserve the last 4 days for writing only.

---

## 6. Verification debt — clear before writing related work

**Cleared 2026-08-18** (fetched from arXiv directly):
- ✅ **SKILL0 (2604.02268)** — real. Lu et al., 2 Apr 2026 (v2 15 May). GRPO; 3 stages; linear decay `M(s)=⌈N·(NS−s)/(NS−1)⌉`, budgets `[6,3,0]` ALFWorld / `[5,3,0]` Search-QA; withdrawal every 10 steps by on-policy helpfulness; Qwen2.5-**VL** 3B/7B; ALFWorld/Search-QA/WebShop; **no symbolic verifier, no math/logic benchmarks**; fades **prompt context**, not the reward signal.
- ✅ **FoVer (2505.15960)** — real. Kamoi et al., ACL 2026 Findings. Z3 on FLDx2, Isabelle on GSM8K/MetaMathQA/Big-Math; 40K steps @ 50% error rate; Llama-3.1-8B and Qwen2.5-7B (+Qwen3-4B robustness). **PRM used only for Best-of-K reranking and step-error detection — no RL.** No curriculum, no tool fading, no verifier removal. Benchmarks: FOLIO, LogicNLI, GSM8K, MATH, AQuA-RAT, AIME, ANLI, HANS, MMLU-Pro-NoMath, BBH, ProcessBench.
- ✅ **VeriGate (2605.30451)** — real. Agrawal, Liu, Huang, 28 May 2026. Qwen2.5-Instruct 1.5B/7B on MATH, ~20%/12% gains. Step signal is a **learned neural PRM**, not symbolic.
- ✅ "Verifiable PRMs for Structured Reasoning" — ACL 2026 Findings, `aclanthology.org/2026.findings-acl.1611`.

Still **unconfirmed** — check before citation:
- VeriBound (2606.20740), 2604.22074, 2605.11467, the CoT-monitorability position paper
- All logic-benchmark IDs: FOLIO, ProofWriter, ProntoQA, LogicBench, ZebraLogic, JustLogic, MALLS, MathCheck
- "RL Finetunes Small Subnetworks"; the DeepSeek-R1 small-model-RL-vs-distillation passage

Do not cite any 2026-dated arXiv ID from the scan without fetching the abstract.

---

## 7. Open decisions for the author

1. **Does the reframed paper interest you enough to spend four weeks on it?** It is a real ICLR-shaped contribution but not the paper you set out to write.
2. **ICLR 2027 vs. the next cycle.** Four weeks is enough for this paper. It is not enough to build the original system properly. Aiming the full CurricSym system at a later venue, with the benchmark as a prior publication that motivates it, is a coherent two-paper strategy.
3. **Domain scope.** Math + FOL, or FOL only? FOL gives cleaner verifier semantics for step-level labels, which is where the contribution lives. Math adds breadth but weaker step verification.
