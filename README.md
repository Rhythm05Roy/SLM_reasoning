# CurricSym-SLM-Lite

**A Compute-Constrained Neurosymbolic Curriculum RL Framework for Small Language Models**

> Thesis implementation · RTX 4090 (RunPod) · Qwen2.5-3B-Instruct · GRPO + AdaRFT Curriculum

---

## Overview

CurricSym-SLM-Lite trains a 3B-parameter language model to internalise symbolic reasoning through a **three-stage curriculum reinforcement learning** pipeline:

| Stage | Tool Ratio | Reward Functions | Goal |
|-------|-----------|-----------------|------|
| 0 – Early | 1.0 (full access) | outcome + format + process | Learn to use Z3 verifier |
| 1 – Mid | 0.5 (partial) | + internalization | Begin fading tool dependency |
| 2 – Late | 0.0 (no tools) | + internalization | Full symbolic internalisation |

The difficulty of training examples is dynamically adjusted per-step using an **AdaRFT-inspired scheduler** that tracks recent reward history and moves the difficulty target up or down accordingly.

---

## Key Contributions (Thesis)

1. **Neurosymbolic curriculum**: Verified (Z3) reasoning traces used as training signal, not just as evaluation oracle
2. **Tool-fading internalization**: Quantified by the *internalization delta* — accuracy drop when symbolic tools are removed
3. **AdaRFT dynamic difficulty**: Per-step curriculum adaptation without a separate curriculum model
4. **Heuristic PRM baseline**: Process reward computed from thinking-trace structural features (step markers, length, verify tokens)
5. **Paired ablation dataset**: Pre-built with/without-tool prompt pairs for unconfounded internalization measurement

---

## Project Structure

```
curricsym/
├── configs/
│   └── config.py            # TrainingConfig dataclass (RTX 4090 profile)
├── data/
│   └── loader.py            # GSM-Symbolic + ProofWriter loading & formatting
├── models/
│   ├── verifier.py          # SymbolicVerifier (Z3 math + FOL oracle)
│   └── model_loader.py      # Unsloth model + QLoRA setup, adapter I/O
├── training/
│   ├── reward_functions.py  # 4 GRPO reward functions
│   ├── curriculum.py        # CurriculumScheduler + CurriculumCallback
│   ├── sft_trainer.py       # SFT warm-up phase
│   └── grpo_trainer.py      # Three-stage GRPO pipeline
├── evaluation/
│   ├── evaluator.py         # EvaluationFramework + InternalizationEvaluator
│   └── visualisation.py     # Matplotlib figures (6 plots + dashboard)
├── scripts/
│   ├── setup_runpod.sh      # One-shot RunPod setup
│   ├── smoke_test.py        # Import + logic validation (no GPU needed)
│   ├── eval_only.py         # Evaluate a saved adapter
│   ├── generate_plots.py    # Re-generate figures from JSON
│   └── push_to_hub.py       # Publish to HuggingFace Hub
├── reporting.py             # JSON report + human summary builder
├── train.py                 # 🚀 Main entry point
└── requirements.txt         # Pinned for CUDA 12.4 + torch 2.3
```

---

## Quickstart on RunPod RTX 4090

### 1. Provision the instance

Recommended pod spec:
- **GPU**: RTX 4090 (24 GB VRAM)
- **Image**: `runpod/pytorch:2.3.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Disk**: 50 GB container + 100 GB network volume mounted at `/workspace`

### 2. Clone and set up

```bash
git clone <your-repo-url> /workspace/curricsym-project
cd /workspace/curricsym-project/curricsym
bash scripts/setup_runpod.sh
```

### 3. Validate (no GPU needed)

```bash
cd /workspace/curricsym-project
python curricsym/scripts/smoke_test.py
```

Expected output:
```
✅  All 17 checks passed — ready to train!
```

### 4. Set environment variables

```bash
export WANDB_API_KEY="your_wandb_key"       # from wandb.ai/authorize
export HF_TOKEN="hf_xxx"                    # only needed for Hub push
```

### 5. Run full training

```bash
cd /workspace/curricsym-project
python curricsym/train.py
```

Training phases and approximate times on a single RTX 4090:

| Phase | Steps | Approx. Time |
|-------|-------|-------------|
| SFT warm-up | 500 | ~45 min |
| GRPO Stage 0 (tools=1.0) | 100 | ~60 min |
| GRPO Stage 1 (tools=0.5) | 100 | ~60 min |
| GRPO Stage 2 (tools=0.0) | 100 | ~60 min |
| Evaluation (300 examples) | — | ~20 min |
| **Total** | | **~4–5 hours** |

---

## CLI Reference

```bash
python curricsym/train.py [OPTIONS]

Model:
  --model_name            unsloth/Qwen2.5-3B-Instruct  (default)
  --max_seq_length        4096
  --load_in_4bit          Use QLoRA 4-bit (for 7B+ or <24 GB GPU)
  --lora_r                32
  --lora_alpha            64

SFT:
  --sft_lr                2e-4
  --sft_batch_size        4
  --sft_max_steps         500

GRPO:
  --grpo_lr               5e-6
  --grpo_batch_size       8
  --grpo_num_generations  8
  --grpo_max_steps        300

Paths:
  --output_dir            /workspace/curricsym_output
  --checkpoint_dir        /workspace/curricsym_checkpoints

Resume:
  --resume_from_sft       /path/to/sft_adapter     (skip SFT phase)
  --resume_grpo_stage     1                         (resume from stage N)

Logging:
  --no_wandb                                        (disable W&B)
  --wandb_project         curricsym-slm-lite

Eval:
  --max_eval_examples     300
  --skip_eval             (faster, for debugging)

Export:
  --merge_model           Save full BF16 merged model
  --push_to_hub           Push to HuggingFace Hub
  --hf_repo               your-username/model-name
```

---

## Resume Workflow

For multi-session training (e.g. 2 × 4-hour RunPod sessions):

**Session 1** — SFT + GRPO Stage 0
```bash
python curricsym/train.py --grpo_max_steps 100   # only runs stage 0
```

**Session 2** — GRPO Stages 1 & 2 + Evaluation
```bash
python curricsym/train.py \
  --resume_from_sft   /workspace/curricsym_checkpoints/sft_adapter \
  --resume_grpo_stage 1
```

**Evaluation only** (any session)
```bash
python curricsym/scripts/eval_only.py \
  --adapter_path /workspace/curricsym_checkpoints/grpo_final \
  --max_eval_examples 400
```

---

## Hyperparameter Guide (RTX 4090)

### Memory Budget

| Config | VRAM Usage |
|--------|-----------|
| 3B BF16 base model | ~7 GB |
| + LoRA r=32 | ~7.5 GB |
| + GRPO 8 generations (batch=8) | ~18–20 GB |
| Headroom | ~4–6 GB ✅ |

For 7B models → set `--load_in_4bit --grpo_num_generations 4`

### Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lora_r` | 32 | Increase to 64 for better capacity (uses ~+1 GB) |
| `grpo_batch_size` | 8 | Must equal `grpo_num_generations` |
| `grpo_num_generations` | 8 | 4090 can handle 8; use 4 if OOM |
| `grpo_lr` | 5e-6 | Decays 0.5× per stage — keep low |
| `grpo_beta` | 0.04 | KL penalty; increase if reward spikes |
| `curriculum_target_reward` | 0.5 | Difficulty calibration target |
| `curriculum_alpha` | 2.0 | Difficulty update rate |
| `curriculum_eta` | 50.0 | Difficulty update scale |

---

## Outputs

After a complete run, `output_dir` contains:

```
curricsym_output/
├── experiment_report.json      # Full results dict (for thesis appendix)
├── evaluation_results.json     # Evaluation metrics
├── grpo_stage_metrics.json     # Per-stage loss + tool_ratio
├── config.json                 # Exact config snapshot (reproducibility)
├── dashboard.png               # 2×3 summary figure
├── fig_stage_losses.png
├── fig_tool_fading.png
├── fig_accuracy_comparison.png
├── fig_process_quality.png
├── fig_internalization_delta.png
└── sft/ grpo_stage_*/ ...      # Trainer output dirs

curricsym_checkpoints/
├── sft_adapter/                # SFT LoRA weights
├── grpo_stage_0/
├── grpo_stage_1/
├── grpo_stage_2/
└── grpo_final/                 # Final adapter (use this for evaluation)
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `overall_accuracy` | Fraction of correct predictions (Z3 oracle) |
| `math_accuracy` | GSM-Symbolic accuracy |
| `fol_accuracy` | ProofWriter accuracy |
| `internalization_delta` | `acc_with_tools − acc_without_tools` (↓ better) |
| `consistency_rate` | Fraction of paired examples with same answer both ways |
| `avg_faithfulness` | Heuristic: does thinking trace align with verifier? |
| `avg_format_score` | `<thinking>/<answer>` structural compliance |
| `avg_latency_s` | Per-example inference time (seconds) |
| `tool_call_rate` | Fraction of completions containing `<verify>` |

---

## Ablation Studies

Five ablations are automatically run and saved in `experiment_report.json`:

1. **Curriculum progression** — stage-wise losses showing dynamic difficulty adaptation
2. **Tool reliance** — tool_call_rate vs accuracy gap (internalization delta)
3. **Domain breakdown** — separate math / FOL accuracy comparison
4. **Process faithfulness** — thinking-trace alignment with Z3 oracle
5. **Efficiency** — latency, format compliance, consistency rate

---

## Reproducibility

All randomness is seeded via `--seed 42` (default). The config snapshot `config.json` records:
- Exact model name and LoRA configuration
- All hyperparameters
- Library versions (torch, transformers, trl, unsloth)
- Hardware info (GPU name, VRAM)

To reproduce a run:
```bash
python curricsym/train.py $(python -c "
import json
c = json.load(open('curricsym_output/config.json'))
flags = ' '.join(f'--{k} {v}' for k,v in c.items()
                  if isinstance(v, (int, float, str, bool)) and not k.startswith('_'))
print(flags)
")
```

---

## Citation

```bibtex
@misc{curricsym2026,
  title   = {CurricSym-SLM-Lite: Neurosymbolic Curriculum Reinforcement Learning
             for Small Language Model Reasoning},
  author  = {[Author]},
  year    = {2026},
  note    = {Thesis implementation. Model: Qwen2.5-3B-Instruct + GRPO + AdaRFT curriculum.}
}
```

---

## Known Issues / Limitations

- **Heuristic PRM**: The process reward is based on structural features (step markers, trace length), not a distilled neural PRM. This is documented in the thesis as a limitation and future work direction.
- **Z3 math oracle**: Implemented as numeric equality via UNSAT — equivalent to float comparison but wrapped in a symbolic interface. Not a full arithmetic constraint solver.
- **FOL "Unknown"**: Z3 has no native 3-valued logic; "Unknown" answers fall back to string matching.
- **Single GPU**: The RTX 4090 path uses single-GPU training only. DDP is not implemented (and not needed for 3B models on 24 GB).
