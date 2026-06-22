#!/usr/bin/env bash
# ============================================================
# scripts/setup_runpod.sh — One-shot RunPod RTX 4090 Setup
# ============================================================
# Run this ONCE after spinning up a fresh RunPod instance.
# Expected base image: runpod/pytorch:2.3.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Usage:
#   chmod +x scripts/setup_runpod.sh
#   bash scripts/setup_runpod.sh
# ============================================================

set -e

echo "============================================================"
echo "  CurricSym-SLM-Lite  ·  RunPod Setup"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq \
    git curl wget build-essential \
    libssl-dev libffi-dev python3-dev \
    > /dev/null

# ── 2. Upgrade pip ────────────────────────────────────────────
echo "[2/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

# ── 3. Install PyTorch (already present in RunPod image, verify) ──
echo "[3/6] Verifying PyTorch + CUDA..."
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}')"

# ── 4. Install Unsloth ────────────────────────────────────────
echo "[4/6] Installing Unsloth..."
pip install "unsloth[cu124-torch230]" --no-deps -q 2>/dev/null || \
pip install "unsloth" --no-deps -q
pip install unsloth_zoo -q

# ── 5. Install project dependencies ───────────────────────────
echo "[5/6] Installing project requirements..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -r "$PROJECT_ROOT/requirements.txt" --no-deps -q 2>/dev/null || \
pip install transformers datasets accelerate peft trl bitsandbytes \
            z3-solver wandb numpy pandas matplotlib seaborn tqdm rich \
            huggingface_hub safetensors -q

# Flash Attention 2 (Ampere / RTX 4090 only)
pip install flash-attn --no-build-isolation -q 2>/dev/null && \
    echo "  Flash Attention 2 installed ✅" || \
    echo "  Flash Attention 2 skipped (may need manual install)"

# ── 6. Set up workspace directories ──────────────────────────
echo "[6/6] Creating workspace directories..."
mkdir -p /workspace/curricsym_output \
         /workspace/curricsym_checkpoints \
         /workspace/curricsym_data

echo ""
echo "============================================================"
echo "  Setup complete ✅"
echo "============================================================"
echo ""
echo "  To start training:"
echo "    cd $PROJECT_ROOT"
echo "    python train.py"
echo ""
echo "  To resume from SFT checkpoint:"
echo "    python train.py --resume_from_sft /workspace/curricsym_checkpoints/sft_adapter"
echo ""
echo "  To run evaluation only:"
echo "    python scripts/eval_only.py \\"
echo "      --adapter_path /workspace/curricsym_checkpoints/grpo_final"
echo ""
