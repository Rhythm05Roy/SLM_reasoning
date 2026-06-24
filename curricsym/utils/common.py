"""utils/common.py — Shared utilities: seeding, timing, VRAM, logging."""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "pct": trainable / total * 100 if total > 0 else 0.0,
    }


def format_time(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    elif secs < 3600:
        return f"{secs / 60:.1f}m"
    return f"{secs / 3600:.1f}h"


class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start

    def __str__(self) -> str:
        return format_time(self.elapsed())


def vram_summary() -> str:
    if not torch.cuda.is_available():
        return "No GPU"
    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024 ** 3
        alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
        lines.append(
            f"GPU {i} [{props.name}]: alloc={alloc:.2f}/{total:.2f} GB "
            f"({alloc / total * 100:.1f}%)"
        )
    return "\n".join(lines)


def list_checkpoints(checkpoint_dir: str) -> dict:
    cp = {}
    p = Path(checkpoint_dir)
    if not p.exists():
        return cp
    for item in sorted(p.iterdir()):
        if item.is_dir():
            size_mb = sum(
                f.stat().st_size for f in item.rglob("*") if f.is_file()
            ) / 1024 ** 2
            cp[item.name] = {"path": str(item), "size_mb": size_mb}
    return cp


def log_environment(config, logger: Optional[logging.Logger] = None) -> dict:
    env = {
        "seed": config.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "num_gpus": torch.cuda.device_count(),
    }
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import unsloth
        env["unsloth"] = getattr(unsloth, "__version__", "installed")
    except ImportError:
        pass
    try:
        import trl
        env["trl"] = trl.__version__
    except ImportError:
        pass
    if logger:
        out = "\n".join(f"  {k:<25}: {v}" for k, v in env.items())
        logger.info("Environment:\n" + out)
    return env
