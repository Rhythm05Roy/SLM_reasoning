"""
models/verifier.py — Symbolic Verifier
========================================
Cached Z3-based correctness oracle for math (numeric) and FOL (propositional).

Design notes (thesis methods section):
  Math:  Z3 as numeric oracle — pred==gt checked via UNSAT.
         Equivalent to float equality but extensible to constraint checking.
  FOL:   Z3 propositional consistency for True/False labels.
         'Unknown' answers fall back to string match (Z3 has no native
         3-valued logic in this minimal interface).
  Cache: keyed on MD5(question, predicted) to avoid redundant solver calls
         during online GRPO reward computation.
"""
from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

try:
    import z3
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False
    logger.warning("z3-solver not found — falling back to string matching")


# ---------------------------------------------------------------------------
# Stand-alone functions (used during offline annotation, Cell 9)
# ---------------------------------------------------------------------------
def verify_math_answer(
    question: str, predicted: str, ground_truth: str
) -> dict:
    """Z3 numeric oracle: predicted == ground_truth?"""
    try:
        pv = float(str(predicted).strip().replace(",", ""))
        gv = float(str(ground_truth).strip().replace(",", ""))
        if _Z3_AVAILABLE:
            s = z3.Solver()
            x = z3.Real("x")
            s.add(x == z3.RealVal(str(pv)))
            s.add(x != z3.RealVal(str(gv)))
            is_correct = s.check() == z3.unsat
            return {
                "correct": is_correct,
                "confidence": 1.0 if is_correct else 0.0,
                "verifier_trace": f"Z3 numeric: {pv} vs {gv}",
                "z3_used": True,
                "method": "z3_numeric_oracle",
            }
        # fallback when Z3 not available
        is_correct = abs(pv - gv) < 1e-9
        return {
            "correct": is_correct,
            "confidence": 1.0 if is_correct else 0.0,
            "verifier_trace": f"float eq: {pv} vs {gv}",
            "z3_used": False,
            "method": "float_eq",
        }
    except (ValueError, Exception):
        is_correct = str(predicted).strip().lower() == str(ground_truth).strip().lower()
        return {
            "correct": is_correct,
            "confidence": 0.8 if is_correct else 0.0,
            "verifier_trace": f"string fallback: '{predicted}' vs '{ground_truth}'",
            "z3_used": False,
            "method": "string_fallback",
        }


def verify_fol_answer(
    question: str, predicted: str, ground_truth: str
) -> dict:
    """Z3 propositional consistency check for True/False/Unknown."""
    pred_norm = predicted.strip().lower()
    gt_norm = ground_truth.strip().lower()
    if pred_norm == gt_norm:
        return {
            "correct": True,
            "confidence": 1.0,
            "verifier_trace": f"FOL exact match: {pred_norm}",
            "z3_used": False,
            "method": "exact_match",
        }
    if _Z3_AVAILABLE:
        try:
            s = z3.Solver()
            P = z3.Bool("P")
            val_map = {"true": True, "false": False}
            pred_val = val_map.get(pred_norm)
            gt_val = val_map.get(gt_norm)
            if pred_val is not None and gt_val is not None:
                s.add(P == z3.BoolVal(pred_val))
                s.add(P != z3.BoolVal(gt_val))
                is_correct = s.check() == z3.unsat
                return {
                    "correct": is_correct,
                    "confidence": 1.0 if is_correct else 0.0,
                    "verifier_trace": f"Z3 FOL: {pred_norm} vs {gt_norm}",
                    "z3_used": True,
                    "method": "z3_propositional",
                }
        except Exception:
            pass
    return {
        "correct": False,
        "confidence": 0.0,
        "verifier_trace": "FOL fallback: unknown mismatch",
        "z3_used": False,
        "method": "string_fallback",
    }


# ---------------------------------------------------------------------------
# Cached class for online reward computation
# ---------------------------------------------------------------------------
class SymbolicVerifier:
    """
    Thread-safe (single-process), cached symbolic verifier.

    Caches results by MD5(question, predicted) hash.  Used during GRPO
    reward computation where the same (q, pred) pair may appear across
    multiple reward-function calls in a single training step.

    Thesis note
    -----------
    Math:  Z3 numeric oracle — equivalent to float equality but provides
           a clean symbolic interface for future extension to arithmetic
           constraint solving.
    FOL:   Z3 propositional consistency.  'Unknown' uses string match only.
    """

    def __init__(self):
        self.cache: dict[str, dict] = {}
        self.stats: dict[str, int] = {
            "total": 0, "cached": 0, "z3_calls": 0, "fallbacks": 0,
        }

    def _key(self, q: str, a: str, gt: str) -> str:
        return hashlib.md5(f"{q}:{a}:{gt}".encode()).hexdigest()

    def verify_math(self, question: str, predicted: str, ground_truth: str) -> dict:
        self.stats["total"] += 1
        key = self._key(question, predicted, ground_truth)
        if key in self.cache:
            self.stats["cached"] += 1
            return self.cache[key]
        try:
            pv = float(str(predicted).strip().replace(",", ""))
            gv = float(str(ground_truth).strip().replace(",", ""))
            if _Z3_AVAILABLE:
                s = z3.Solver()
                x = z3.Real("x")
                s.add(x == z3.RealVal(str(pv)))
                s.add(x != z3.RealVal(str(gv)))
                ok = s.check() == z3.unsat
                result = {"correct": ok, "confidence": 1.0 if ok else 0.0,
                          "method": "z3_numeric_oracle"}
                self.stats["z3_calls"] += 1
            else:
                ok = abs(pv - gv) < 1e-9
                result = {"correct": ok, "confidence": 1.0 if ok else 0.0,
                          "method": "float_eq"}
        except (ValueError, Exception):
            ok = str(predicted).strip().lower() == str(ground_truth).strip().lower()
            result = {"correct": ok, "confidence": 0.8 if ok else 0.0,
                      "method": "string_fallback"}
            self.stats["fallbacks"] += 1
        self.cache[key] = result
        return result

    def verify_fol(self, question: str, predicted: str, ground_truth: str) -> dict:
        self.stats["total"] += 1
        key = self._key(question, predicted, ground_truth)
        if key in self.cache:
            self.stats["cached"] += 1
            return self.cache[key]
        pn, gn = predicted.strip().lower(), ground_truth.strip().lower()
        if pn == gn:
            result = {"correct": True, "confidence": 1.0, "method": "exact_match"}
        elif _Z3_AVAILABLE:
            try:
                s = z3.Solver()
                P = z3.Bool("P")
                vm = {"true": True, "false": False}
                pv, gv = vm.get(pn), vm.get(gn)
                if pv is not None and gv is not None:
                    s.add(P == z3.BoolVal(pv))
                    s.add(P != z3.BoolVal(gv))
                    ok = s.check() == z3.unsat
                    self.stats["z3_calls"] += 1
                else:
                    ok = False
                result = {"correct": ok, "confidence": 1.0 if ok else 0.0,
                          "method": "z3_propositional"}
            except Exception:
                result = {"correct": False, "confidence": 0.0, "method": "string_fallback"}
                self.stats["fallbacks"] += 1
        else:
            result = {"correct": False, "confidence": 0.0, "method": "string_fallback"}
        self.cache[key] = result
        return result

    def verify(self, domain: str, question: str, predicted: str, ground_truth: str) -> dict:
        if domain == "math":
            return self.verify_math(question, predicted, ground_truth)
        return self.verify_fol(question, predicted, ground_truth)

    def get_stats(self) -> dict:
        return {**self.stats, "cache_size": len(self.cache)}

    def clear_cache(self) -> None:
        self.cache.clear()
