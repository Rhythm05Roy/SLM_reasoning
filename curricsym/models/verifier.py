"""
models/verifier.py — Symbolic Verifier (Z3 oracle + string fallback)

Math:  Z3 numeric UNSAT check (pred == gt).
FOL:   Z3 propositional consistency (True/False labels).
Cache: MD5(question, predicted) keyed dict — avoids redundant solver calls.

Thesis note: this is a pragmatic oracle, not a full SMT reasoning engine.
Full SMT encoding of multi-step math problems is documented as future work.
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
# Stand-alone functions (used for offline annotation)
# ---------------------------------------------------------------------------
def verify_math_answer(question: str, predicted: str, ground_truth: str) -> dict:
    try:
        pv = float(str(predicted).strip().replace(",", ""))
        gv = float(str(ground_truth).strip().replace(",", ""))
        if _Z3_AVAILABLE:
            s = z3.Solver()
            x = z3.Real("x")
            s.add(x == z3.RealVal(str(pv)))
            s.add(x != z3.RealVal(str(gv)))
            ok = s.check() == z3.unsat
            return {"correct": ok, "confidence": 1.0 if ok else 0.0,
                    "verifier_trace": f"Z3 numeric: {pv} vs {gv}",
                    "z3_used": True, "method": "z3_numeric_oracle"}
        ok = abs(pv - gv) < 1e-9
        return {"correct": ok, "confidence": 1.0 if ok else 0.0,
                "verifier_trace": f"float eq: {pv} vs {gv}",
                "z3_used": False, "method": "float_eq"}
    except (ValueError, Exception):
        ok = str(predicted).strip().lower() == str(ground_truth).strip().lower()
        return {"correct": ok, "confidence": 0.8 if ok else 0.0,
                "verifier_trace": f"string fallback: '{predicted}' vs '{ground_truth}'",
                "z3_used": False, "method": "string_fallback"}


def verify_fol_answer(question: str, predicted: str, ground_truth: str) -> dict:
    pn = predicted.strip().lower()
    gn = ground_truth.strip().lower()
    if pn == gn:
        return {"correct": True, "confidence": 1.0,
                "verifier_trace": f"FOL exact match: {pn}",
                "z3_used": False, "method": "exact_match"}
    if _Z3_AVAILABLE:
        try:
            s = z3.Solver()
            P = z3.Bool("P")
            vm = {"true": True, "false": False}
            pv, gv = vm.get(pn), vm.get(gn)
            if pv is not None and gv is not None:
                s.add(P == z3.BoolVal(pv))
                s.add(P != z3.BoolVal(gv))
                ok = s.check() == z3.unsat
                return {"correct": ok, "confidence": 1.0 if ok else 0.0,
                        "verifier_trace": f"Z3 FOL: {pn} vs {gn}",
                        "z3_used": True, "method": "z3_propositional"}
        except Exception:
            pass
    return {"correct": False, "confidence": 0.0,
            "verifier_trace": "FOL string mismatch", "z3_used": False,
            "method": "string_fallback"}


# ---------------------------------------------------------------------------
# Cached class for online GRPO reward computation
# ---------------------------------------------------------------------------
class SymbolicVerifier:
    """
    Cached symbolic verifier for GRPO reward computation.

    Limitations (documented for thesis):
    - Math: Z3 as numeric equality oracle, NOT full arithmetic constraint solver
    - FOL: Propositional only; 'Unknown' answers use string fallback
    - Future work: full SMT encoding of multi-step arithmetic reasoning chains
    """

    def __init__(self):
        self.cache: dict[str, dict] = {}
        self.stats: dict[str, int] = {
            "total": 0, "cached": 0, "z3_calls": 0, "fallbacks": 0,
        }

    def _key(self, q: str, p: str, g: str) -> str:
        return hashlib.md5(f"{q}:{p}:{g}".encode()).hexdigest()

    def verify_math(self, question: str, predicted: str, ground_truth: str) -> dict:
        self.stats["total"] += 1
        key = self._key(question, predicted, ground_truth)
        if key in self.cache:
            self.stats["cached"] += 1
            return self.cache[key]
        result = verify_math_answer(question, predicted, ground_truth)
        if result.get("z3_used"):
            self.stats["z3_calls"] += 1
        if result.get("method") == "string_fallback":
            self.stats["fallbacks"] += 1
        self.cache[key] = result
        return result

    def verify_fol(self, question: str, predicted: str, ground_truth: str) -> dict:
        self.stats["total"] += 1
        key = self._key(question, predicted, ground_truth)
        if key in self.cache:
            self.stats["cached"] += 1
            return self.cache[key]
        result = verify_fol_answer(question, predicted, ground_truth)
        if result.get("z3_used"):
            self.stats["z3_calls"] += 1
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
