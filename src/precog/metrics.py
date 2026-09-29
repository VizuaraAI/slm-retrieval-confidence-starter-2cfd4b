"""Answer metrics, calibration metrics and the budget-curve summary.

The exact-match normalisation follows the standard SQuAD / open-domain QA
convention so numbers are comparable with published work.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(a) for a in answers))


def contains_match(prediction: str, answers: Sequence[str]) -> float:
    """Lenient variant. Useful for chatty small models that pad the answer."""
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(a) and normalize_answer(a) in pred
                     for a in answers))


def _f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, answers: Sequence[str]) -> float:
    return max((_f1(prediction, a) for a in answers), default=0.0)


def expected_calibration_error(probs: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """Standard equal-width ECE over predicted confidence."""
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - probs[mask].mean())
    return float(ece)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank statistic. Returns 0.5 if a class is missing."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return 0.5
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for value in np.unique(scores):
        tie = scores == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def budget_curve(utility_scores: np.ndarray, correct_retrieve: np.ndarray,
                 correct_no_retrieve: np.ndarray,
                 budgets: Sequence[float]) -> List[Dict[str, float]]:
    """Accuracy the gate reaches at each allowed retrieval rate.

    At budget b we retrieve the b fraction of questions with the highest
    predicted utility, and answer the rest closed-book.
    """
    utility_scores = np.asarray(utility_scores, dtype=float)
    correct_retrieve = np.asarray(correct_retrieve, dtype=float)
    correct_no_retrieve = np.asarray(correct_no_retrieve, dtype=float)
    n = len(utility_scores)
    order = np.argsort(-utility_scores)
    points = []
    for b in budgets:
        k = int(round(b * n))
        retrieve = np.zeros(n, dtype=bool)
        retrieve[order[:k]] = True
        acc = np.where(retrieve, correct_retrieve, correct_no_retrieve).mean()
        points.append({"budget": float(b), "retrieval_rate": k / n if n else 0.0,
                       "accuracy": float(acc)})
    return points


def area_under_budget_curve(points: List[Dict[str, float]]) -> float:
    """Trapezoidal area under accuracy-versus-retrieval-rate. Higher is better.

    A gate that orders questions perfectly gets a high area because it buys
    most of its accuracy in the first few percent of the budget.
    """
    xs = np.array([p["retrieval_rate"] for p in points], dtype=float)
    ys = np.array([p["accuracy"] for p in points], dtype=float)
    order = xs.argsort()
    return float(np.trapezoid(ys[order], xs[order]))


def summarize(records: List[Dict]) -> Dict[str, float]:
    """Aggregate a list of per-question result rows."""
    if not records:
        return {}
    keys = ["em", "f1", "retrieved", "latency_ms"]
    out = {}
    for key in keys:
        values = [r[key] for r in records if key in r]
        if values:
            out[key] = float(np.mean(values))
    out["n"] = len(records)
    return out
