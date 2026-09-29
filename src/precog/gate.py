"""The calibrated utility gate.

Two ideas live here.

1. A three-class probe over signed retrieval utility u in {-1, 0, +1}, scored
   as expected utility u_hat = P(+1) - P(-1). Predicting the sign, not just the
   difficulty, is the paper's central claim.

2. A risk-controlled threshold. Instead of eyeballing a validation curve, we
   scan candidate thresholds on a held-out calibration split and keep the most
   aggressive one whose UPPER confidence bound on the accuracy drop versus
   always-retrieve stays below alpha.

   The per-question loss of skipping is max(0, c_rag - c_closed), and with a
   0/1 correctness metric that is Bernoulli. So we use an exact Clopper-Pearson
   upper bound, which is far tighter than Hoeffding at the calibration sizes
   you can actually afford. Hoeffding is kept as a fallback for when the metric
   is continuous (token F1, a judge score) or scipy is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = [-1, 0, 1]


def hoeffding_upper_bound(mean: float, n: int, delta: float) -> float:
    """Upper confidence bound on the mean of any variable in [0, 1]."""
    if n <= 0:
        return 1.0
    return mean + math.sqrt(math.log(1.0 / max(delta, 1e-12)) / (2.0 * n))


def binomial_upper_bound(successes: float, n: int, delta: float) -> float:
    """Clopper-Pearson upper bound on a Bernoulli rate. Exact, not asymptotic.

    Valid only when the loss really is 0/1 per question, which it is for exact
    match. Falls back to Hoeffding if scipy is not installed.
    """
    if n <= 0:
        return 1.0
    k = int(round(successes))
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta
    except ImportError:  # pragma: no cover
        return hoeffding_upper_bound(k / n, n, delta)
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def _is_binary(values: np.ndarray) -> bool:
    return bool(np.all(np.isin(np.asarray(values), (0.0, 1.0))))


@dataclass
class GateDecision:
    retrieve: np.ndarray      # bool array
    scores: np.ndarray        # predicted expected utility
    threshold: float


class UtilityGate:
    """Predicts the signed utility of retrieving for a given query."""

    def __init__(self, pca_dim: int = 128, hidden_layer_sizes: Sequence[int] = (256,),
                 seed: int = 13):
        steps = [("scale", StandardScaler())]
        if pca_dim and pca_dim > 0:
            steps.append(("pca", PCA(n_components=pca_dim, random_state=seed)))
        steps.append((
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                activation="relu",
                alpha=1e-3,
                max_iter=400,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=seed,
            ),
        ))
        self.pipeline = Pipeline(steps)
        self.threshold: float = 0.0
        self.classes_: List[int] = []

    def fit(self, features: np.ndarray, utility: np.ndarray) -> "UtilityGate":
        utility = np.asarray(utility).astype(int)
        # PCA cannot ask for more components than samples or dimensions.
        if "pca" in dict(self.pipeline.steps):
            pca = self.pipeline.named_steps["pca"]
            pca.n_components = min(pca.n_components, *features.shape)
        self.pipeline.fit(features, utility)
        self.classes_ = list(self.pipeline.named_steps["mlp"].classes_)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Expected utility of retrieving, in [-1, 1]. Higher means retrieve."""
        probs = self.pipeline.predict_proba(features)
        index = {c: i for i, c in enumerate(self.classes_)}
        p_pos = probs[:, index[1]] if 1 in index else np.zeros(len(features))
        p_neg = probs[:, index[-1]] if -1 in index else np.zeros(len(features))
        return p_pos - p_neg

    # -------------------------------------------------------------- thresholds
    def calibrate_risk(self, scores: np.ndarray, correct_retrieve: np.ndarray,
                       correct_skip: np.ndarray, alpha: float = 0.02,
                       delta: float = 0.1, n_grid: int = 200) -> float:
        """Most aggressive threshold whose accuracy loss is provably <= alpha.

        The loss of skipping question i is max(0, c_retrieve_i - c_skip_i). We
        scan thresholds from low (retrieve almost always, zero loss) to high
        (skip almost always) and keep the last one whose upper confidence bound
        still satisfies alpha. Monotonicity in tau is what makes the single
        scan valid: raising tau only ever adds skipped questions.
        """
        scores = np.asarray(scores, dtype=float)
        c_r = np.asarray(correct_retrieve, dtype=float)
        c_s = np.asarray(correct_skip, dtype=float)
        n = len(scores)
        per_question_loss = np.maximum(0.0, c_r - c_s)
        binary = _is_binary(per_question_loss)

        def bound(total_loss: float) -> float:
            if binary:
                return binomial_upper_bound(total_loss, n, delta)
            return hoeffding_upper_bound(total_loss / n, n, delta)

        # Even a gate that never loses anything pays the bound's floor. If that
        # floor already exceeds alpha, no threshold can pass and the gate will
        # retrieve everything. Say so rather than returning a silent degenerate.
        floor = bound(0.0)
        if floor > alpha:
            need = "raise alpha, raise delta, or grow calib_size"
            print(f"warning: at n={n}, delta={delta} the tightest achievable "
                  f"bound is {floor:.4f}, already above alpha={alpha}. "
                  f"The gate will retrieve everything; {need}.")

        grid = np.linspace(scores.min() - 1e-6, scores.max() + 1e-6, n_grid)
        best = float(grid[0])
        for tau in grid:
            skipped = scores < tau
            total_loss = float(per_question_loss[skipped].sum())
            if bound(total_loss) <= alpha:
                best = float(tau)
            else:
                break
        self.threshold = best
        return best

    def calibrate_budget(self, scores: np.ndarray, budget: float) -> float:
        """Threshold that retrieves exactly the top `budget` fraction."""
        scores = np.asarray(scores, dtype=float)
        if budget <= 0:
            self.threshold = float(scores.max() + 1.0)
        elif budget >= 1:
            self.threshold = float(scores.min() - 1.0)
        else:
            self.threshold = float(np.quantile(scores, 1.0 - budget))
        return self.threshold

    def decide(self, features: np.ndarray,
               threshold: Optional[float] = None) -> GateDecision:
        scores = self.score(features)
        tau = self.threshold if threshold is None else threshold
        return GateDecision(retrieve=scores >= tau, scores=scores, threshold=tau)

    # ------------------------------------------------------------------- io
    def save(self, path: str) -> None:
        import joblib
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "threshold": self.threshold,
             "classes": self.classes_},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "UtilityGate":
        import joblib

        blob = joblib.load(path)
        gate = cls.__new__(cls)
        gate.pipeline = blob["pipeline"]
        gate.threshold = blob["threshold"]
        gate.classes_ = blob["classes"]
        return gate


class DifficultyGate(UtilityGate):
    """Ablation baseline: the same probe trained on binary difficulty labels.

    Label is 1 when the closed-book answer is wrong, which is what a
    complexity-style router predicts. Comparing this against UtilityGate on
    identical features is the cleanest test of the signed-utility claim.
    """

    def fit(self, features: np.ndarray, closed_book_correct: np.ndarray
            ) -> "DifficultyGate":
        labels = (1 - np.asarray(closed_book_correct).astype(int))
        self.pipeline.fit(features, labels)
        self.classes_ = list(self.pipeline.named_steps["mlp"].classes_)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        probs = self.pipeline.predict_proba(features)
        index = {c: i for i, c in enumerate(self.classes_)}
        return probs[:, index[1]] if 1 in index else np.zeros(len(features))


def oracle_scores(correct_retrieve: np.ndarray,
                  correct_skip: np.ndarray) -> np.ndarray:
    """The ceiling: perfect knowledge of the signed utility."""
    return np.asarray(correct_retrieve, dtype=float) - np.asarray(
        correct_skip, dtype=float
    )


def utility_labels(correct_retrieve: Sequence[float],
                   correct_skip: Sequence[float]) -> np.ndarray:
    """u = c_rag - c_closed, clipped to {-1, 0, +1}."""
    c_r = np.asarray(correct_retrieve, dtype=float)
    c_s = np.asarray(correct_skip, dtype=float)
    return np.sign(c_r - c_s).astype(int)


def label_distribution(utility: np.ndarray) -> Dict[str, float]:
    utility = np.asarray(utility).astype(int)
    total = max(1, len(utility))
    return {
        "retrieval_helps_+1": float((utility == 1).sum() / total),
        "retrieval_neutral_0": float((utility == 0).sum() / total),
        "retrieval_hurts_-1": float((utility == -1).sum() / total),
    }
