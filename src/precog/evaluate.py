"""Stage 3: routed evaluation, baselines and the budget sweep.

This is the script that produces the rows of the paper's main table. Because
stage 1 already recorded BOTH branch outcomes for every test question, routing
evaluation is pure arithmetic: pick a decision per question and read off which
correctness value applies. No model calls happen here, so you can iterate on
gates and thresholds in seconds.

Outputs, under artifacts/<run_name>/:
    results.json   every metric, machine readable
    results.md     a markdown table you can paste into experiments/log.md
    budget_curve.json  points for figure 2 panel (a)
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .config import base_parser, load_config, load_jsonl, run_dir, save_json, set_seed
from .gate import DifficultyGate, UtilityGate, oracle_scores
from .metrics import (
    area_under_budget_curve,
    auroc,
    budget_curve,
    expected_calibration_error,
)
from .train_gate import load_split


def route(scores: np.ndarray, threshold: float, rows: List[Dict],
          name: str) -> Dict[str, float]:
    """Score one routing policy against the recorded branch outcomes."""
    retrieve = np.asarray(scores) >= threshold
    em_r = np.array([r["em_rag"] for r in rows])
    em_s = np.array([r["em_closed"] for r in rows])
    f1_r = np.array([r["f1_rag"] for r in rows])
    f1_s = np.array([r["f1_closed"] for r in rows])
    lat_r = float(np.mean([r["latency_rag_ms"] for r in rows]))
    lat_s = float(np.mean([r["latency_closed_ms"] for r in rows]))

    rate = float(retrieve.mean())
    return {
        "method": name,
        "em": float(np.where(retrieve, em_r, em_s).mean() * 100),
        "f1": float(np.where(retrieve, f1_r, f1_s).mean() * 100),
        "retrieval_rate": rate,
        "latency_ms": rate * lat_r + (1 - rate) * lat_s,
        "auroc_vs_oracle": auroc(scores, (em_r - em_s > 0).astype(int)),
    }


def per_dataset(scores: np.ndarray, threshold: float,
                rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    out = {}
    names = sorted({r["dataset"] for r in rows})
    for name in names:
        idx = [i for i, r in enumerate(rows) if r["dataset"] == name]
        subset = [rows[i] for i in idx]
        out[name] = route(scores[idx], threshold, subset, name)
    return out


def to_markdown(table: List[Dict[str, float]]) -> str:
    header = ("| Method | EM | F1 | Retrieval rate | ms/q | AUROC |\n"
              "| --- | ---: | ---: | ---: | ---: | ---: |\n")
    lines = []
    for row in table:
        lines.append(
            f"| {row['method']} | {row['em']:.1f} | {row['f1']:.1f} | "
            f"{row['retrieval_rate']:.2f} | {row['latency_ms']:.0f} | "
            f"{row['auroc_vs_oracle']:.3f} |"
        )
    return header + "\n".join(lines) + "\n"


def main() -> None:
    parser = base_parser("Evaluate routing policies on the test split.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    out = run_dir(cfg)

    rows, features = load_split(out, "test")
    em_r = np.array([r["em_rag"] for r in rows])
    em_s = np.array([r["em_closed"] for r in rows])

    gate = UtilityGate.load(os.path.join(out, "gate_utility.joblib"))
    diff_gate = DifficultyGate.load(os.path.join(out, "gate_difficulty.joblib"))

    precog = gate.score(features)
    difficulty = diff_gate.score(features)
    verbalized = 1.0 - np.array([r["p_yes"] for r in rows])   # low p_yes: retrieve
    entropy = np.array([r["entropy"] for r in rows])
    rng = np.random.default_rng(cfg.data.seed)
    random_scores = rng.random(len(rows))
    oracle = oracle_scores(em_r, em_s)

    always = float(np.max(np.concatenate([precog, [1.0]]))) + 1.0

    table = [
        route(np.full(len(rows), -np.inf), always, rows, "Never retrieve"),
        route(np.full(len(rows), np.inf), 0.0, rows, "Always retrieve"),
    ]

    # Budget-matched comparisons: every gate gets the same retrieval rate as
    # the risk-controlled PreCoG threshold, which is the only fair comparison.
    target_rate = float((precog >= gate.threshold).mean())
    matched = {
        "Random gate": random_scores,
        "Verbalized confidence": verbalized,
        "First-token entropy": entropy,
        "Query complexity (difficulty labels)": difficulty,
        "PreCoG (budget-matched)": precog,
        "Oracle utility gate": oracle,
    }
    for name, scores in matched.items():
        tau = float(np.quantile(scores, 1.0 - target_rate)) if 0 < target_rate < 1 \
            else (float(scores.min()) - 1 if target_rate >= 1 else float(scores.max()) + 1)
        table.append(route(scores, tau, rows, name))

    table.append(route(precog, gate.threshold, rows,
                       f"PreCoG (risk-controlled, alpha={cfg.gate.alpha})"))

    # Budget sweep for figure 2 panel (a).
    curves = {}
    for name, scores in [("precog", precog), ("difficulty", difficulty),
                         ("verbalized", verbalized), ("random", random_scores),
                         ("oracle", oracle)]:
        points = budget_curve(scores, em_r, em_s, cfg.gate.target_budgets)
        curves[name] = {"points": points, "aurc": area_under_budget_curve(points)}

    # Did the risk guarantee hold on unseen data?
    skipped = precog < gate.threshold
    realized_loss = float(np.maximum(0.0, em_r - em_s)[skipped].sum() / len(rows))

    results = {
        "run_name": cfg.run_name,
        "model": cfg.model.name,
        "n_test": len(rows),
        "table": table,
        "per_dataset": per_dataset(precog, gate.threshold, rows),
        "budget_curves": curves,
        "risk_control": {
            "alpha": cfg.gate.alpha,
            "delta": cfg.gate.delta,
            "threshold": gate.threshold,
            "realized_accuracy_loss": realized_loss,
            "guarantee_held": bool(realized_loss <= cfg.gate.alpha),
        },
        "calibration": {
            "ece_p_yes": expected_calibration_error(
                np.array([r["p_yes"] for r in rows]), em_s
            ),
        },
    }
    save_json(results, os.path.join(out, "results.json"))
    save_json(curves, os.path.join(out, "budget_curve.json"))

    md = to_markdown(table)
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# {cfg.run_name} ({cfg.model.name}, n={len(rows)})\n\n")
        fh.write(md)
        fh.write(f"\nAURC (PreCoG): {curves['precog']['aurc']:.4f}\n")
        fh.write(f"AURC (oracle): {curves['oracle']['aurc']:.4f}\n")
        fh.write(f"\nRisk guarantee held: {results['risk_control']['guarantee_held']}"
                 f" (realized loss {realized_loss:.4f} vs alpha {cfg.gate.alpha})\n")

    print("\n" + md)
    print(f"AURC PreCoG {curves['precog']['aurc']:.4f} | "
          f"oracle {curves['oracle']['aurc']:.4f}")
    print(f"risk guarantee held: {results['risk_control']['guarantee_held']}")
    print(f"\nwrote {os.path.join(out, 'results.md')}")


if __name__ == "__main__":
    main()
