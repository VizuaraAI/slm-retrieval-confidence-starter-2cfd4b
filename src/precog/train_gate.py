"""Stage 2: fit the utility probe and calibrate its threshold.

The split discipline matters more than the model class. The probe sees `train`
only; the threshold is chosen on `calib`, which the probe has never seen. If
you ever calibrate on data the probe was fit on, the risk bound in the paper
becomes meaningless, and a reviewer will find it.
"""

from __future__ import annotations

import os

import numpy as np

from .config import base_parser, load_config, load_jsonl, run_dir, save_json, set_seed
from .gate import DifficultyGate, UtilityGate, label_distribution
from .metrics import auroc


def load_split(out: str, name: str):
    rows = load_jsonl(os.path.join(out, f"{name}_labels.jsonl"))
    features = np.load(os.path.join(out, f"{name}_features.npy"))
    if len(rows) != len(features):
        raise RuntimeError(
            f"{name}: {len(rows)} label rows but {len(features)} feature rows; "
            "rerun precog.label_utility"
        )
    return rows, features


def main() -> None:
    parser = base_parser("Fit and calibrate the pre-retrieval gate.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    out = run_dir(cfg)

    train_rows, train_x = load_split(out, "train")
    calib_rows, calib_x = load_split(out, "calib")

    train_u = np.array([r["utility"] for r in train_rows])
    print("train utility distribution:", label_distribution(train_u))

    if len(set(train_u.tolist())) < 2:
        raise RuntimeError(
            "the utility labels are all one class, so there is nothing to learn. "
            "Usually this means retrieval never changed the answer: increase "
            "train_size, or check that the retriever is actually returning "
            "relevant passages."
        )

    print(f"fitting utility gate on {len(train_rows)} examples")
    gate = UtilityGate(
        pca_dim=cfg.gate.pca_dim,
        hidden_layer_sizes=cfg.gate.hidden_layer_sizes,
        seed=cfg.data.seed,
    ).fit(train_x, train_u)

    calib_scores = gate.score(calib_x)
    c_r = np.array([r["em_rag"] for r in calib_rows])
    c_s = np.array([r["em_closed"] for r in calib_rows])

    tau = gate.calibrate_risk(calib_scores, c_r, c_s,
                              alpha=cfg.gate.alpha, delta=cfg.gate.delta)
    retrieval_rate = float((calib_scores >= tau).mean())
    print(f"risk-controlled threshold tau = {tau:.4f} "
          f"(alpha={cfg.gate.alpha}, delta={cfg.gate.delta})")
    print(f"calibration retrieval rate = {retrieval_rate:.3f}")

    gate.save(os.path.join(out, "gate_utility.joblib"))

    # Ablation baseline on identical features.
    print("fitting difficulty-labelled ablation gate")
    diff_gate = DifficultyGate(
        pca_dim=cfg.gate.pca_dim,
        hidden_layer_sizes=cfg.gate.hidden_layer_sizes,
        seed=cfg.data.seed,
    ).fit(train_x, np.array([r["em_closed"] for r in train_rows]))
    diff_gate.save(os.path.join(out, "gate_difficulty.joblib"))

    report = {
        "train_size": len(train_rows),
        "calib_size": len(calib_rows),
        "train_utility_distribution": label_distribution(train_u),
        "threshold": tau,
        "alpha": cfg.gate.alpha,
        "delta": cfg.gate.delta,
        "calib_retrieval_rate": retrieval_rate,
        "calib_auroc_helps": auroc(calib_scores,
                                   (c_r - c_s > 0).astype(int)),
        "calib_accuracy_always_retrieve": float(c_r.mean()),
        "calib_accuracy_gated": float(
            np.where(calib_scores >= tau, c_r, c_s).mean()
        ),
    }
    save_json(report, os.path.join(out, "gate_report.json"))
    print("\nreport:", report)
    print("\ndone. next: python -m precog.evaluate --config", args.config)


if __name__ == "__main__":
    main()
