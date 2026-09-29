"""Stage 1: build the signed retrieval-utility dataset.

For every question in every split we run BOTH branches once, score both, and
store the pair. This is the only expensive stage, and you only pay it once per
(backbone, retriever) combination. Everything downstream reads its output.

Output per split, under artifacts/<run_name>/:
    <split>_labels.jsonl   one row per question with both answers and both scores
    <split>_features.npy   the design matrix from signals.py
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .config import base_parser, load_config, run_dir, save_json, save_jsonl, set_seed
from .data import build_splits
from .gate import label_distribution, utility_labels
from .generator import load_model
from .metrics import contains_match, exact_match, token_f1
from .retriever import build_retriever
from .signals import SignalExtractor, stack_features


def process_split(name: str, records: List[Dict], lm, retriever, extractor,
                  cfg) -> Dict[str, object]:
    print(f"\n[{name}] {len(records)} questions", flush=True)
    questions = [r["question"] for r in records]

    print(f"[{name}] extracting query-only signals", flush=True)
    signals = extractor.extract(questions, batch_size=8)
    features = stack_features(signals, cfg.gate.features)

    print(f"[{name}] retrieving top-{cfg.retriever.top_k} passages", flush=True)
    contexts = [
        lm.build_context(retriever.search(q, top_k=cfg.retriever.top_k))
        for q in questions
    ]

    print(f"[{name}] answering closed-book", flush=True)
    closed = _batched_answer(lm, questions, None, batch_size=8)

    print(f"[{name}] answering with retrieved context", flush=True)
    rag = _batched_answer(lm, questions, contexts, batch_size=4)

    rows = []
    for i, record in enumerate(records):
        answers = record["answers"]
        rows.append({
            **record,
            "answer_closed": closed["answers"][i],
            "answer_rag": rag["answers"][i],
            "em_closed": exact_match(closed["answers"][i], answers),
            "em_rag": exact_match(rag["answers"][i], answers),
            "contains_closed": contains_match(closed["answers"][i], answers),
            "contains_rag": contains_match(rag["answers"][i], answers),
            "f1_closed": token_f1(closed["answers"][i], answers),
            "f1_rag": token_f1(rag["answers"][i], answers),
            "p_yes": float(signals[i]["p_yes"]),
            "entropy": float(signals[i]["entropy"]),
            "latency_closed_ms": closed["latency_ms"],
            "latency_rag_ms": rag["latency_ms"],
        })

    utility = utility_labels([r["em_rag"] for r in rows],
                             [r["em_closed"] for r in rows])
    for row, u in zip(rows, utility):
        row["utility"] = int(u)

    out = run_dir(cfg)
    save_jsonl(rows, os.path.join(out, f"{name}_labels.jsonl"))
    np.save(os.path.join(out, f"{name}_features.npy"), features)

    dist = label_distribution(utility)
    print(f"[{name}] utility distribution: {dist}", flush=True)
    return {"n": len(rows), "distribution": dist,
            "em_closed": float(np.mean([r["em_closed"] for r in rows])),
            "em_rag": float(np.mean([r["em_rag"] for r in rows]))}


def _batched_answer(lm, questions: List[str], contexts, batch_size: int):
    answers, latencies = [], []
    for start in range(0, len(questions), batch_size):
        chunk_q = questions[start:start + batch_size]
        if contexts is None:
            result = lm.answer_closed_book(chunk_q)
        else:
            result = lm.answer_with_context(
                chunk_q, contexts[start:start + batch_size]
            )
        answers.extend(result["answers"])
        latencies.append(result["latency_ms"])
        if start % (batch_size * 10) == 0:
            print(f"  answered {start + len(chunk_q)}/{len(questions)}", flush=True)
    return {"answers": answers, "latency_ms": float(np.mean(latencies))}


def main() -> None:
    parser = base_parser("Build signed retrieval-utility labels.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.data.seed)

    print("loading splits from the hub", flush=True)
    splits = build_splits(
        cfg.data.datasets, cfg.data.train_size, cfg.data.calib_size,
        cfg.data.test_size, cfg.data.seed,
    )

    print(f"loading backbone {cfg.model.name}", flush=True)
    lm = load_model(cfg)
    extractor = SignalExtractor(lm, cfg.model.probe_layer_frac)
    print(f"probing layer {extractor.layer} of {lm.n_layers}", flush=True)

    print("building retriever (first run downloads and indexes the corpus)",
          flush=True)
    retriever = build_retriever(cfg)

    summary = {"config": cfg.to_dict(), "probe_layer": extractor.layer, "splits": {}}
    for name in ["train", "calib", "test"]:
        summary["splits"][name] = process_split(
            name, splits[name], lm, retriever, extractor, cfg
        )

    save_json(summary, os.path.join(run_dir(cfg), "label_summary.json"))
    print("\ndone. next: python -m precog.train_gate --config", args.config)


if __name__ == "__main__":
    main()
