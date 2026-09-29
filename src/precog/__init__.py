"""PreCoG: pre-retrieval confidence gating for small language model RAG.

The package is deliberately small. Each module does one job:

    config.py         load and validate the YAML config
    data.py           load open-domain QA datasets from the Hugging Face hub
    retriever.py      BM25 (and optional dense) retrieval over Wikipedia passages
    generator.py      the frozen small LM: closed-book and RAG answering
    signals.py        query-only confidence signals (the pre-retrieval features)
    gate.py           the calibrated utility gate plus risk-controlled thresholds
    metrics.py        exact match, F1, ECE, AUROC, area under the budget curve
    label_utility.py  stage 1: build retrieval-utility labels
    train_gate.py     stage 2: fit and calibrate the gate
    evaluate.py       stage 3: routed evaluation and the budget sweep
"""

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "retriever",
    "generator",
    "signals",
    "gate",
    "metrics",
]
