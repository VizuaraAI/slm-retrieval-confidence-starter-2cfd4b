"""Open-domain QA datasets.

Every loader returns the same record shape so the rest of the pipeline never
has to care which benchmark it is looking at:

    {"qid": str, "dataset": str, "question": str, "answers": [str, ...]}

PopQA is the interesting one for this project: it is stratified by entity
popularity, so it contains many questions a 1.5B model has genuinely never
memorised. TriviaQA is the opposite end, which is where a gate should save the
most retrieval calls.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from datasets import load_dataset

SUPPORTED = ["nq_open", "trivia_qa", "popqa", "hotpot_qa"]


def _qid(dataset: str, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}-{digest}"


def _record(dataset: str, question: str, answers: List[str]) -> Dict:
    answers = [a for a in dict.fromkeys(a.strip() for a in answers) if a]
    return {
        "qid": _qid(dataset, question),
        "dataset": dataset,
        "question": question.strip(),
        "answers": answers,
    }


def load_nq_open(split: str, limit: int) -> List[Dict]:
    ds = load_dataset("google-research-datasets/nq_open", split=split)
    ds = ds.select(range(min(limit, len(ds))))
    return [_record("nq_open", r["question"], list(r["answer"])) for r in ds]


def load_trivia_qa(split: str, limit: int) -> List[Dict]:
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    ds = ds.select(range(min(limit, len(ds))))
    out = []
    for r in ds:
        aliases = list(r["answer"].get("normalized_aliases") or [])
        answers = [r["answer"]["value"]] + aliases
        out.append(_record("trivia_qa", r["question"], answers))
    return out


def load_popqa(split: str, limit: int) -> List[Dict]:
    # PopQA ships a single "test" split; we slice it ourselves upstream.
    ds = load_dataset("akariasai/PopQA", split="test")
    ds = ds.select(range(min(limit, len(ds))))
    out = []
    for r in ds:
        possible = r.get("possible_answers")
        if isinstance(possible, str):
            # stored as a JSON-encoded list of strings
            import json

            try:
                answers = json.loads(possible)
            except json.JSONDecodeError:
                answers = [possible]
        else:
            answers = list(possible or [])
        answers = [r["obj"]] + answers if r.get("obj") else answers
        out.append(_record("popqa", r["question"], answers))
    return out


def load_hotpot_qa(split: str, limit: int) -> List[Dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split,
                      trust_remote_code=True)
    ds = ds.select(range(min(limit, len(ds))))
    return [_record("hotpot_qa", r["question"], [r["answer"]]) for r in ds]


_LOADERS = {
    "nq_open": load_nq_open,
    "trivia_qa": load_trivia_qa,
    "popqa": load_popqa,
    "hotpot_qa": load_hotpot_qa,
}

# PopQA has no train split, so it is evaluation-only by default.
_SPLIT_MAP = {
    "nq_open": {"train": "train", "test": "validation"},
    "trivia_qa": {"train": "train", "test": "validation"},
    "popqa": {"train": "test", "test": "test"},
    "hotpot_qa": {"train": "train", "test": "validation"},
}


def load_qa(name: str, split: str, limit: int) -> List[Dict]:
    """Load `limit` records from one benchmark. `split` is "train" or "test"."""
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}, expected one of {SUPPORTED}")
    hf_split = _SPLIT_MAP[name][split]
    return _LOADERS[name](hf_split, limit)


def build_splits(datasets: List[str], train_size: int, calib_size: int,
                 test_size: int, seed: int = 13) -> Dict[str, List[Dict]]:
    """Make the three disjoint splits the pipeline needs.

    train: fits the probe.
    calib: never seen by the probe, used only to pick the threshold. Keeping it
           separate is what makes the risk guarantee meaningful.
    test:  reported numbers.
    """
    import random as _random

    rng = _random.Random(seed)
    per_ds_train = max(1, (train_size + calib_size) // max(1, len(datasets)))
    per_ds_test = max(1, test_size // max(1, len(datasets)))

    pool, test = [], []
    for name in datasets:
        pool.extend(load_qa(name, "train", per_ds_train))
        test.extend(load_qa(name, "test", per_ds_test))

    rng.shuffle(pool)
    rng.shuffle(test)

    # PopQA reuses one split, so drop any test question that leaked into pool.
    test_ids = {r["qid"] for r in test}
    pool = [r for r in pool if r["qid"] not in test_ids]

    return {
        "train": pool[:train_size],
        "calib": pool[train_size:train_size + calib_size],
        "test": test[:test_size],
    }
