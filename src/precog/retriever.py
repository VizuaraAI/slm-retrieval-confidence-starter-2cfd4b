"""Retrieval over a Wikipedia passage subset.

BM25 is the default on purpose. It needs no GPU, no Java (we use the pure
Python `bm25s` package), it indexes 200k passages in a couple of minutes, and
it makes your results reproducible on a laptop. Swap in the dense retriever
once the BM25 pipeline works end to end, not before.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Passage:
    doc_id: str
    title: str
    text: str

    def render(self) -> str:
        return f"{self.title}: {self.text}" if self.title else self.text


def load_corpus(dataset: str, config: str, max_docs: int) -> List[Passage]:
    """Stream a passage corpus from the hub and stop after `max_docs`."""
    from datasets import load_dataset

    stream = load_dataset(dataset, config, split="corpus", streaming=True)
    passages = []
    for i, row in enumerate(stream):
        if i >= max_docs:
            break
        text = (row.get("text") or "").strip()
        if not text:
            continue
        passages.append(
            Passage(
                doc_id=str(row.get("_id", i)),
                title=(row.get("title") or "").strip(),
                text=text,
            )
        )
    return passages


class BM25Retriever:
    """Thin wrapper over bm25s so the rest of the code sees one interface."""

    def __init__(self, passages: Sequence[Passage]):
        import bm25s

        self.passages = list(passages)
        corpus = [p.render() for p in self.passages]
        self.tokenizer = bm25s.tokenize
        tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
        self.index = bm25s.BM25()
        self.index.index(tokens, show_progress=False)

    def search(self, query: str, top_k: int = 5) -> List[Passage]:
        import bm25s

        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        k = min(top_k, len(self.passages))
        results, _ = self.index.retrieve(tokens, k=k, show_progress=False)
        return [self.passages[int(i)] for i in results[0]]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.index.save(os.path.join(path, "bm25"))
        with open(os.path.join(path, "passages.jsonl"), "w", encoding="utf-8") as fh:
            for p in self.passages:
                fh.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        import bm25s

        obj = cls.__new__(cls)
        obj.index = bm25s.BM25.load(os.path.join(path, "bm25"))
        obj.tokenizer = bm25s.tokenize
        obj.passages = []
        with open(os.path.join(path, "passages.jsonl"), "r", encoding="utf-8") as fh:
            for line in fh:
                obj.passages.append(Passage(**json.loads(line)))
        return obj


class DenseRetriever:
    """Optional dense variant. Used by the retriever-quality ablation."""

    def __init__(self, passages: Sequence[Passage], model_name: str):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.passages = list(passages)
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            [p.render() for p in self.passages],
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.embeddings = np.asarray(embeddings, dtype="float32")

    def search(self, query: str, top_k: int = 5) -> List[Passage]:
        import numpy as np

        q = self.model.encode([query], normalize_embeddings=True)
        scores = np.asarray(q, dtype="float32") @ self.embeddings.T
        idx = np.argsort(-scores[0])[:top_k]
        return [self.passages[int(i)] for i in idx]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "dense.pkl"), "wb") as fh:
            pickle.dump({"passages": self.passages, "embeddings": self.embeddings}, fh)


def build_retriever(cfg) -> object:
    """Build (or reuse) the retriever described by the config."""
    index_dir = cfg.retriever.index_dir
    if cfg.retriever.kind == "bm25" and os.path.exists(
        os.path.join(index_dir, "passages.jsonl")
    ):
        return BM25Retriever.load(index_dir)

    passages = load_corpus(
        cfg.retriever.corpus_dataset,
        cfg.retriever.corpus_config,
        cfg.retriever.max_docs,
    )
    if cfg.retriever.kind == "bm25":
        retriever = BM25Retriever(passages)
        retriever.save(index_dir)
        return retriever
    if cfg.retriever.kind == "dense":
        return DenseRetriever(passages, cfg.retriever.dense_model)
    raise ValueError(f"unknown retriever kind {cfg.retriever.kind!r}")
