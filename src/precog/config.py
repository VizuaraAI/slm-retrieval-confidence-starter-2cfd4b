"""Config loading. One YAML file drives every stage, so runs stay reproducible."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    device: str = "auto"
    max_new_tokens: int = 24
    # Fraction of the depth at which the hidden-state probe is tapped.
    # 0.6 means "60% of the way up the stack", which is a good default for
    # 1B to 3B decoders. The layer sweep ablation replaces this number.
    probe_layer_frac: float = 0.6


@dataclass
class RetrieverConfig:
    kind: str = "bm25"  # "bm25" or "dense"
    corpus_dataset: str = "BeIR/nq"
    corpus_config: str = "corpus"
    max_docs: int = 200_000
    top_k: int = 5
    dense_model: str = "BAAI/bge-small-en-v1.5"
    index_dir: str = "artifacts/index"


@dataclass
class DataConfig:
    datasets: List[str] = field(default_factory=lambda: ["nq_open", "trivia_qa"])
    train_size: int = 1500   # used to fit the gate
    calib_size: int = 500    # used only to calibrate the threshold
    test_size: int = 1000
    seed: int = 13


@dataclass
class GateConfig:
    features: List[str] = field(default_factory=lambda: ["hidden", "p_yes", "entropy"])
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [256])
    pca_dim: int = 128        # compress the hidden state before the probe
    alpha: float = 0.02       # tolerated accuracy loss versus always-retrieve
    delta: float = 0.1        # failure probability of the risk bound
    target_budgets: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )


@dataclass
class Config:
    run_name: str = "default"
    out_dir: str = "artifacts"
    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(defaults: Any, override: Any) -> Any:
    if isinstance(defaults, dict) and isinstance(override, dict):
        out = dict(defaults)
        for key, value in override.items():
            out[key] = _merge(defaults.get(key), value)
        return out
    return override if override is not None else defaults


def load_config(path: str) -> Config:
    """Read a YAML file on top of the dataclass defaults."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    merged = _merge(Config().to_dict(), raw)
    return Config(
        run_name=merged["run_name"],
        out_dir=merged["out_dir"],
        model=ModelConfig(**merged["model"]),
        retriever=RetrieverConfig(**merged["retriever"]),
        data=DataConfig(**merged["data"]),
        gate=GateConfig(**merged["gate"]),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def run_dir(cfg: Config) -> str:
    path = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="path to the YAML config",
    )
    return parser
