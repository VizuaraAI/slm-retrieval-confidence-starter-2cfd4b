"""Query-only confidence signals: the features the gate sees.

This is the heart of the project. Everything here must be computable from the
question alone, with no generation and no retrieval call, or the cost argument
in the paper collapses.

Three signals per question:
    hidden   last-token hidden state at layer l of the closed-book prompt
    p_yes    P(yes) over {yes, no} on a self-knowledge probe prompt
    entropy  Shannon entropy of the next-token distribution

Both prompts go through the model in a single batched forward pass.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .generator import CLOSED_BOOK_PROMPT, SELF_KNOWLEDGE_PROMPT, SmallLM


def _yes_no_token_ids(tokenizer) -> Dict[str, List[int]]:
    """Collect the token ids that plausibly spell yes / no for this tokenizer."""
    out = {"yes": [], "no": []}
    for word, variants in (
        ("yes", ["yes", " yes", "Yes", " Yes", "YES", " YES"]),
        ("no", ["no", " no", "No", " No", "NO", " NO"]),
    ):
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                out[word].append(ids[0])
    # Deduplicate while keeping order.
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


class SignalExtractor:
    def __init__(self, lm: SmallLM, probe_layer_frac: float = 0.6):
        self.lm = lm
        self.layer = max(1, min(lm.n_layers, int(round(probe_layer_frac * lm.n_layers))))
        self.yes_no = _yes_no_token_ids(lm.tokenizer)
        if not self.yes_no["yes"] or not self.yes_no["no"]:
            raise RuntimeError(
                "could not find single-token yes/no for this tokenizer; "
                "set gate.features to drop 'p_yes' or pick another backbone"
            )

    @torch.no_grad()
    def _forward(self, prompts: List[str]):
        enc = self.lm.tokenizer(
            [self.lm.format_chat(p) for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.lm.device)
        out = self.lm.model(**enc, output_hidden_states=True)
        # Left padding means the final column is always the real last token.
        return out, enc

    @torch.no_grad()
    def extract_batch(self, questions: List[str]) -> List[Dict[str, np.ndarray]]:
        answer_prompts = [CLOSED_BOOK_PROMPT.format(question=q) for q in questions]
        probe_prompts = [SELF_KNOWLEDGE_PROMPT.format(question=q) for q in questions]

        n = len(questions)
        out, _ = self._forward(answer_prompts + probe_prompts)

        hidden_states = out.hidden_states[self.layer][:, -1, :].float().cpu().numpy()
        logits = out.logits[:, -1, :].float()

        # Answer-prompt half: hidden state and predictive entropy.
        answer_hidden = hidden_states[:n]
        answer_logits = logits[:n]
        log_probs = torch.log_softmax(answer_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()

        # Probe-prompt half: renormalised P(yes) over {yes, no}.
        probe_logits = logits[n:]
        probe_probs = torch.softmax(probe_logits, dim=-1)
        p_yes_raw = probe_probs[:, self.yes_no["yes"]].sum(dim=-1)
        p_no_raw = probe_probs[:, self.yes_no["no"]].sum(dim=-1)
        p_yes = (p_yes_raw / (p_yes_raw + p_no_raw + 1e-9)).cpu().numpy()

        return [
            {
                "hidden": answer_hidden[i],
                "p_yes": np.float32(p_yes[i]),
                "entropy": np.float32(entropy[i]),
            }
            for i in range(n)
        ]

    def extract(self, questions: List[str], batch_size: int = 8,
                verbose: bool = True) -> List[Dict[str, np.ndarray]]:
        results = []
        for start in range(0, len(questions), batch_size):
            chunk = questions[start:start + batch_size]
            results.extend(self.extract_batch(chunk))
            if verbose and start % (batch_size * 10) == 0:
                print(f"  signals: {start + len(chunk)}/{len(questions)}", flush=True)
        return results


def stack_features(signals: List[Dict[str, np.ndarray]],
                   use: List[str]) -> np.ndarray:
    """Concatenate the selected signals into one design matrix."""
    columns = []
    if "hidden" in use:
        columns.append(np.stack([s["hidden"] for s in signals]).astype("float32"))
    if "p_yes" in use:
        columns.append(np.array([[s["p_yes"]] for s in signals], dtype="float32"))
    if "entropy" in use:
        columns.append(np.array([[s["entropy"]] for s in signals], dtype="float32"))
    if not columns:
        raise ValueError("no features selected; check gate.features in the config")
    return np.concatenate(columns, axis=1)
