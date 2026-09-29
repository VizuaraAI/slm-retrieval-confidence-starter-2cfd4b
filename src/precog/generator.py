"""The frozen small language model: closed-book and RAG answering.

Nothing here is trained. The model is loaded once and used for three things:
answering without context, answering with context, and (in signals.py) exposing
its hidden states. Keeping all three on one loaded model is what makes the cost
comparison in the paper honest.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CLOSED_BOOK_PROMPT = (
    "Answer the question with a short factual answer of at most five words. "
    "If you do not know, answer with your best guess.\n\n"
    "Question: {question}\nAnswer:"
)

RAG_PROMPT = (
    "Use the passages below to answer the question with a short factual answer "
    "of at most five words. If the passages do not contain the answer, rely on "
    "what you know.\n\n"
    "{context}\n"
    "Question: {question}\nAnswer:"
)

SELF_KNOWLEDGE_PROMPT = (
    "Question: {question}\n"
    "Do you know the answer to this question from memory? Answer yes or no.\n"
    "Answer:"
)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SmallLM:
    def __init__(self, name: str, dtype: str = "bfloat16", device: str = "auto",
                 max_new_tokens: int = 24):
        self.device = pick_device(device)
        torch_dtype = _DTYPES.get(dtype, torch.float32)
        if self.device == "cpu":
            torch_dtype = torch.float32  # bf16 on CPU is slow and fiddly
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch_dtype, output_hidden_states=True
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.n_layers = self.model.config.num_hidden_layers

    def format_chat(self, prompt: str) -> str:
        """Use the chat template when the checkpoint has one."""
        template = getattr(self.tokenizer, "chat_template", None)
        if not template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_context(self, passages: Sequence) -> str:
        lines = []
        for i, p in enumerate(passages, start=1):
            text = p.render() if hasattr(p, "render") else str(p)
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> List[str]:
        encoded = self.tokenizer(
            [self.format_chat(p) for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        output = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output[:, encoded["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [self._clean(t) for t in texts]

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip().split("\n")[0].strip()
        for prefix in ("Answer:", "A:"):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text

    def answer_closed_book(self, questions: List[str]) -> Dict[str, object]:
        prompts = [CLOSED_BOOK_PROMPT.format(question=q) for q in questions]
        start = time.perf_counter()
        answers = self.generate(prompts)
        elapsed = (time.perf_counter() - start) * 1000.0 / max(1, len(questions))
        return {"answers": answers, "latency_ms": elapsed}

    def answer_with_context(self, questions: List[str],
                            contexts: List[str]) -> Dict[str, object]:
        prompts = [
            RAG_PROMPT.format(question=q, context=c)
            for q, c in zip(questions, contexts)
        ]
        start = time.perf_counter()
        answers = self.generate(prompts)
        elapsed = (time.perf_counter() - start) * 1000.0 / max(1, len(questions))
        return {"answers": answers, "latency_ms": elapsed}


def load_model(cfg, override_name: Optional[str] = None) -> SmallLM:
    return SmallLM(
        name=override_name or cfg.model.name,
        dtype=cfg.model.dtype,
        device=cfg.model.device,
        max_new_tokens=cfg.model.max_new_tokens,
    )
