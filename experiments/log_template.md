# Experiment log

Copy this file to `experiments/log.md` on day one and append to it after every
run. Never edit a past entry; if you were wrong, add a new entry saying so.

The columns below are deliberately the same as Table 1 in the paper, so that
filling this log in fills the paper in. `artifacts/<run_name>/results.md` is
generated with exactly these columns, so you can paste rather than retype.

---

## Main results table (maps to paper Table 1)

Backbone: `___`  Retriever: `___`  Test set: `___` (n = ___)

| Method | EM | F1 | Retrieval rate `r` | ms/q | AURC | Run name | Date |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Never retrieve (closed-book) | | | 0.00 | | n/a | | |
| Always retrieve | | | 1.00 | | n/a | | |
| Random gate (matched budget) | | | | | | | |
| Verbalized confidence | | | | | | | |
| First-token entropy only | | | | | | | |
| SKR-style kNN | | | | | | | |
| Query complexity (difficulty labels) | | | | | | | |
| Generation-time trigger | | | | | | | |
| **PreCoG (budget-matched)** | | | | | | | |
| **PreCoG (risk-controlled, alpha=0.02)** | | | | | | | |
| Oracle utility gate | | | | | | | |

## Per-dataset breakdown (maps to paper Figure 2b)

| Dataset | Never | Always | PreCoG | Oracle | PreCoG `r` | Run name |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NQ-Open | | | | | | |
| TriviaQA | | | | | | |
| PopQA | | | | | | |
| HotpotQA | | | | | | |

## Utility label distribution (the health check)

If `-1` is empty, the signed-utility claim is in trouble. Find out early.

| Dataset | `+1` rescues | `0` redundant | `-1` harmful | n | Run name |
| --- | ---: | ---: | ---: | ---: | --- |
| | | | | | |

## Risk control (maps to paper H3)

| alpha | delta | tau | Calib `r` | Test `r` | Realized loss | Held? | Run name |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.02 | 0.1 | | | | | | |
| 0.05 | 0.1 | | | | | | |

## Ablations (maps to paper Section 5)

| Ablation | Setting | EM at matched `r` | AURC | Delta vs full | Run name |
| --- | --- | ---: | ---: | ---: | --- |
| Signals | hidden only | | | | |
| Signals | p_yes only | | | | |
| Signals | entropy only | | | | |
| Signals | hidden + p_yes | | | | |
| Signals | all three (full) | | | | |
| Probe layer | 0.2 L | | | | |
| Probe layer | 0.6 L (default) | | | | |
| Probe layer | 1.0 L | | | | |
| Labels | signed utility (ours) | | | | |
| Labels | binary difficulty | | | | |
| Backbone | Qwen2.5-0.5B | | | | |
| Backbone | Qwen2.5-1.5B | | | | |
| Backbone | Qwen2.5-3B | | | | |
| Backbone | SmolLM2-1.7B | | | | |
| Transfer | fit NQ, test PopQA | | | | |
| Transfer | fit NQ, test HotpotQA | | | | |
| Retriever | BM25 (default) | | | | |
| Retriever | dense bge-small | | | | |
| Retriever | degraded index | | | | |

---

## Run journal

One entry per run. Keep it short and honest.

### YYYY-MM-DD, run `___`

- **Config**: `configs/___.yaml`, commit `___`
- **Question this run answers**: one sentence.
- **Result**: the number, and whether it moved in the predicted direction.
- **Surprise**: anything you did not expect. This is the most valuable line in
  the whole log and it is usually where the paper's real contribution comes
  from.
- **Next**: the single next run this implies.

### YYYY-MM-DD, run `___`

- **Config**:
- **Question this run answers**:
- **Result**:
- **Surprise**:
- **Next**:
