# slm-retrieval-confidence-starter

Starter code for **PreCoG: pre-retrieval confidence gating for small language
model RAG**, the project in Sarang's Vizuara Research Starter Kit.

## The one-sentence idea

Before a small language model calls a retriever, it reads the question once
and predicts whether retrieving will actually help. If the answer is already
in its weights, it skips retrieval entirely. If evidence is likely to rescue
it, it retrieves. The prediction costs a single prompt-only forward pass, and
the threshold that turns the prediction into a decision comes with a bound on
how much accuracy you are risking.

Most adaptive-retrieval systems decide *during* generation, after they have
already paid for a draft. This one decides *before*, and it predicts the
**signed utility** of retrieval, `u = correct_with_RAG - correct_closed_book`,
which separates three different things:

| `u` | meaning | what a gate should do |
| --- | --- | --- |
| `+1` | retrieval rescues a wrong parametric answer | retrieve |
| `0` | retrieval changes nothing | skip, it is pure cost |
| `-1` | retrieval breaks a correct parametric answer | skip, it is actively harmful |

A question-difficulty classifier collapses the bottom two rows. That is the
gap this project goes after.

## Setup

Requires Python 3.10 or newer. A GPU makes stage 1 much faster but is not
required for the smoke test.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Or by hand:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## The first three things to run

**1. Prove the wiring works (about ten minutes, laptop is fine).**

```bash
bash scripts/run_smoke.sh
```

This runs all three stages with a 0.5B model, 20k passages and 240 questions.
The numbers mean nothing. What you are checking is that
`artifacts/smoke/results.md` exists and has a table in it.

**2. Look at the utility label distribution.**

```bash
python -c "
import json;
print(json.load(open('artifacts/smoke/label_summary.json'))['splits']['train'])"
```

This single number is the whole project in miniature. If almost every question
is `u = 0`, your retriever is not returning anything useful, or your prompts
are too loose, and no gate will save you. Fix that before anything else.

**3. Run the real baseline.**

```bash
bash scripts/run_baseline.sh configs/default.yaml
```

1.5B backbone, BM25 over 200k Wikipedia passages, NQ-Open plus TriviaQA.
Budget a few hours the first time. Stage 1 does all the work; stages 2 and 3
finish in seconds afterwards, so you can iterate on the gate quickly.

## How the pipeline is organised

Three stages, run in order, each reading the previous one's output.

```
stage 1  precog.label_utility   run BOTH branches, extract signals, save labels
stage 2  precog.train_gate      fit the probe on train, set tau on calib
stage 3  precog.evaluate        route the test split, all baselines, budget sweep
```

The split discipline is the part that matters most:

- `train` fits the probe.
- `calib` sets the threshold and the probe has **never** seen it.
- `test` is reported.

If you ever calibrate on data the probe trained on, the risk guarantee in the
paper becomes meaningless and a reviewer will catch it.

## Layout

```
configs/         default.yaml (the real run), smoke.yaml (the first run)
scripts/         setup.sh, run_smoke.sh, run_baseline.sh, run_ablations.sh
src/precog/
  config.py         one YAML drives every stage
  data.py           NQ-Open, TriviaQA, PopQA, HotpotQA loaders
  retriever.py      BM25 (default) and an optional dense variant
  generator.py      the frozen small LM: closed-book and RAG answering
  signals.py        the three query-only confidence signals
  gate.py           the utility probe, the risk-controlled threshold
  metrics.py        EM, F1, ECE, AUROC, budget curve
  label_utility.py  stage 1
  train_gate.py     stage 2
  evaluate.py       stage 3
experiments/     log_template.md, your running experiment log
artifacts/       generated: indexes, labels, features, gates, results
```

No data files are committed. Every loader downloads a public dataset from the
Hugging Face hub on first use and caches it.

## Mapping code to the paper

| Paper | Code |
| --- | --- |
| Section 3.2, query-only signals | `signals.SignalExtractor` |
| Section 3.3, signed utility labels | `gate.utility_labels`, `label_utility.py` |
| Section 3.4, risk-controlled threshold | `gate.UtilityGate.calibrate_risk` |
| Section 3.6, baselines | `evaluate.main`, the `matched` dict |
| Table 1 | `artifacts/<run>/results.md` |
| Figure 2(a), budget curve | `artifacts/<run>/budget_curve.json` |
| Section 5, label formulation ablation | `gate.DifficultyGate` |

## Things that will go wrong, and what they mean

- **All utility labels are 0.** The retriever is not helping. Check that
  `max_docs` is large enough and print a few retrieved passages by hand.
- **`p_yes` is 0.99 for everything.** Your backbone is overconfident, which is
  a known failure mode of self-assessed retrieval gating. This is a finding,
  not a bug: report it, and lean on the hidden-state probe instead.
- **The risk-controlled threshold retrieves everything.** `alpha` is too tight
  for the calibration split you have. `calibrate_risk` prints the tightest
  bound achievable at your `n` and `delta`; if that floor is already above
  `alpha`, no threshold can pass. Raise `alpha`, raise `delta`, or grow
  `calib_size`.
- **Out of memory in stage 1.** Lower the batch sizes in
  `label_utility._batched_answer` and `signals.SignalExtractor.extract`, or
  drop `retriever.top_k`.

## License and citation

Add a license before you publish. When you post the preprint, put the arXiv
link here and in `CLAUDE.md`.
