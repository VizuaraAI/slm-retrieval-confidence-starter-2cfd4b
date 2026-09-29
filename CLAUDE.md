# CLAUDE.md

Instructions for the coding agent working in this repository with Sarang.

## What this project is

PreCoG: a pre-retrieval confidence gate for small language model RAG. Given a
question, a frozen 0.5B to 3B decoder is run once over the prompt with no
generation; three query-only signals are read out of that forward pass; a small
probe predicts the **signed utility** of retrieving; a calibrated threshold
turns that prediction into a skip-or-retrieve decision with a bound on the
accuracy risked.

The research claim has three parts, and every change should be judged against
them:

1. The decision can be made **before** retrieval, from prompt-encoding signals
   alone. If a change makes the gate depend on generated tokens or on retrieved
   passages, it breaks the central cost argument. Say so loudly.
2. **Signed utility** (`+1` rescue, `0` redundant, `-1` harmful) beats binary
   difficulty. `gate.DifficultyGate` exists specifically to be beaten.
3. The threshold is **calibrated, not tuned**. It is chosen on a split the
   probe never saw, with a distribution-free bound.

## Repository layout

```
configs/         YAML configs, one per experiment, never edited in place
scripts/         shell entry points
src/precog/      the package (see README for the per-module map)
experiments/     log_template.md and the running experiment log
artifacts/       generated output, gitignored
```

Three stages run in order: `precog.label_utility` (expensive, calls the model),
`precog.train_gate` (seconds), `precog.evaluate` (seconds, pure arithmetic over
stage-1 outputs).

## Conventions

- **Config-driven.** Nothing is hard-coded in a script. If you need a new knob,
  add it to the dataclasses in `config.py` and to `configs/default.yaml` with a
  comment explaining the default.
- **One run, one `run_name`.** To change something, copy a config, rename the
  run, change one thing. Never mutate a config that already has results logged
  against it.
- **Stage 1 is sacred.** It writes `*_labels.jsonl` and `*_features.npy`.
  Everything downstream is cheap because it reads those. Do not add model calls
  to stages 2 or 3.
- **Split discipline.** `train` fits the probe, `calib` sets the threshold and
  is never seen by the probe, `test` is reported. If a change would leak
  calibration data into probe training, refuse and explain why.
- **Seed everything.** `set_seed(cfg.data.seed)` at the top of every entry
  point, and pass `random_state` to every sklearn object.
- **No em dashes in prose.** Use commas, colons, parentheses or semicolons.
- Python: standard library first, then third party, then local. Type hints on
  public functions. Docstrings that say *why*, not *what*. Keep modules small
  enough to read in one sitting; that is a feature, not an accident.
- Prefer editing an existing module over adding a new one. This codebase should
  stay small enough that Sarang can hold all of it in his head.

## What to do when asked to add an experiment

1. Write the config first, with a `run_name` that describes the change.
2. Check whether stage 1 has to rerun. It does if the model, the retriever, the
   data or the feature set changed. It does not if only the gate, the threshold
   or the budgets changed.
3. Run it, then append a row to `experiments/log.md` using the template. The
   row's columns are the same as the paper's Table 1 on purpose.
4. Say plainly whether the result supports or undercuts the claim. A result
   that undercuts it is more useful than one that does not, and the paper has a
   Limitations section waiting for it.

## Near-term task list

These are in rough order. Each is a self-contained session.

1. **Make the smoke test pass end to end** and eyeball ten questions from
   `artifacts/smoke/train_labels.jsonl`. Are the closed-book answers sane? Are
   the retrieved passages relevant? Nothing else matters until this is true.
2. **Report the utility label distribution per dataset.** If `u = -1` is
   essentially empty, the signed-utility claim is in trouble and the project
   should pivot toward the cost story. Find this out in week 2, not week 7.
3. **Refactor `stack_features`** so stage 1 always saves all three signal
   blocks plus a column index, and `train_gate` slices columns. This makes the
   signal ablation free instead of a full stage-1 rerun.
4. **Add the layer sweep**: loop `probe_layer_frac` over 0.2 to 1.0 and plot
   gate AUROC against depth. Cheap once task 3 is done (save all layers).
5. **Add PopQA and HotpotQA** to `data.datasets` and report per-dataset
   retrieval rates. This is the plot that carries the paper.
6. **Add the Adaptive-RAG-style baseline properly**: a classifier over a
   separate sentence encoder, not over the LM's hidden states, so the
   comparison isolates the label definition rather than the feature source.
7. **Add bootstrap confidence intervals** to `evaluate.py`. Every number in the
   paper needs one, and reviewers will ask.
8. **Add a `--degrade-retriever` flag** that shuffles a fraction of retrieved
   passages, for the retriever-quality ablation. The utility gate should
   retrieve less as the retriever gets worse; the difficulty gate cannot
   express that. This is the paper's falsifiable qualitative prediction.

## Things to push back on

- Any suggestion to finetune the base model. That is Self-RAG's approach; it is
  out of scope, out of compute budget, and it couples the gate to one backbone.
- Any metric added without also adding it to `experiments/log.md`.
- Any claim in the paper that is not backed by a number in `artifacts/`.
- Scope creep into multi-step or iterative retrieval before the single-step
  result is solid. Write that down as future work instead.

## Paper

The draft lives alongside this repo (`paper/main.tex`). Its Table 1 and
Figure 2 are placeholders marked `XX.X`. As results land, replace them and
update `figures/make_figures.py` with the real numbers rather than editing the
PNGs by hand.
