# Logbooks

These track how data mixing, training and evaluation evolved: decisions made,
what real runs revealed, fixes applied, and limitations left open. The goal is
that the next session (human or agent) doesn't have to re-derive a quirk we
already found. `git log` has the diffs; logbooks have the *why* and the numbers.

## Convention: one logbook per module

A logbook maps 1:1 to a module or workflow, not to a single run. Training runs
go in [`../experiments/`](../experiments/), one file per experiment, with a
comparison table in its README.

- [`data-mixing.md`](data-mixing.md) -- `scripts/mix_translation_data.py`, `data.md` (bidirectional mos/dyu/fra/eng mix, source filters, decontamination, validation sets).
- [`training.md`](training.md) -- `train/train_dense.py`, `train/modal_single_gpu.py` (dense LoRA training, Modal launcher, chat template and loss mask).
- [`evaluation.md`](evaluation.md) -- `scripts/evaluate_translation.py`, `train/modal_vllm_server.py` (vLLM on Modal, chrF++/BLEU per direction on FLORES+ and moore-web).

Add a new logbook when a new module or workflow is added, not for each run.

## Entry format

Dated, terse, newest entry on top. Prefer "what we learned / decided" over
"what the diff was". Give scores with their dataset, split, number of
examples, direction and checkpoint, or they can't be compared later.

```markdown
## 2026-10-27

- Finding or decision, one or two sentences.
- Why it matters / what to check next.
```
