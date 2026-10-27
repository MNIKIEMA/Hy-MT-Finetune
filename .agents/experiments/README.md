# Experiments

One file per experiment (a training run, or a set of runs answering one
question): what was tried, with which settings, what came out, and what to try
next. Raw logs and checkpoints stay in the Modal volumes (workspace
`nikiemamahamadi01`: `hy-mt-data`, `hy-mt-training-output`); these files keep
the summary, the reasoning and the links.

Code findings made during an experiment (bugs, quirks, design decisions) go in
the module's logbook under [`../logbooks/`](../logbooks/), and the experiment
file links to them in one line.

## Convention

- File name: `YYYY-MM-DD-<model>-<task>.md` (date the run started).
- Sections: Goal, Setup, Links, Results, Observations, Issues found, Next.
- Setup names the git commit the run used, the exact launch command or its
  arguments, and the data file; say what changed compared with the previous
  experiment.
- Report chrF++ per direction (BLEU is noisy on a few hundred short segments),
  and compare a direction across experiments, not directions with each other.
- Add a row to the table below.

## Comparison

Eval loss is on `validation_v1.jsonl` (1,200 rows: 100 per direction from
FLORES+ `dev` for 10 directions, plus moore-web `validation` fra→mos and
mos→fra). chrF++ is on FLORES+ `devtest` (1,012 sentences per direction),
averaged over the 10 directions; per-direction scores are in each file.

| Date | Experiment | Model | Data | Epochs / LR | Best eval loss | FLORES+ devtest chrF++ (avg of 10) |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-27 | [HY-MT1.5-1.8B, mos/dyu/fra/eng bidirectional](2026-10-27-hy-mt15-1_8b-mos-dyu-bidirectional.md) | tencent/HY-MT1.5-1.8B + LoRA r64 | `mixed_v1.jsonl` (200k, 5 pairs) | 2 (early stopping) / 2e-4 | 3.082 (step 1000, running) | not scored |
