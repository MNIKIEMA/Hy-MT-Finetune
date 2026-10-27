# Training (`train/train_dense.py`, `train/modal_single_gpu.py`)

`train_dense.py` fine-tunes dense Hy-MT models (full or LoRA) on ShareGPT
JSONL, masking the loss to the assistant reply. `modal_single_gpu.py` runs it
on one Modal GPU with models, data and outputs in Modal Volumes.

## 2026-10-27 (Modal launcher on A100)

- **The Modal resources are in the `nikiemamahamadi01` workspace**, not in
  the active default profile (`burkimbia-project`). Run Modal commands with
  `MODAL_PROFILE=nikiemamahamadi01`. Volumes: `hy-mt-models` (holds
  `tencent--HY-MT1.5-1.8B`), `hy-mt-data`, `hy-mt-training-output`. Secret:
  `huggingface-secret`.
- **GPU is chosen at launch** with `MODAL_GPU=A100` (read at import time;
  default `L4`). The data Volume was renamed `hy-mt-training-data` ->
  `hy-mt-data`.
- **`--eval-data-file`** evaluates every 200 steps (same as `save_steps`),
  keeps the best checkpoint by `eval_loss` and enables early stopping
  (`early_stopping_patience` 5 evals in `train_dense.py`).
- **`--load_best_model_at_end` failed at argument parsing** ("requires the
  save and eval strategy to match"): Transformers validates it before
  `train_dense.py` switches `eval_strategy` to `steps`. The launcher now passes
  `--eval_strategy steps` explicitly.
- **Loss mask checked for HY-MT1.5-1.8B with the v1 template**: the label span
  is `<｜hy_Assistant｜>…<｜hy_place▁holder▁no▁2｜>`, 0 of 3,000 sampled rows
  had no labels. Token lengths of the 200k mix: p50 76, p95 257, p99 386,
  max 630; 3 of 3,000 over 512.
- `--use_qk_norm` is passed but never read by `train_dense.py`; harmless.
- The same `model_size` special tokens work for Hy-MT 1.5 and Hy-MT2 dense
  models (`model_version.md`). A wrong `--model-size` silently trains on
  nothing (no label span found).
