# Training (`train/train_dense.py`, `train/modal_single_gpu.py`)

`train_dense.py` fine-tunes dense Hy-MT models (full or LoRA) on ShareGPT
JSONL, masking the loss to the assistant reply. `modal_single_gpu.py` runs it
on one Modal GPU with models, data and outputs in Modal Volumes.

## 2026-11-25 (save/eval interval flags for short runs)

- Launcher flags `--save-steps`, `--eval-steps`, `--save-total-limit`
  (defaults 200, 200, 2, unchanged for SFT). They were fixed at 200, while
  the first CPT run is ~105 steps (1.72M tokens, 2048-token blocks, 16 blocks
  per step, 2 epochs; settings in the CPT plan), so it would never have
  evaluated or saved mid-run.
- With an eval file, save steps must be a multiple of eval steps:
  `--load_best_model_at_end` can only choose among evaluated checkpoints. The
  launcher now refuses other values before starting a GPU.
- Checked: `modal run train/modal_single_gpu.py --help` lists the flags.
- **5-step CPT trial** (A100, HY-MT1.5-1.8B LoRA r64, `cpt_mos_v1.jsonl`,
  `--data-format text`, 2048 × 8 × 2, save/eval every 5): no OOM with
  gradient checkpointing (peak memory not logged); 840 train / 28 eval
  blocks; ~4.5 s per step, 0.019 epoch per step (52 steps per epoch), so the
  2-epoch run is ~8 min of training. Loss 3.66 → 3.24 over warmup, eval loss
  3.32 at step 5. Eval, save (`checkpoint-5`: adapter, optimizer, scheduler)
  and `load_best_model_at_end` all ran. Output
  `hy-mt1_5-1_8b-mos-cpt-trial` on `hy-mt-training-output` is a test only.

## 2026-11-25 (faster launcher defaults)

- **Speed test**: HY-MT1.5-1.8B LoRA r64, `fra_mos_mix_train_v1.jsonl`,
  64 rows per optimizer step, max length 512, A100 40GB:

  | Settings | 50 steps |
  |---|---|
  | old (eager/SDPA, random batches, 4 x 16, checkpointing) | 398 s |
  | flash + grouped, 16 x 4, checkpointing | 120 s |
  | flash + grouped, 32 x 2, checkpointing | 82 s (4.8x) |
  | flash + grouped, 16 x 4, no checkpointing | OOM on the first batch |

  40-step runs of flash + grouping alone at 4 x 16 took the same time as
  the old settings (332 s vs 330 s) while computing ~35% fewer padded
  tokens: micro-batches of 4 short rows (mean 107 tokens) leave the GPU
  idle, so step time barely tracks tokens. The batch size is the lever;
  grouping keeps its memory bounded.
- **Padding**: random batches of 4 compute 1.78x the real tokens of the
  mix; length grouping 1.00x. Computing the lengths tokenizes every row
  once at startup (~35 s locally for 50k rows).
- **FlashAttention-2** comes from `transformers[kernels]` (kernels 0.18.3
  resolved): without `flash_attn` installed, `flash_attention_2` falls back
  to `kernels-community/flash-attn2` from the Hub. No compile.
- **Grouping also applies to eval batches**, and eval loss is a mean of
  per-batch means, so eval losses of grouped runs are not exactly
  comparable to the old run's (1.524 at step 800): rerun the baseline.
- `group_by_length` is skipped in text mode: packed blocks all have the
  same length.

## 2026-11-24 (merge on Modal; merged models as `--model-id`)

- **`train/modal_merge.py`** runs `merge_lora_weight.py` on a CPU container:
  adapter from `hy-mt-training-output`, merged model to `hy-mt-models/<name>`.
  A separate app, because a second function in `modal_single_gpu.py` would
  make `modal run train/modal_single_gpu.py --model-id ...` ambiguous. It
  refuses to overwrite, and removes a half-written output on failure.
- **Launcher**: a `--model-id` without `/` is a folder of ours in
  `hy-mt-models`: a missing one raises `FileNotFoundError` instead of a Hub
  download, a name with other characters than letters, digits, `.`, `_`, `-`
  raises `ValueError`. The check runs in the GPU container, so it waits for a
  GPU first (an L4 test waited >10 min for capacity; A100 was free).
- Checked on Modal: merged `fra-mos-mix-lr2e-4/checkpoint-800` (bf16
  safetensors plus tokenizer/config files), re-merge refused, and 2
  training steps from the merged model on an A100 (fresh LoRA r64, no
  download). Test folders deleted afterwards.

## 2026-11-24 (merge defaults: bf16, safetensors)

- `merge_lora_weight.py` now saves `model.safetensors` (was `.bin`) in bf16 by
  default (was fp32: ~7.2 GB instead of ~3.6 GB for 1.8B). It only reads the
  adapter, so the adapter checkpoint is kept; tokenizer and config files are
  copied from the base model folder.
- Checked on CPU with a 2-layer random HunYuanDenseV1: with an adapter that
  moves logits by up to 0.77, merged vs base + adapter differ by at most 0.006
  (bf16 rounding).

## 2026-11-24 (plain-text mode for continued pretraining)

- **`--data_format text`** (launcher `--data-format text`): `PackedTextDataset`
  reads `{"text": ...}` rows, wraps each document as
  `<｜hy_begin▁of▁sentence｜> text <｜hy_place▁holder▁no▁2｜>` (the bos/eos
  the chat template uses; the tokenizer adds no bos by itself), concatenates
  them in file order and cuts `max_seq_length` blocks; the last partial block
  is dropped, empty documents are skipped. Labels = inputs, no attention mask
  between documents in a block (usual packed pretraining).
- The pad token (`<｜hy_▁pad▁｜>`, 120002) differs from eos (120020), so the
  collator's `input_ids != pad` attention mask keeps document ends visible.
- Checked on CPU with a 2-layer random HunYuanDenseV1 and the real tokenizer:
  text mode trains, evaluates and saves; chat mode unchanged. Chat `eval_loss`
  is `nan` when `max_seq_length` cuts every reply off (64 tokens in the test),
  fine at 512.
- `train_dense.py` run by hand needs `--lr_scheduler_type cosine_with_min_lr`
  (the launcher passes it): it always sets `min_lr_rate`, which other
  schedulers reject.

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
