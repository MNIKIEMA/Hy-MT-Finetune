# HY-MT1.5-1.8B fine-tuned for mos/dyu/fra/eng in both directions (2026-10-27)

## Goal

First run of the bidirectional recipe: fine-tune `tencent/HY-MT1.5-1.8B` with
LoRA on 10 directions between Mooré, Dyula, French and English (fra↔eng not
trained), and see how far 2 epochs help.

## Setup

- **Code**: branch `bidirectional-data-mix`, mixer at `8c9aaa4`; the launcher
  changes (`MODAL_GPU`, `--eval-data-file`, `--eval_strategy steps`) were
  uncommitted at launch and committed as `2ac7108` right after.
- **Data**: `hy-mt-data/mixed_v1.jsonl`, 200,000 rows, built with
  `scripts/mix_translation_data.py mix --instruction-version v1
  --total-examples 200000` (seed 42, default pair shares: mos-fra 25%, mos-eng
  20%, dyu-fra 22%, dyu-eng 16%, dyu-mos 17%). Sources, filters and
  decontamination: [data-mixing logbook](../logbooks/data-mixing.md).

  | Direction | Rows | Direction | Rows |
  | --- | ---: | --- | ---: |
  | fra→mos | 29,940 | mos→fra | 20,060 |
  | eng→mos | 23,923 | mos→eng | 16,077 |
  | fra→dyu | 26,645 | dyu→fra | 17,355 |
  | eng→dyu | 19,305 | dyu→eng | 12,695 |
  | dyu→mos | 16,985 | mos→dyu | 17,015 |

  Domain: NLLB rows 54% (≥35% of the mix from jw.org / bibles.org / bible.is /
  beblia.com), Council of Ministers reports ~29%, dictionary ~8.5%,
  human-checked 7.5%.
- **Validation**: `hy-mt-data/validation_v1.jsonl`, 1,200 rows (100 per
  direction; FLORES+ `dev` ×10, moore-web `validation` ×2).
- **Model**: LoRA r=64, alpha=128, dropout 0.05 on `q/k/v/o_proj` and
  `gate/up/down_proj` (77.6M trainable, 4.15%); v1 chat template (instruction
  in the system message), loss on the assistant reply only.
- **Training**: 2 epochs (6,250 steps), lr 2e-4, 100 warmup steps, cosine to
  1e-6, batch 16 × 4 accumulation = 64, sequence length 512, bf16, gradient
  checkpointing. Eval and checkpoint every 200 steps, best model by eval loss,
  early stopping patience 5 evals.
- **Hardware**: Modal, one A100 (40 GB), ~2.9–3.0 s/step training.

Launch:

```bash
MODAL_PROFILE=nikiemamahamadi01 MODAL_GPU=A100 uv run --extra modal modal run --detach train/modal_single_gpu.py \
  --model-id tencent/HY-MT1.5-1.8B --model-size 1.8B \
  --train-data-file mixed_v1.jsonl --eval-data-file validation_v1.jsonl \
  --output-name hy-mt1_5-1_8b-mos-dyu-lora \
  --num-train-epochs 2 --per-device-train-batch-size 16 \
  --gradient-accumulation-steps 4 --warmup-steps 100
```

## Links

- Modal training app: `ap-7aebeLQbwAhymcXA5UsMKh` (workspace
  `nikiemamahamadi01`); failed first launch: `ap-Sbmk9WmZEqG41mIuAFgtZ0`.
- Output: volume `hy-mt-training-output`, `hy-mt1_5-1_8b-mos-dyu-lora/`.
- No W&B (`--report_to none`).

## Results

### Eval loss (validation, 1,200 rows)

Train loss is the mean over the previous 200 steps.

| Step | 200 | 400 | 600 | 800 | 1,000 | 1,200 | 1,400 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Epoch | 0.06 | 0.13 | 0.19 | 0.26 | 0.32 | 0.38 | 0.45 |
| Train loss | 2.73 | 1.70 | 1.49 | 1.38 | 1.30 | 1.23 | 1.21 |
| Eval loss | 3.408 | 3.223 | 3.198 | 3.155 | **3.082** | 3.106 | 3.131 |

Still running at the time of writing (step ~1,660, train loss 1.17 over
steps 1,401–1,600).

### Benchmarks

Not scored yet.

## Observations

- **Eval loss bottomed at step 1,000 (0.32 epoch) and is rising** while train
  loss keeps falling. Grad norm is stable (~1.3, max < 1.7 after warmup), so
  lr 2e-4 is not unstable; the model is moving away from the validation
  distribution.
- The validation set is ~83% FLORES (general domain) and the mix is mostly
  religious and government text, so part of the train/eval gap is domain
  mismatch. Eval loss is not translation quality: the decision needs chrF++.
- With patience 5, early stopping should end the run around step 2,000 and
  keep checkpoint-1000, far short of the planned 2 epochs.

## Issues found

- First launch failed on `--load_best_model_at_end` before training
  ([training logbook](../logbooks/training.md)).

## Next

- Score checkpoint-1000 and the last checkpoint with chrF++ per direction on
  FLORES+ `devtest` and moore-web `test`; add the averages to the README table.
- If the checkpoints are close: try lr 1e-4, and a mix with religious text
  capped at ~20%.
- Treat 1 epoch (or fewer steps) as the default until a run shows eval loss
  still falling after epoch 1.
