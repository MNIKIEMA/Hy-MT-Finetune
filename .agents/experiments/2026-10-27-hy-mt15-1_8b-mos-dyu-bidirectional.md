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

Stopped early at step 3,400 (epoch 1.09) of 6,250: five evals without beating
step 2,400. `train_runtime` 10,740 s (2 h 59 min), mean train loss 1.241.
Saved: `checkpoint-2400` (best) and `checkpoint-3400` (last). Train loss is
the mean over the previous 200 steps.

| Step | Epoch | Train loss | Eval loss |
| ---: | ---: | ---: | ---: |
| 200 | 0.06 | 2.73 | 3.408 |
| 400 | 0.13 | 1.70 | 3.223 |
| 600 | 0.19 | 1.49 | 3.198 |
| 800 | 0.26 | 1.38 | 3.155 |
| 1,000 | 0.32 | 1.30 | 3.082 |
| 1,200 | 0.38 | 1.23 | 3.106 |
| 1,400 | 0.45 | 1.21 | 3.131 |
| 1,600 | 0.51 | 1.17 | 3.082 |
| 1,800 | 0.58 | 1.12 | 3.104 |
| 2,000 | 0.64 | 1.10 | 3.066 |
| 2,200 | 0.70 | 1.06 | 3.117 |
| 2,400 | 0.77 | 1.03 | **3.029** |
| 2,600 | 0.83 | 1.01 | 3.075 |
| 2,800 | 0.90 | 0.99 | 3.034 |
| 3,000 | 0.96 | 0.96 | 3.065 |
| 3,200 | 1.02 | 0.88 | 3.117 |
| 3,400 | 1.09 | 0.74 | 3.146 |

### Benchmarks

`checkpoint-2400` (served as LoRA `mos-dyu`) against the base model (`hy-mt`),
vLLM 0.31.0 `--enforce-eager` on one A100, greedy (`temperature 0`,
`repetition_penalty 1.05`), `max_tokens 256`, v1 prompt. Scored 2026-10-28
with `scripts/evaluate_translation.py`; predictions and `summary.md` in
`eval/results/run1-checkpoint-2400/` (local, not committed). chrF++ (BLEU),
length ratio of output to reference.

**FLORES+ `devtest`** (1,012 rows per direction):

| Direction | Base | checkpoint-2400 | Δ chrF++ |
| --- | --- | --- | ---: |
| mos→fra | 18.06 (2.36), 0.91 | 25.46 (5.10), 1.01 | +7.40 |
| fra→mos | 14.08 (0.90), 1.34 | 20.89 (2.28), 1.06 | +6.81 |
| mos→eng | 19.72 (2.12), 1.11 | 22.12 (3.92), 0.95 | +2.40 |
| eng→mos | 14.45 (1.35), 1.18 | 17.69 (1.44), 1.18 | +3.24 |
| dyu→fra | 18.18 (2.00), 0.96 | 21.48 (1.95), 1.12 | +3.30 |
| fra→dyu | 14.01 (0.63), 1.27 | 14.69 (0.75), 1.06 | +0.68 |
| dyu→eng | 19.82 (1.85), 1.16 | 19.61 (2.17), 0.98 | −0.21 |
| eng→dyu | 14.19 (1.04), 1.12 | 13.01 (0.49), 0.96 | −1.18 |
| dyu→mos | 13.97 (0.50), 1.11 | 15.12 (0.52), 1.50 | +1.15 |
| mos→dyu | 13.12 (0.58), 0.99 | 13.72 (0.36), 1.22 | +0.60 |
| **average** | **15.96** | **18.38** | **+2.42** |

**moore-web `test`** (2,573 rows per direction):

| Direction | Base | checkpoint-2400 | Δ chrF++ |
| --- | --- | --- | ---: |
| fra→mos | 13.04 (1.86), 1.39 | 30.32 (8.93), 1.12 | +17.28 |
| mos→fra | 15.55 (2.32), 0.89 | 34.67 (17.41), 0.91 | +19.12 |

**Bouquet `test`** (`facebook/bouquet`, references as published; sentences
256 new tokens, paragraphs 384). Predictions in
`eval/results/run1-checkpoint-2400-bouquet/`.

| Direction | Level | Rows | Base | checkpoint-2400 | Δ chrF++ |
| --- | --- | ---: | --- | --- | ---: |
| fra→mos | sentence | 854 | 10.42 (1.98), 1.28 | 25.57 (7.08), 1.06 | +15.15 |
| mos→fra | sentence | 854 | 14.04 (2.29), 1.03 | 27.45 (8.34), 0.99 | +13.41 |
| eng→mos | sentence | 854 | 11.10 (2.35), 1.14 | 21.94 (4.71), 1.25 | +10.84 |
| mos→eng | sentence | 854 | 15.81 (2.47), 1.21 | 25.12 (8.43), 0.88 | +9.31 |
| fra→mos | paragraph | 198 | 12.60 (2.78), 1.23 | 26.32 (7.96), 1.01 | +13.72 |
| mos→fra | paragraph | 198 | 16.45 (4.01), 0.84 | 31.27 (10.32), 0.92 | +14.82 |
| eng→mos | paragraph | 198 | 13.36 (3.04), 1.11 | 18.78 (5.06), 0.74 | +5.42 |
| mos→eng | paragraph | 198 | 19.34 (4.16), 0.95 | 26.87 (8.50), 0.73 | +7.53 |

Gains are even across Bouquet domains (sentence fra→mos: 22.1 how-to to
27.5 narration/reflection; base 9.5-12.0). Looping outputs: eng→mos 44 of
854 sentences and 55 of 198 paragraphs; fra→mos 12 and 30; out of Mooré 1-3.

For reference, the NLLB-600M fr→mos models in `mt-training` score 30.87
(FrMos-bt-v1) and 30.82 (BIA-CRCL) chrF++ on Bouquet fra→mos sentences, 35.44
and 34.40 on paragraphs, but with CT2 int8, beam 4, `no_repeat_ngram_size=3`
and normalized references (’ « » → ' "), so not directly comparable.

**Repetition** (checkpoint-2400; "looping" = output of 8+ words with under
half of them distinct): dyu→mos 197 of 1,012 looping and 31.5% of outputs
over 1.5× the reference length; mos→dyu 117 (18.4%); eng→mos 99 (13.7%);
eng→dyu 84; fra→dyu 82; moore-web fra→mos 55 of 2,573 (19.2% over 1.5×).
Directions out of mos/dyu into fra/eng loop least (12-30).

**Repetition penalty 1.15** (checkpoint-2400 only, everything else as
above; `eval/results/run1-checkpoint-2400-rp1.15[-bouquet]/`). chrF++ at
1.05 → 1.15, looping outputs 1.05 → 1.15.

| Set | Direction | chrF++ | Δ | Looping |
| --- | --- | --- | ---: | --- |
| FLORES+ | dyu→mos | 15.12 → 16.57 | +1.45 | 197 → 14 |
| FLORES+ | mos→dyu | 13.72 → 15.07 | +1.35 | 117 → 2 |
| FLORES+ | fra→dyu | 14.69 → 15.66 | +0.97 | 82 → 1 |
| FLORES+ | eng→dyu | 13.01 → 13.94 | +0.93 | 84 → 0 |
| FLORES+ | eng→mos | 17.69 → 17.86 | +0.17 | 99 → 1 |
| FLORES+ | fra→mos | 20.89 → 20.43 | −0.46 | 36 → 1 |
| FLORES+ | into fra/eng (4) | 22.17 → 21.85 avg | −0.32 | 68 → 3 |
| FLORES+ | **average** | **18.38 → 18.69** | **+0.31** | |
| moore-web | fra→mos | 30.32 → 28.16 | −2.16 | 55 → 6 |
| moore-web | mos→fra | 34.67 → 33.33 | −1.34 | 16 → 1 |
| Bouquet sentence | average (4) | 25.02 → 24.14 | −0.88 | 58 → 3 |
| Bouquet paragraph | average (4) | 25.81 → 24.53 | −1.28 | 90 → 8 |

BLEU drops almost everywhere (e.g. moore-web mos→fra 17.41 → 14.86).

## Observations

- **Eval loss is flat from step ~1,000 to the end of epoch 1** (3.03-3.13,
  noisy, best 3.029 at step 2,400) while train loss falls from 1.30 to 0.96.
  Grad norm stayed stable (~1.3, max < 1.7 after warmup), so lr 2e-4 was not
  unstable.
- **Epoch 2 is memorization**: train loss drops sharply once rows repeat
  (0.96 at step 3,000 → 0.74 at 3,400) and eval loss rises (3.146). A second
  epoch on this mix does not help.
- The validation set is ~83% FLORES (general domain) and the mix is mostly
  religious and government text, so part of the train/eval gap is domain
  mismatch. Eval loss is not translation quality: the decision needs chrF++.
- Early stopping ended the run at step 3,400 and kept checkpoint-2400
  (0.77 epoch).

- **Mooré learned, Dyula barely**: large gains in-domain (moore-web +17-19
  chrF++) and on FLORES+ out of and into Mooré (+3 to +7); into-Dyula is flat
  or worse (fra→dyu +0.7, eng→dyu −1.2, mos→dyu +0.6). Part of this is
  FLORES+ Dyula's French-style spelling ([evaluation logbook](../logbooks/evaluation.md));
  the rest needs a Dyula test set in standard orthography to tell apart.
- **Repetition loops are the main failure**, worst when generating between
  the two local languages (dyu→mos 19% of rows loop). dyu-mos training rows
  were repeated 2.6-3.3x and come from long Council sentences.

- **Bouquet confirms the Mooré gains out of domain** (+9 to +15 chrF++ on
  sentences, every domain), but the model is ~5 chrF++ behind dedicated
  NLLB-600M fr→mos models on sentences and ~9 on paragraphs (different
  decoding and reference normalization, see above).
- **Paragraphs into and out of English come out too short** (eng→mos length
  ratio 0.74, mos→eng 0.73) and eng→mos paragraphs loop (55 of 198): training
  rows are single sentences, so the model drops or repeats content in long
  inputs.

- **Repetition penalty 1.15 removes the loops but is not a better default.**
  It helps only where the model loops and is weak (FLORES+ into Dyula and
  between mos and dyu, +0.9 to +1.5 chrF++), and costs 1-2 chrF++ and more
  BLEU where it already translates well (moore-web, Bouquet, into
  fra/eng): the penalty also punishes the legitimate repeats of articles
  and function words. Outputs into fra/eng get longer (more over 1.5× the
  reference). So most of the dyu→mos / mos→dyu gap is the model, not
  decoding; fix loops in training (less repeated dyu-mos rows) or with
  `no_repeat_ngram_size`-style constraints, and keep 1.05 for reported
  scores.

## Issues found

- First launch failed on `--load_best_model_at_end` before training
  ([training logbook](../logbooks/training.md)).

## Next

- ~~Re-score with `repetition_penalty` 1.15~~ done: loops gone, scores mixed
  (see above).
- Build a Dyula test set in standard orthography (e.g. held-out faso-web
  sentences) so into-dyu progress is measurable.
- If the checkpoints are close: try lr 1e-4, and a mix with religious text
  capped at ~20%.
- Default to 1 epoch until a run shows eval loss still falling after epoch 1.
