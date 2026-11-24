# HY-MT1.5-1.8B on the TranslateGemma French↔Mooré mix (2026-11-24)

## Goal

Compare HY-MT1.5-1.8B with TranslateGemma 4B on the same data: the 50k
French↔Mooré rows TranslateGemma trains on, with no eng/dyu data. Plan and
TranslateGemma side:
`translategemma-finetune/.agents/logbooks/2026-10-29-plan-fair-hy-mt-comparison.md`.

## Setup

- **Code**: branch `fra-mos-mix-comparison` at `98c55bc`. Changed from the
  previous experiment: data and validation file only.
- **Data**: `fra_mos_mix_train_v1.jsonl`, 50,000 rows (fra→mos 29,940,
  mos→fra 20,060), from `scripts/mix_translation_data.py convert` on
  `madoss/translategemma-fra-mos-mix` revision `ca1409b`
  ([data-mixing logbook](../logbooks/data-mixing.md)).
- **Validation**: `fra_mos_mix_validation_v1.jsonl`, 1,000 moore-web
  `validation` rows (500 per direction), the set TranslateGemma selects on.
- **Model and training**: as in
  [2026-10-27](2026-10-27-hy-mt15-1_8b-mos-dyu-bidirectional.md): LoRA r64,
  alpha 128, dropout 0.05, loss on the reply only, lr 2e-4, 100 warmup steps,
  cosine to 1e-6, batch 16 × 4 = 64, length 512, bf16, 2 epochs (1,564 steps),
  eval every 200 steps, best by eval loss. One A100 (40 GB), ~3.0 s/step.

```bash
MODAL_PROFILE=nikiemamahamadi01 MODAL_GPU=A100 uv run --extra modal modal run --detach train/modal_single_gpu.py \
  --model-id tencent/HY-MT1.5-1.8B --model-size 1.8B \
  --train-data-file fra_mos_mix_train_v1.jsonl --eval-data-file fra_mos_mix_validation_v1.jsonl \
  --output-name hy-mt1_5-1_8b-fra-mos-mix-lr2e-4 \
  --num-train-epochs 2 --per-device-train-batch-size 16 \
  --gradient-accumulation-steps 4 --warmup-steps 100 --learning-rate 2e-4
```

## Links

- Modal training app `ap-exsxHgJBeyXWHHuYZ3v5YB`; output
  `hy-mt-training-output/hy-mt1_5-1_8b-fra-mos-mix-lr2e-4/` (`checkpoint-800`
  best, `checkpoint-1564` last).
- Served by `hy-mt-vllm` as `fra-mos-mix`. Predictions:
  `eval/results/fra-mos-mix-ckpt800-rp{1.0,1.05}[-bouquet]/` (local).

## Results

Eval loss (1,000 moore-web validation rows): 1.791 (step 200), 1.606 (400),
1.528 (600), **1.524 (800, epoch 1.02)**, 1.542 (1000), 1.534 (1200), 1.534
(1400), 1.532 (1564). Not comparable with TranslateGemma's eval loss
(different tokenizer).

Test scores, `checkpoint-800`, greedy, `max_tokens 256`, v1 prompt. chrF++
(BLEU); looping = outputs of 8+ words with under 40% unique words. Both
models rescored with the same sacrebleu code and identical references.
TranslateGemma = bf16, translation-only loss, best checkpoint (step 2400).

**Bouquet `test`, sentence level** (854 per direction):

| Model | Penalty | fra→mos | mos→fra | fra→mos looping |
| --- | ---: | --- | --- | ---: |
| TranslateGemma 4B | 1.0 | **24.92** (7.76) | 24.34 (6.82) | 15 |
| HY-MT 1.8B | 1.0 | 23.76 (5.97) | **26.97** (8.92) | 44 |
| TranslateGemma 4B | 1.05 | 24.93 (7.70) | 24.83 (6.59) | 3 |
| HY-MT 1.8B | 1.05 | 24.94 (6.89) | **27.44** (9.00) | 9 |
| HY-MT 1.8B, 200k mix | 1.05 | 25.57 (7.08) | 27.45 (8.34) | 12 |

**moore-web `test`** (2,573 per direction):

| Model | Penalty | fra→mos | mos→fra | fra→mos looping |
| --- | ---: | --- | --- | ---: |
| TranslateGemma 4B | 1.0 | **30.63** (8.94) | 31.34 (15.78) | 74 |
| HY-MT 1.8B | 1.0 | 29.10 (7.43) | **34.54** (17.76) | 150 |
| TranslateGemma 4B | 1.05 | **31.02** (9.27) | 31.75 (15.63) | 15 |
| HY-MT 1.8B | 1.05 | 30.24 (8.25) | **35.11** (17.45) | 40 |
| HY-MT 1.8B, 200k mix | 1.05 | 30.32 (8.93) | 34.67 (17.41) | 55 |

## Observations

- **Into French, HY-MT is clearly better**: +2.6 chrF++ on Bouquet and +3.4
  on moore-web at penalty 1.05, with ~2 BLEU more.
- **Into Mooré the two are close**: Bouquet tied at 1.05 (24.94 vs 24.93,
  TranslateGemma +0.8 BLEU), TranslateGemma +0.8 on moore-web; TranslateGemma
  loops less (Bouquet 3 vs 9, moore-web 15 vs 40) and its lead grows without
  a penalty (+1.2, +1.5). HY-MT outputs are long (moore-web fra→mos length
  ratio 1.17 at 1.05, 1.27 at 1.0).
- **The eng/dyu data of the 200k mix barely matters for fra↔mos**: within
  ±0.6 chrF++ of the 50k-only run (Bouquet fra→mos −0.6, others ±0.4).
- Eval loss stops improving at the end of epoch 1, as on the 200k mix.

## Next

- Into Mooré is within a learning-rate sweep's reach: HY-MT at 1e-4 and
  TranslateGemma at 2e-4 would settle it. Into French needs no sweep.
