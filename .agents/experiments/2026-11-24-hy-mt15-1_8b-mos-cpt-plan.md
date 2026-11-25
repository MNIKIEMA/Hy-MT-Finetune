# Plan — continued pretraining (CPT) on Mooré text for HY-MT1.5-1.8B (rung 3: no gain beyond SFT run-to-run noise)

## Goal

Find out whether continued pretraining on monolingual Mooré text improves
French↔Mooré translation, and which data and settings help most, by hill
climbing on HY-MT1.5-1.8B.

Why HY-MT and not TranslateGemma 4B: on the same 50k French↔Mooré rows, HY-MT
is better into French and tied into Mooré
([2026-11-24-hy-mt15-1_8b-fra-mos-mix.md](2026-11-24-hy-mt15-1_8b-fra-mos-mix.md)),
at about half the compute per Mooré word (1.8B vs 4.3B parameters; 2.73 vs
2.45 tokens per Mooré word on the 1,000 moore-web validation rows).

What CPT should fix: HY-MT's into-Mooré output loops (Bouquet test 9 of 854,
moore-web 40 of 2,573 at penalty 1.05, vs 3 and 15 for TranslateGemma) and runs
long (moore-web fra→mos length ratio 1.17).

## Protocol (fixed for every candidate)

1. **CPT** on a Mooré text mix (the variable).
2. **SFT**: the same French↔Mooré stage for every candidate,
   `fra_mos_mix_train_v1.jsonl`, lr 2e-4, **1 epoch** (the 2-epoch run picked
   its checkpoint at epoch 1.02, so the second epoch is wasted compute).
3. **Select and compare on dev sets only**: Bouquet sentence-level `dev` and
   moore-web `validation`, chrF++ both directions, penalty 1.05, greedy, 256
   max tokens, plus looping and length ratio. Bouquet `test` is the
   leaderboard set: report on it once at the end, never select on it.

Baseline = step 2 alone (no CPT), rerun with the same 1-epoch SFT so every
candidate differs from it only by the CPT stage.

## Code to add first

- ~~**Plain-text rows in `train/train_dense.py`**~~ Done 2026-11-24:
  `--data_format text` (launcher `--data-format text`), documents packed into
  `max_seq_length` blocks with bos/eos, loss on every token
  ([training logbook](../logbooks/training.md)).
- ~~**Launcher**: start SFT from a CPT output~~ Done 2026-11-24: merge the
  CPT adapter with `train/modal_merge.py`, then `--model-id <merged-name>`
  trains a fresh LoRA on it (option chosen: merge, not continuing the CPT
  adapter, so SFT is identical to the baseline's).
- **Faster runs** (done, now the launcher defaults; rerun the baseline with
  them before comparing): FlashAttention-2 via `kernels`, length-grouped SFT
  batches, micro-batch 32 x 2 accumulation. 4.8x faster on an A100 40GB
  (training logbook, 2026-11-25).
- ~~**Held-out check**~~ Done 2026-11-25 in `scripts/build_cpt_data.py`:
  sentence-level removal of moore-web validation/test, FLORES+ dev/devtest and
  Bouquet dev/test (sentence and paragraph level) Mooré, with the mixer's
  `normalize_text`; the build fails if one of 4+ words and 20+ characters is
  still inside a document.
- ~~**Save/eval interval**~~ Done 2026-11-25: launcher flags `--save-steps`,
  `--eval-steps`, `--save-total-limit` (defaults 200, 200, 2 as before). Were
  fixed at 200, more than a whole CPT run.

## CPT settings (first run, rung 3)

`cpt_mos_v1.jsonl` is 1,721,957 HY-MT tokens (2.92 per word, counted, not
the 2.73 estimate); validation 58,713. Median document 791 tokens, p90
10,482 (Council sessions), max 37,677.

| Setting | SFT default | CPT |
|---|---|---|
| `--max-seq-length` | 512 | 2048 (most articles and tales fit in one block) |
| `--per-device-train-batch-size` × `--gradient-accumulation-steps` | 32 × 2 | 8 × 2 |
| Tokens per micro-batch / per step | 16,384 / 32,768 | 16,384 / 32,768 |
| Steps | — | 840 blocks → 52 per epoch |
| `--num-train-epochs` | 1 | 2 (~105 steps); epochs are a knob |
| `--warmup-steps` | 10 | 10 (~10%) |
| `--learning-rate` | 2e-4 | 2e-4 (knob) |
| `--save-steps` / `--eval-steps` / `--save-total-limit` | 200 / 200 / 2 | 26 / 26 / 5 |

- Same tokens per micro-batch as SFT, which fits an A100 with gradient
  checkpointing (the training logbook has an OOM without it). The default
  32 × 2 at 2048 would be 131k tokens per step, ~13 steps per epoch, mostly
  warmup.
- Saving every 26 steps and keeping all 5 checkpoints keeps ~epoch 1
  (step 52) and ~epoch 2 (step 104): each can go through SFT, since we
  select after SFT, not on CPT loss.
- Check memory and speed with `--max-steps 5` first.

```bash
MODAL_GPU=A100 uv run --extra modal modal run train/modal_single_gpu.py \
  --model-id tencent/HY-MT1.5-1.8B --model-size 1.8B --data-format text \
  --train-data-file cpt_mos_v1.jsonl --eval-data-file cpt_mos_validation_v1.jsonl \
  --max-seq-length 2048 --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
  --num-train-epochs 2 --save-steps 26 --eval-steps 26 --save-total-limit 5 \
  --output-name hy-mt1_5-1_8b-mos-cpt-v1
```

## Data candidates

Measured 2026-11-25 (our own document sources; the Hub sources below them are
still unmeasured). Words are whitespace words of the Mooré side; tokens are
estimated at 2.73 HY-MT tokens per Mooré word (measured on moore-web
validation), not tokenized. "Held-out hits" = documents containing at least one
moore-web validation/test, FLORES+ dev/devtest or Bouquet dev/test Mooré
sentence of 4+ words, after the mixer's `normalize_text`, as a substring.

| Source | Where | Mooré docs | Mooré words | ~Tokens | Held-out hits | Notes |
|---|---|---:|---:|---:|---|---|
| Council of Ministers reports | `madoss/conseil-ministres-parsed` `corpus.json` (`mos` sections, sentence lists) | 93 sessions (2024-02 to 2026-09-10) | 421,550 | 1.15M | **93 of 93 sessions**: 644 moore-web validation + 628 test sentences | Government register. Archive (`hf://buckets/madoss/faso-web-docs/conseils-ministres`) has 3 newer sessions (2026-09-24, 10-08, 10-22) not in the release: re-run `conseil-ministres build`. The SFT mix already has the aligned pairs |
| Raamde news | `moore-web/data/raamde/raamde_corpus_with_glotlid.json`, `segments.moore` (moore-web's Mooré run) | 429 articles (425 without job ads) | 127,845 | 350k | 163 of 429 articles: 139 validation + 61 test sentences | News. Mooré is a rewrite of the French, fine for CPT. Use moore-web's split, not per-unit GlotLID labels (data-mixing logbook) |
| Ministry of Education Facebook posts | `hf://buckets/madoss/ministereduc-bf`, `data/YYYY/MM/*/mos.txt` | 27 | 17,624 | 48k | 0 | Since 2026-05, grows ~weekly. All 27 have a French version too |
| faso-web-docs: running text with a text layer | `hf://buckets/madoss/faso-web-docs` | ~15 PDFs/texts | ~100k (rough: several are bilingual, Mooré share estimated) | ~270k | expert 12, kade 69, sida 16, udhr 4, nouvel-an 8 (+2 Bouquet test), mooreburkina 183 (+2 Bouquet test) | mooreburkina tales/proverbs apps 41.6k; *Pag ne a sid viim* 16.6k; Kadé SIDA facilitator 14.7k; Du Mooré literacy series ~30k (bilingual); SIDA book 7.8k (bilingual); expert batch 45k (bilingual); UDHR 2.5k; New Year 1.3k. These are moore-web's human-quality dev/test documents |
| faso-web-docs: scanned Mooré PDFs | `burkinadoc/{contes,agriculture,documentaires}-moore`, `alimentation-vaches-laitieres`, `kelyam` Mooré sides | ~25 PDFs, ~140 pages | ~20 w/page extractable (headers only) | — | — | Need OCR first |
| faso-web-docs: dictionaries | `moore-dictionaries/`, `mooreburkina-priority/pdf/01-03,10-12,14,15,17,18` | — | 265k–345k per dictionary (all three languages) | — | — | Lexicon entries, not running text; `02/11` and `03/12` are duplicates. At most a small separate rung (example sentences only) |
| Mooré side of the training mix | Mooré sentences of `fra_mos_mix_train_v1.jsonl` | | | | | No new text: controls for "more passes over the same Mooré" |
| `madoss/moore-web-mono` | Web Mooré | | | | | Check overlap with moore-web test first |
| `madoss/mos-latn-hplt` | HPLT Mooré documents (1,892 docs, 28.5 MB with html/xml fields) | | | | | Small; `text` column, LID `prob` to filter |
| `burkimbia/gomdeesa-ocr-brut` | Raw OCR | | | | | Needs cleaning; check OCR noise before use |
| `burkimbia/sidbi-ziri-dataset` | — | | | | | Large; skipped for now |

Our own sources total ~0.67M Mooré words (~1.8M tokens), about two thirds
from the Council reports.

**Decontamination has to be sentence-level, not document-level**:
moore-web's validation/test are split by row, so every Council session holds
some of them, and dropping whole documents would drop all Council text.
Remove held-out sentences from each document, as the SFT mix already does
for the same sessions. The rest of a session stays, so the model sees the
context around moore-web test sentences: **select on Bouquet dev** (2 Bouquet
test hits only, none in dev) and treat moore-web validation as a secondary
check for CPT candidates that include Council or Raamde.

## Runs

### 2026-11-25 — rung 3 CPT (`hy-mt1_5-1_8b-mos-cpt-v1`)

Settings as in "CPT settings" above, A100, 106 steps, 541.8 s of training.
Checkpoints on `hy-mt-training-output`: 26, 52, 78, 104, 106 (all kept).

| Step | Epoch | Train loss | Eval loss (`cpt_mos_validation_v1`, 28 blocks) |
|---:|---:|---:|---:|
| 1 | 0.02 | 3.66 | — (3.32 at step 5 in the trial) |
| 26 | 0.50 | 2.18 | 2.233 |
| 52 | 0.99 | 1.98 | 1.805 |
| 78 | 1.48 | 1.51 | 1.671 |
| 104 | 1.97 | — | 1.641 |
| 106 | 2.00 | 1.48 | 1.641 |

- Eval loss still falls in epoch 2 (1.805 → 1.641) but slows (−0.03 over the
  last 26 steps); train loss drops faster than eval in epoch 2 (second pass
  over the same text). No sign of overfitting yet by CPT loss.
- CPT loss is not translation quality: next is merge, 1-epoch SFT and Bouquet
  dev for checkpoint 52 (~epoch 1) and 106 (epoch 2), against the 1-epoch
  SFT baseline (rung 1).

### 2026-11-25 — 1-epoch SFT: baseline (rung 1) and on CPT epochs 1 and 2

Same SFT for all three: `fra_mos_mix_train_v1.jsonl`, 1 epoch (782 steps),
lr 2e-4, 100 warmup, launcher defaults (32 × 2, flash, grouped), A100. CPT
checkpoints merged with `train/modal_merge.py` into
`hy-mt1_5-1_8b-mos-cpt-v1-ep1` (checkpoint-52) and `-ep2` (checkpoint-106)
on `hy-mt-models`.

| Run (`hy-mt-training-output/…`) | Starts from | Eval loss 0.26 / 0.51 / 0.77 / 1.0 epoch |
|---|---|---|
| `hy-mt1_5-1_8b-sft1ep-baseline` | HY-MT1.5-1.8B | 2.439 / 1.842 / 1.615 / **1.569** |
| `hy-mt1_5-1_8b-cpt-v1-ep1-sft` | CPT epoch 1 | 2.288 / 1.804 / 1.606 / **1.567** |
| `hy-mt1_5-1_8b-cpt-v1-ep2-sft` | CPT epoch 2 | 2.263 / 1.793 / 1.609 / **1.570** |

Eval loss on `fra_mos_mix_validation_v1.jsonl` (moore-web validation, 1,000
rows). Best = `checkpoint-782` for all three.

- CPT starts SFT ahead (−0.15/−0.18 at 0.26 epoch) but the gap is gone by
  the end of the epoch (1.567–1.570). By SFT loss CPT v1 changes nothing;
  the comparison is decided by chrF++, looping and length ratio on Bouquet
  dev and moore-web validation (not yet run).
- Train time 1,181 s (baseline) vs 1,329/1,335 s (merged models): ~12%
  slower per step in training and ~9% in eval. Configs, trainable params
  (77.6M) and tied embeddings match; the merged file lacks only the base
  file's redundant `lm_head` copy. Likely different A100 hardware (not
  logged), not established.

### 2026-11-25 — dev results: CPT v1 does not help

`checkpoint-782` of each SFT run, served by `train/modal_vllm_server.py`
(base model = that run's starting model), `scripts/evaluate_translation.py`
v1 prompt, greedy, penalty 1.05, 256 max tokens. chrF++ (length ratio,
looping outputs = 8+ words with under 40% unique). Predictions and
summaries: `eval/results/cpt-v1-dev/{baseline,cpt-ep1,cpt-ep2}/`. No empty
outputs.

| Set | Dir | Rows | Baseline | CPT ep1 + SFT | CPT ep2 + SFT |
|---|---|---:|---|---|---|
| Bouquet sentence dev | fra→mos | 504 | 25.02 (1.007, 2) | 24.67 (1.010, 5) | 24.71 (1.077, 9) |
| | mos→fra | 504 | 26.86 (0.998, 1) | 27.14 (0.994, 1) | 26.78 (0.997, 1) |
| | eng→mos | 504 | 24.33 (0.934, 1) | 24.39 (1.017, 7) | 24.51 (0.987, 4) |
| | mos→eng | 504 | 29.15 (0.989, 0) | 29.38 (0.991, 0) | 29.45 (0.986, 0) |
| Bouquet paragraph dev | fra→mos | 120 | 25.97 (0.848, 7) | 25.81 (0.852, 8) | 25.95 (0.852, 8) |
| | mos→fra | 120 | 32.73 (0.913, 0) | 32.12 (0.926, 2) | 30.62 (0.895, 2) |
| | eng→mos | 120 | 25.88 (0.874, 10) | 25.17 (0.854, 11) | 25.58 (0.806, 7) |
| | mos→eng | 120 | 35.54 (0.962, 0) | 34.94 (0.982, 1) | 34.62 (0.927, 0) |
| moore-web validation | fra→mos | 2,491 | 31.07 (1.066, 31) | 30.88 (1.067, 41) | 30.84 (1.076, 46) |
| | mos→fra | 2,491 | 35.28 (0.948, 4) | 35.03 (0.930, 6) | 35.03 (0.952, 9) |

- **No chrF++ gain**: Bouquet sentence dev within ±0.35 in every direction
  for both CPT runs; moore-web −0.2 both directions (despite CPT having seen
  the context around moore-web validation sentences).
- **The opposite of the goal on looping** (superseded: within SFT
  run-to-run variation, see the null-merge control below): into-Mooré
  looping goes up with CPT, not down: moore-web fra→mos 31 → 41 → 46, Bouquet sentence fra→mos
  2 → 5 → 9, and the moore-web fra→mos length ratio stays ~1.07.
- **Into-French paragraphs drop with more CPT**: Bouquet paragraph mos→fra
  32.73 → 32.12 → 30.62 (120 rows only), mos→eng −0.6/−0.9. Consistent with
  forgetting long-context French/English after 2 epochs of Mooré-only text.
- Single runs, no seeds: differences under ~1 chrF++ are not established.
  The looping increase is consistent across sets and grows with CPT epochs
  (superseded by the null-merge control: a baseline rerun loops 57).
- Next options: replay (rung 7, French/translation rows in CPT), parallel
  rows (rung 3b), lower CPT lr or 1 epoch, or more Mooré (rungs 4–6). Bouquet
  `test` not used.

### 2026-11-25 — CPT lr 5e-5 (`hy-mt1_5-1_8b-mos-cpt-v1-lr5e-5`): more looping, not less

Same as the first CPT run except `--learning-rate 5e-5` (552 s). CPT eval
loss 2.649 / 2.183 / 2.020 / 1.985 / 1.984 at steps 26 / 52 / 78 / 104 /
106 (2e-4 run: 1.641 at the end). Checkpoints 52 and 106 merged
(`hy-mt1_5-1_8b-mos-cpt-v1-lr5e-5-ep{1,2}`), same 1-epoch SFT (final eval
loss 1.574 / 1.573), scored with `train/modal_vllm_eval.py`
(`eval/results/cpt-v1-dev-offline/`). Baseline column = the same baseline
adapter scored offline.

| Set | Dir | Baseline (offline) | 2e-4 ep1 / ep2 (server) | 5e-5 ep1 / ep2 (offline) |
|---|---|---|---|---|
| Bouquet sentence dev | fra→mos | 24.82 (1.007, 3) | 24.67 (5) / 24.71 (9) | 24.37 (1.033, 5) / 24.97 (1.007, 3) |
| | mos→fra | 27.06 | 27.14 / 26.78 | 27.29 / 27.64 |
| | eng→mos | 24.31 (2) | 24.39 (7) / 24.51 (4) | 24.68 (4) / 24.46 (6) |
| | mos→eng | 28.97 | 29.38 / 29.45 | 29.51 / 28.74 |
| Bouquet paragraph dev | fra→mos | 25.65 (11) | 25.81 / 25.95 | 25.27 (12) / 24.48 (10) |
| | mos→fra | 33.80 | 32.12 / 30.62 | 32.19 / 30.99 |
| moore-web validation | fra→mos | 31.08 (1.064, 30) | 30.88 (41) / 30.84 (46) | 30.55 (1.078, 54) / 30.58 (1.087, 57) |
| | mos→fra | 35.27 | 35.03 / 35.03 | 34.93 / 34.97 |

chrF++ (length ratio, looping outputs) where relevant; no empty outputs.

- **Gentler CPT loops more, not less** (superseded: the null-merge
  baseline rerun also loops 57, see below): moore-web fra→mos looping 30
  (baseline) → 41/46 (2e-4) → 54/57 (5e-5), length ratio 1.064 → 1.087,
  chrF++ −0.5. The model that moved least from HY-MT (higher CPT loss) is
  the worst, so looping does not scale with how much CPT changed the model.
- Bouquet sentence dev: all within ~±0.6 of the baseline (eval noise ~0.2);
  paragraph mos→fra lower for every CPT run (−1.6 to −3.2 vs offline
  baseline, 120 rows, noise ~1).
- **Suspect: something every merged model shares, not CPT content or
  strength.** Ruled out: stop tokens (`eos_token_id` 120020 in config and
  generation config), generation config, tokenizer config, trainable
  params. Remaining difference: the merged `config.json` is re-saved by
  transformers 5 (`rope_scaling` → `rope_parameters` with dynamic-RoPE
  `alpha` 1000, added `eod_token_id: 3`), plus the bf16 merge itself.
- **Control to run next**: a "null CPT" model through the same merge path
  (a LoRA adapter whose update is zero, merged and saved like the others),
  then the same SFT and offline eval. Loops like the CPT runs → the merge
  path is the cause; like the baseline → CPT data is.

### 2026-11-25 — null-merge control: single SFT runs are not reproducible

- vLLM 0.31's transformers 5.17 parses the base and merged configs
  identically (dynamic RoPE, alpha 1000, theta 10000).
- Zero LoRA adapter (CPT config, all 224 `lora_B` zeroed) merged with
  `train/modal_merge.py` into `hy-mt1_5-1_8b-mos-null-merge`: all 354 weight
  tensors bit-identical to HY-MT1.5-1.8B (`lm_head` absent, equal to the
  tied embeddings). Same 1-epoch SFT (eval loss 1.572 vs baseline 1.569),
  offline eval.
- Same weights, same seed (trainer default 42: same LoRA init and data
  order), the training code never reads the added `eod_token_id`: this run
  is a rerun of the baseline, differing most likely only by GPU
  nondeterminism (inferred, not tested). Yet only 3–7% of its
  sentence outputs equal the baseline's, and:

  | moore-web fra→mos | chrF++ | length ratio | looping |
  |---|---:|---:|---:|
  | Baseline | 31.08 | 1.064 | 30 |
  | **Null merge (= baseline rerun)** | 30.88 | 1.082 | **57** |
  | CPT 5e-5 ep1 / ep2 | 30.55 / 30.58 | 1.078 / 1.087 | 54 / 57 |
  | CPT 2e-4 ep1 / ep2 (server) | 30.88 / 30.84 | 1.067 / 1.076 | 41 / 46 |

  Bouquet sentence dev: null merge within −0.44 to −0.08 chrF++ of the
  baseline (eng→mos looping 2 vs 7).
- **Conclusion: the CPT "looping increase" is within SFT run-to-run
  variation.** A rerun of the baseline alone moves moore-web fra→mos
  looping from 30 to 57 and chrF++ by up to ~0.5 in this one pair of runs.
  The baseline's 30 may be the low end; how wide the spread usually is
  needs more repeats. None of the CPT v1 results (2e-4 or 5e-5) differs from
  the baseline beyond that spread, in looping or chrF++.
- Merging is not the cause (the null merge loops like the CPT runs while
  its weights equal the base model's).
- From now on: compare candidates on 3 SFT repeats each (or a deterministic
  SFT), reporting mean and spread; single runs may not separate effects of
  this size.

## Ladder

Each rung is one CPT + SFT + dev eval, compared with the baseline and the
previous best.

1. **Baseline**: 1-epoch SFT, faster settings.
2. **CPT on the mix's own Mooré text**: tells apart "new Mooré text helps"
   from "more Mooré tokens of any kind help".
3. **Our documents**: `data/cpt_mos_v1.jsonl` (630 docs) +
   `cpt_mos_validation_v1.jsonl` (20 docs), 609,252 words (~1.66M tokens
   estimated): council 397,000, raamde 123,504, ministereduc 17,538,
   faso_docs 71,210. Built by `scripts/build_cpt_data.py` (seed 42).
   - **3b. + French↔Mooré parallel text** (not built yet): rung 3 plus
     `cpt_mos_fra_parallel_v1.jsonl` at ~20–30% of CPT tokens, compared with
     rung 3 alone. Label format (`French: …` / `Mossi: …`, random order,
     many pairs per row), no SFT instruction; Council sections matched by
     number, tale volume 5, Ministry posts with both languages; both sides
     decontaminated. Details and reasons in the
     [data-mixing logbook](../logbooks/data-mixing.md) (2026-11-25).
4. **+ moore-web-mono**, decontaminated.
5. **+ HPLT**.
6. **+ gomdeesa OCR** and OCRed burkinadoc scans, if they survive cleaning.
7. **Replay**: mix some French text or translation rows into CPT if into-French
   scores drop (forgetting).

Knobs, one at a time once a data mix wins: CPT tokens (epochs), LoRA rank vs
full fine-tuning (1.8B fits full fine-tuning on an A100-80GB; LoRA r64 may cap
what CPT can learn), CPT learning rate.

## Notes

- CPT loss on Mooré text is not translation quality: select on dev chrF++
  after SFT, not on CPT eval loss.
- A gain found on 1.8B usually carries to a larger model but not always; if
  the recipe moves to TranslateGemma, re-check the winning mix there.
