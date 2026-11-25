# Plan — continued pretraining (CPT) on Mooré text for HY-MT1.5-1.8B (not started)

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
- **Held-out check**: drop CPT documents/sentences that contain moore-web
  validation/test, Bouquet dev/test or FLORES+ dev/devtest sentences, with
  the mixer's `normalize_text` matching. Web Mooré text is likely to overlap
  moore-web test (same sites).

## Data candidates (sizes not yet measured)

| Source | What it is | Notes |
|---|---|---|
| Mooré side of the training mix | Mooré sentences of `fra_mos_mix_train_v1.jsonl` | No new text: controls for "more passes over the same Mooré" |
| `madoss/moore-web-mono` | Web Mooré | Check overlap with moore-web test first |
| `madoss/mos-latn-hplt` | HPLT Mooré documents (1,892 docs, 28.5 MB with html/xml fields) | Small; `text` column, LID `prob` to filter |
| `burkimbia/gomdeesa-ocr-brut` | Raw OCR | Needs cleaning; check OCR noise before use |
| `burkimbia/sidbi-ziri-dataset` | — | Large; skipped for now |

First step: measure each source (documents, words, tokens with the HY-MT
tokenizer, share already in the held-out sets) without pulling more than a
sample, and decide what to download.

## Ladder

Each rung is one CPT + SFT + dev eval, compared with the baseline and the
previous best.

1. **Baseline**: 1-epoch SFT, faster settings.
2. **CPT on the mix's own Mooré text**: tells apart "new Mooré text helps"
   from "more Mooré tokens of any kind help".
3. **+ moore-web-mono**, decontaminated.
4. **+ HPLT**.
5. **+ gomdeesa OCR**, if it survives cleaning.
6. **Replay**: mix some French text or translation rows into CPT if into-French
   scores drop (forgetting).

Knobs, one at a time once a data mix wins: CPT tokens (epochs), LoRA rank vs
full fine-tuning (1.8B fits full fine-tuning on an A100-80GB; LoRA r64 may cap
what CPT can learn), CPT learning rate.

## Notes

- CPT loss on Mooré text is not translation quality: select on dev chrF++
  after SFT, not on CPT eval loss.
- A gain found on 1.8B usually carries to a larger model but not always; if
  the recipe moves to TranslateGemma, re-check the winning mix there.
