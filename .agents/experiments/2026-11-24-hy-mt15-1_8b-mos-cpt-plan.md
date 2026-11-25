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
- ~~**Held-out check**~~ Done 2026-11-25 in `scripts/build_cpt_data.py`:
  sentence-level removal of moore-web validation/test, FLORES+ dev/devtest and
  Bouquet dev/test (sentence and paragraph level) Mooré, with the mixer's
  `normalize_text`; the build fails if one of 4+ words and 20+ characters is
  still inside a document.

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
