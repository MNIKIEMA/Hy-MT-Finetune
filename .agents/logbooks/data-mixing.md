# Data mixing (`scripts/mix_translation_data.py`, `data.md`)

`mix` builds a weighted ShareGPT-style JSONL for bidirectional translation
between Mooré (`mos`), Dyula (`dyu`), French (`fra`) and English (`eng`) from
three Hub datasets; `validation` builds FLORES+ dev and moore-web validation
files per direction. Weights, filters and the evaluation plan are in `data.md`
(gitignored, local only).

## 2026-10-28 (Dyula spelling variants)

- **All training Dyula is in the standard Manding alphabet** (ɛ, ɔ, ɲ;
  é/è/ê 0.1‰, ô 0.0‰ of letters), while FLORES+ Dyula is written in a
  French-style spelling (é/è for ɛ, ô for ɔ, gn for ɲ, dj for j). The model
  never sees the second form, so it misreads FLORES+ sources and writes
  standard spelling against FLORES+ references
  ([evaluation logbook](evaluation.md)).
- Idea for the next mix: add rows whose Dyula *source* is converted to the
  French-style spelling (keep standard spelling on the target side), so the
  model reads both. Not implemented yet.

## 2026-10-27 (domain makeup of the 200k mix)

- **The mix is mostly religious and government text.** Measured from `trg_url`
  hosts and `source` labels after filtering, for the 200k-row seed-42 mix:
  NLLB rows 54% (of which ≥35% of the whole mix comes from jw.org,
  bibles.org, bible.is or beblia.com; NLLB rows without a URL are unclassified),
  Council of Ministers reports ~29%, dictionary rows ~8.5%, rows checked by a
  person 7.5%.
- NLLB hosts after filtering: eng-mos jw.org 52%, no URL 32%, beblia 4%;
  eng-dyu bibles.org 65%, jw.org 6%, bible.is 6%; fra-dyu bibles.org 64%,
  jw.org 7%. Joined pairs (nllb `fra-mos`, `mos-dyu`) reuse these English
  sentences.
- Likely part of the train/validation loss gap (validation is ~83% FLORES,
  general domain). Next mix: cap religious text (e.g. ~20%) via `trg_url`, or
  add general-domain data.

## 2026-10-27 (rewrite: fr/en -> mos buckets to bidirectional pairs)

- **Goal changed** from fr/en -> mos to both directions between mos, dyu, fra
  and eng. fra<->eng is not trained: not the goal, and the base model already
  handles it. 10 directions.
- **Pair shares**: mos-fra 25%, mos-eng 20%, dyu-fra 22%, dyu-eng 16%,
  dyu-mos 17%; each source has a share within its pair (table in `data.md`).
  Each sampled row gets one direction: 60% into mos/dyu for pairs with fra/eng
  (`--into-local-ratio`), 50/50 for dyu-mos.
- **madoss/faso-web-parallel has no mos-fra or mos-eng config**; it is the main
  source of Dyula and of nearly all dyu-mos (Council reports joined through
  French).
- **~16% of NLLB "Dyula" rows are GlotLID `bam_Latn`** (eng-dyu 8,935 of
  56,924; fra-dyu 9,158 of 56,625). The NLLB filter requires the expected
  `src_lid`/`trg_lid` with probability ≥ 0.9, plus `score >= 1.06`,
  `length_ratio >= 0.5`.
- **moore-web `news` is excluded**: its Mooré side is a summary rewrite.
  moore-web has 5,466 `reviewed` train rows; `lexicon_entries` (16,545) are
  single words, capped at 10% of mos-fra.
- **Decontamination caught a real leak**: faso-web and moore-web both contain
  Council of Ministers reports. Matching any side or the pivot sentence against
  moore-web validation/test and FLORES+ dev/devtest dropped 1,035 faso dyu-fra
  rows and 961 faso dyu-mos rows. NLLB eng-mos lost 1,529 rows; NLLB says it
  removed FLORES, so these are likely moore-web eval sentences or short common
  phrases (not inspected).
- **Usable pools after filters and dedupe** (seed 42): moore_web_reviewed
  5,374; moore_web_aligned 10,897; nllb_fra_mos_pivot 11,153;
  moore_web_lexicon 16,569; nllb_eng_mos 51,120; faso_dyu_fra_sentences
  13,339; nllb_fra_dyu 23,112; faso_dyu_fra_lexicon 12,396; nllb_eng_dyu
  31,855; faso_dyu_eng_examples 1,162; faso_dyu_eng_lexicon 12,362;
  faso_dyu_mos_pivot 9,820; nllb_mos_dyu_pivot 2,610.
- **dyu-mos is small**: ~12.4k usable rows sampled 34k times (2.6-3.3x per
  row in one epoch). Reviewed mos-fra rows are repeated 2.8x. Watch these for
  memorization before adding epochs.
- **Thresholds are starting points** picked from score distributions
  (moore-web `laser_score` p25 0.70, `comet_qe` p10 0.51; faso dyu-fra
  `laser_score` p25 0.75), not validated by reading rows near the cutoffs.
- **Direction metadata not used**: moore-web `original_lang` and faso
  `is_source_orig` are known for some rows, but locking direction would leave
  mos->fra and dyu->fra mostly dictionary and NLLB rows (most known-original
  rows are French-original). Kept random direction; adding an `original_lang`
  field for later ablation is an open option.
- HY-MT1.5 needs `--instruction-version v1` (instruction in a system message).
