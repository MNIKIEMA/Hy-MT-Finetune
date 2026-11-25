# Data mixing (`scripts/mix_translation_data.py`, `data.md`)

`mix` builds a weighted ShareGPT-style JSONL for bidirectional translation
between Mooré (`mos`), Dyula (`dyu`), French (`fra`) and English (`eng`) from
three Hub datasets; `validation` builds FLORES+ dev and moore-web validation
files per direction. Weights, filters and the evaluation plan are in `data.md`
(gitignored, local only).

## 2026-11-25 (plan for `cpt_mos_v2.jsonl`; not built yet)

Additions for the next CPT build, written to `cpt_mos_v2.jsonl` /
`cpt_mos_validation_v2.jsonl` so `cpt_mos_v1.jsonl` (all CPT runs so far)
stays reproducible.

- **`sida-bilingual-book/2 SIDA mooré - français.pdf`**: left out of v1
  because plain `pdftotext` mixes its two columns line by line. The Mooré is
  the left half of each A5 page (420 × 595 pt): `pdftotext -x 0 -y 0 -W 205
  -H 595` gives clean Mooré (4,101 words, 34% with a Mooré-only letter, 2.1%
  French function words, front matter the line filter drops). Needs a
  per-file crop in `build_cpt_data.py` (`FASO_DOCS` takes plain paths only).
- **Small gain**: 140 of its 262 sentences (5+ words) are already in v1
  verbatim, via the learner edition `mooreburkina-priority/pdf/20-sida-…`
  (105) and the Kadé facilitator manual (90). New text ≈ 2,000 words (~0.3%
  of v1), minus moore-web validation/test sentences (16 overlaps counted).
- Other v2 candidates already noted: the burkinadoc scans once the moore-ocr
  work can read them, the 3 newer Council sessions (2026-09-24, 10-08,
  10-22; needs a `conseil-ministres` rebuild), new Ministry of Education
  posts (the mirror grows weekly).

## 2026-11-25 (format for French↔Mooré parallel text in CPT; not built yet)

Decided for a later CPT rung ("3b": rung 3's Mooré documents plus parallel
rows, compared with rung 3 alone). To go in a separate
`data/cpt_mos_fra_parallel_v1.jsonl` so rung 3's file stays unchanged.

- **Format: language labels, no instruction.** One pair per block, many
  blocks per row:

  ```
  French: {fra_text}
  Mossi: {mos_text}

  Mossi: {mos_text}
  French: {fra_text}
  ```

- **Not the SFT prompt** (`INSTRUCTION_V1`, "Translate the following segment
  into …"), nor a close variant such as "Translate {src} from {src_lang} into
  {tgt_lang}: {tgt}". CPT puts loss on every token, so the model would learn
  the instruction text, and the CPT rung would partly become translation
  training, so the ladder could no longer separate the effect of CPT data
  from SFT. Wordy natural templates ("This is a sentence … The translation
  in … is") repeat the same filler on every pair for no gain over labels.
- **Order of the two sides random, ~50/50** per block: French first trains
  into Mooré (the direction that loops and runs long), Mooré first trains
  into French.
- **Language names as in SFT** (`LANG_NAMES`: "French", "Mossi").
- **Pack many pairs per row**, in document order, from one document, not one
  short pair per row.
- **Prefer section- or document-level pairs to sentence-aligned ones**, and
  text the SFT mix doesn't already hold:
  - Council: French and Mooré sections share a section number (`I.1.`, …)
    in `conseil-ministres-parsed` `corpus.json` (the sentence-aligned
    Council pairs are already in SFT).
  - Tale volume 5: 30 tales, Mooré page n and its French page n+1.
  - Ministry of Education: the 27 posts with both `fra.txt` and `mos.txt`.
- **Decontaminate both sides**: remove moore-web, FLORES+ and Bouquet
  held-out sentences from the French side too, not only the Mooré.
- **Share**: start at ~20–30% of CPT tokens (a starting guess, not
  measured).

## 2026-11-25 (`build_cpt_data.py`: the CPT v1 documents)

- `data/cpt_mos_v1.jsonl` 630 docs + `cpt_mos_validation_v1.jsonl` 20 docs,
  609,252 Mooré words: council 92 sessions 397,000; raamde 422 articles
  123,504; ministereduc 27 posts 17,538; faso_docs 109 files 71,210. Council
  from `madoss/conseil-ministres-parsed` at `841e617` (up to 2026-09-10).
  Both files uploaded to the `hy-mt-data` volume root under the same names.
- **Raamde: use moore-web's Mooré run (`segments.moore`), not per-unit GlotLID
  labels.** moore-web's `segment_news_data.py` cuts each article at the
  "Kibarã yii"/"Zɩll KAFAANDO" credit marker, else at the first French unit.
  Filtering units on `mos_Latn` instead missed credit lines GlotLID mislabels
  (`cto_Latn`, `sna_Latn`, ...) and the 5 unlabelled articles, and kept 11
  units after the boundary (author names, "Source : SIG", one French
  sentence). The ~100 `bam`/`nso` units are all on the French side.
  moore-web's fallback (no marker, no French label: all Mooré) lets through
  the 4 `recrutement` pages, French job ads: skipped by URL.
- Two more generic filters: a paragraph of 20+ words with no Mooré-only
  letter (ɩ ʋ ẽ ã õ ĩ ũ ɛ) is dropped (6, English credits in the SIDA book;
  no Mooré among those checked), and documents under 5 words. Not applied to
  sentences or shorter paragraphs: syllable drills, proverbs and short
  dialogue are Mooré without those letters.
- **Tale volume 5** (`mooreburkina-priority/apps/mos-contes-volume-5/text/`,
  from media.ipsapps.org/mos/ora/vol5) is 30 tales as 60 one-language files:
  odd numbers Mooré (24–42% of words with a Mooré-only letter), even numbers
  the French translation (0%). No page-level language in the archive (every
  segment says `['mos', 'fra']`), but only the 30 Mooré pages have audio
  timings in `segments.jsonl` (Mooré narration). The build reads only those
  pages (`FASO_NARRATED_APPS`); reading all 60 and filtering French lines let
  fragments through. Volume 5 is also a document-level fra↔mos parallel set
  (30 tales), unused so far.
- Option A (decided with the user): held-out sentences removed one by one,
  1,660 sentences + 365 pieces. Three cases needed more than exact sentence
  matching: a held-out sentence spanning a Raamde title and its first unit,
  one our splitter cut at `:` inside quoted speech (kade SIDA), and junk
  held-out "sentences" like `a b c d` (list letters) that match ordinary
  text as substrings. So: substring matching on the whole document's
  normalized text, dropping every sentence that overlaps a match, only for
  held-out sentences of 4+ words and 20+ characters. 10 shorter stock phrases
  (`kibarã yii faso net`, `piig la a yiibu`) stay inside longer sentences.
- French filter (share of French function words ≥ 0.25 on lines/sentences of
  3+ words): flags 8 of 15,938 Council Mooré sentences (all French or page
  headers) and 11,072 of 16,635 Council French sentences. It removed 48 French
  sentences left in Council/Raamde Mooré sections. It misses short French
  fragments, which is why the three *Du Mooré au Français* lesson PDFs are
  left out; the Mooré reading and grammar books are kept.
- burkinadoc literacy PDFs write `ɡ` (U+0261) and Greek `ῖ`, the UDHR text
  `ʊ` for `ʋ`; Raamde has 1.1k no-break spaces and 25k combining tildes
  overall. All text is mapped to the standard letters and NFC.
- Left out for now: dictionaries, burkinadoc scans (wait for the moore-ocr
  work), the two-column SIDA book and the expert batch.

## 2026-11-25 (CPT Mooré sources: sizes and held-out overlap)

- Measured our own Mooré document sources for CPT (table in
  [the CPT plan](../experiments/2026-11-24-hy-mt15-1_8b-mos-cpt-plan.md)):
  Council reports 421.6k words (93 sessions), Raamde 127.6k (422 articles),
  Ministry of Education posts 17.6k (27), faso-web-docs running text ~100k.
- **All 93 Council sessions contain moore-web validation/test Mooré
  sentences** (1,272 sentences), and 163 of 429 Raamde articles do (200):
  moore-web splits by row, not by document. So CPT decontamination must
  drop sentences, not documents. Bouquet is nearly clean: 4 test hits
  (New Year message, mooreburkina), 0 dev hits.
- Most faso-web-docs Mooré PDFs from burkinadoc are scans (~20 extractable
  words per page) and need OCR. The dictionaries are most of the bucket's
  Mooré words but are lexicon entries.
- `madoss/ministereduc-bf` is a bucket, not a dataset; the local mirror is
  `ministereduc_bf/data/` (27 `mos.txt`, matched the bucket on 2026-11-25).

## 2026-11-24 (`convert`: the TranslateGemma French↔Mooré mix)

- **Goal**: compare HY-MT1.5-1.8B and TranslateGemma 4B on the same data
  (plan in `translategemma-finetune/.agents/logbooks/2026-10-29-plan-fair-hy-mt-comparison.md`).
  `convert` writes a split of `madoss/translategemma-fra-mos-mix` (revision
  `ca1409b`) as v1 ShareGPT rows: same rows, order and direction, codes
  `fr`/`mos_Latn` mapped to French/Mossi.
- `fra_mos_mix_train_v1.jsonl`: 50,000 rows (fra→mos 29,940, mos→fra 20,060).
  `fra_mos_mix_validation_v1.jsonl`: 1,000 rows, 500 moore-web validation
  pairs per direction (TranslateGemma's validation set, not FLORES). Both
  uploaded to the `hy-mt-data` volume.
- **Not identical to the fra↔mos slice of `mixed_v1.jsonl`**: same direction
  counts, but 1,760 of 50,000 rows differ (48,240 match exactly). The
  TranslateGemma mix also holds out BOUQuET dev/test, which drops a few pool
  rows and shifts the seeded sampling.

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
