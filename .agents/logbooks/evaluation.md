# Evaluation (`scripts/evaluate_translation.py`, `train/modal_vllm_server.py`)

`modal_vllm_server.py` serves the base model (`hy-mt`) and LoRA adapters
(one served name each) with vLLM on Modal. `evaluate_translation.py` sends
FLORES+ `devtest` (10 directions) and moore-web `test` (fra↔mos) to it with
the training prompt, and scores chrF++ / BLEU per direction.

## TODO (next session)

- **Try `no_repeat_ngram_size` at decoding** (e.g. 3, 4, 5) on an existing
  adapter (the 1-epoch SFT baseline), with `train/modal_vllm_eval.py` on
  Bouquet dev and moore-web validation. The goal is a better translation
  model, not only fewer loops: keep a setting only if chrF++ (and BLEU) are
  not lower than at the current setting (greedy, repetition penalty 1.05)
  beyond eval noise (~0.2 chrF++ on Bouquet sentence dev, ~1 on paragraph
  dev), and report looping and length ratio next to them. Repetition
  penalty 1.15 already showed the trap: loops nearly gone, chrF++ −1 to −2
  (2026-10-28 entry). Mooré repeats short function words legitimately
  ("n", "a", "la", "tɩ"), so small n-gram sizes may hurt; check both
  directions, since the setting also applies into French and English.
  Needs a new option in `modal_vllm_eval.py`. Checked in vLLM 0.31.0:
  `SamplingParams` has no `no_repeat_ngram_size` and no per-request
  `logits_processors` (V1 engine: logits processors are engine-level), so
  blocking repeated n-grams needs an engine-level logits processor.
- **Also try vLLM's `repetition_detection`** (`RepetitionDetectionParams`
  in `SamplingParams`, vLLM 0.31.0): it detects repeating n-gram patterns
  in the output and *ends generation early*; it does not block the
  repeat. A cut-off loop keeps the useful start of the translation and
  fixes the length ratio, so it may help chrF++ where `no_repeat_ngram_size`
  could hurt legitimate Mooré repetition. Its parameters were not read
  yet. Same rule: keep it only if chrF++ is not lower.

## 2026-11-25 (Burkimbia leaderboard chrF++ = mean of sentence chrF++)

- The 2026-10-28 submission (`mos-dyu` checkpoint-2400, 200k mix) scored
  **24.83** fra→mos on the leaderboard (Bouquet sentence `test`, 854 rows);
  our `summary.json` says 25.57 for the same translations (all 854
  identical to the submitted CSV).
- **The leaderboard averages sentence-level chrF++** (sacrebleu
  `CHRF(word_order=2).sentence_score` per row, then mean): that gives
  exactly 24.83. Ours is corpus-level chrF++. Not the cause: lowercasing
  (26.19), chrF without word n-grams (26.42); GlotLID filtering removed
  nothing here.
- So leaderboard ≈ ours − 0.7 for this system, but the offset depends on
  the system (corpus and sentence averages weigh long and short sentences
  differently). Compare with leaderboard systems on the mean-sentence
  number, not by subtracting 0.7.
- Leaderboard fra→mos (chrF++ / MetricX, lower is better): NLLB-200 3.3B
  30.72 / 13.98, OMT-NLLB 28.73 / 14.12, NLLB-200 1.3B 28.17 / 14.55,
  OMT-LLaMA-3B 25.32 / 14.39, our 200k-mix HY-MT 1.8B 24.83. We are ~6
  behind NLLB-200 3.3B and ~3.3 behind NLLB-200 1.3B: much larger than
  anything measured for CPT (< 0.5).
- TODO: report mean sentence chrF++ next to corpus chrF++ in
  `evaluate_translation.py`'s `score()`.

## 2026-11-25 (redeploying the server between models)

- **After `modal deploy` with another `HY_MT_BASE_MODEL`/`HY_MT_LORAS`, the
  previous warm container keeps answering** with the old models (seen:
  `/v1/models` listed `hy-mt`, `baseline` after deploying `cpt-ep1`).
  Requests for the new adapter name would fail and be scored as empty.
  Stop the app's containers first (`modal container stop --yes <id>`, ids
  from `modal container list --json`), then wait until `/v1/models` lists
  the new name. `container stop` asks for confirmation without `--yes`.
- Decided: evaluate comparisons offline instead (one `modal run` that loads
  the base model and adapter with vLLM, scores, exits), so nothing is left
  deployed and runs can go in parallel.

## 2026-11-25 (`train/modal_vllm_eval.py`: offline evaluation, and eval noise)

- **Offline evaluator**: one `modal run` loads the base model and adapters
  with `vllm.LLM` (same engine settings as the server: bf16,
  `max_model_len` 1024, eager), translates with `LLM.chat`, returns the
  texts and exits. Sets, prompts, scoring and output layout are
  `evaluate_translation.py`'s; datasets are loaded locally. 7,478 rows
  (Bouquet dev both levels + moore-web validation) in a few minutes on an
  A100.
- **Checked against the server** on the 1-epoch SFT baseline adapter
  (`hy-mt1_5-1_8b-sft1ep-baseline/checkpoint-782`): chrF++ moore-web
  validation ±0.01, Bouquet sentence dev −0.20 to +0.20, Bouquet paragraph
  dev −0.50 to +1.07. 61% of outputs identical; of the differing ones,
  93.5% share a prefix and diverge later (median 39% of the output):
  greedy bf16 decoding under different batching, not a prompt difference.
- **So a single eval has noise**: up to ~0.2 chrF++ on Bouquet sentence dev
  (504 rows), ~1 on paragraph dev (120 rows), ~0 on moore-web (2,491).
  Looping counts move too (moore-web mos→fra 4 vs 11, Bouquet paragraph
  fra→mos 7 vs 11). Treat smaller differences between runs as no
  difference.
- **Correction (same day, null-merge control)**: eval noise is the small
  part. Retraining the same SFT from identical weights and seed moves
  moore-web fra→mos looping from 30 to 57 and chrF++ by up to ~0.5 in one
  pair of runs (most likely GPU nondeterminism; [training
  logbook](training.md)). The CPT v1 looping rise (30 → 41–57) is within
  that, not beyond it as first written here.
  Compare candidates on several SFT repeats.

## 2026-11-24 (API key)

- `evaluate_translation.py` reads `VLLM_API_KEY` from the environment only
  and `uv run` does not load `.env` by default: run it with
  `uv run --env-file .env ...`, or every request is rejected.
- `--sets bouquet` always scores eng↔mos too; there is no direction filter.

## 2026-10-28 (dyu→mos loops and FLORES+ Dyula spelling)

- **The dyu→mos loops are not memorized training phrases.** Of the 197
  looping outputs (FLORES+ `devtest`, checkpoint-2400, penalty 1.05) almost
  every one repeats a different phrase: the most frequent repeated trigram
  ("n kɩt tɩ", "zĩig ning sẽn") occurs in 4 outputs, most in 1, and no
  Council-of-Ministers formulas show up. What repeats is common Mooré glue
  ("sẽn na n", "a na n") or the model's own last clause, until
  `max_tokens`.
- **The loops follow a source the model did not understand**: the output
  starts as fluent but unrelated Mooré (an athletics sentence becomes
  soldiers and war; a USOC sentence becomes "our country and its people")
  and then cycles. Many such sources are in FLORES+'s French-style Dyula
  spelling ("diabéti", "sôrô", "gnanaman").
- **FLORES+ Dyula is the same language in a different spelling, not rarer
  words.** Letters per thousand, FLORES+ `devtest` vs the 62,965 Dyula
  targets of `mixed_v1`: ɛ 1.4 vs 44.3, ɔ 2.4 vs 42.0, ɲ 0.3 vs 8.6,
  é/è/ê 45.7 vs 0.1, ô 28.4 vs 0.0. Word counts (train / FLORES+):
  bɛ 558 / 5 vs bé 3 / 169; dɔ 3,073 / 14 vs dô 1 / 93; sɔrɔ 3,623 / 8 vs
  sôrô 0 / 54; jamana 7,075 / 20 vs djamana 0 / 64. Words without these
  letters match ("ka" 1,043 in FLORES+). 45.2% of FLORES+ Dyula word tokens
  never occur in the training Dyula, mostly from spelling (not separated
  from real vocabulary differences).
- Consequences: into-dyu FLORES+ chrF++ is penalized even for correct
  standard-spelling output (fra→dyu, eng→dyu flat); out of dyu the model
  half-reads the source (loops, hallucination); the Burkimbia leaderboard
  scores fra-dyu on this set too. Options: map our Dyula output to the
  FLORES+ spelling before a leaderboard submission (ɛ→è/é, ɔ→ô, ɲ→gn,
  j→dj; ɛ→é vs è needs rules, check on FLORES+ `dev` first), and train on
  spelling variants (see the data-mixing logbook). mos→dyu (117 loops) and
  Bouquet eng→mos paragraphs (55) are not yet inspected; their sources are
  not in this spelling, so the cause may differ.

## 2026-10-28 (Burkimbia leaderboard submission)

- The Burkimbia MT leaderboard scores Mooré on Bouquet sentence-level
  `test` and Dyula on FLORES+ `devtest`. Its CSV has `direction`
  (`fra-mos`), `id` (Bouquet `uniq_id`, FLORES+ `id`) and `translation`;
  the predictions of `evaluate_translation.py` map onto it directly.
- **The leaderboard runs language ID itself** (GlotLID): at upload it
  rejects a direction where under half of the translations are in the
  target language (dyu accepts `dyu_Latn` or `bam_Latn`), and it checks
  every sentence again when scoring. Submitters don't compute it.
- Submission file (local): `eval/submissions/hy-mt1_5-1_8b-mos-dyu-lora-ckpt2400.csv`,
  fra↔mos from Bouquet and fra↔dyu from FLORES+, repetition penalty 1.05.

## 2026-10-28 (repetition penalty)

- **`--repetition-penalty`** (default 1.05). At 1.15 looping outputs drop
  from hundreds to a handful, but chrF++ only rises where the model was
  looping (FLORES+ into dyu, dyu↔mos, +0.9 to +1.5) and falls 1-2 on
  moore-web and Bouquet; BLEU falls almost everywhere. Keep 1.05 for
  reported scores and compare runs at the same penalty. Details in the
  [experiment](../experiments/2026-10-27-hy-mt15-1_8b-mos-dyu-bidirectional.md).

## 2026-10-28 (Bouquet)

- **`--sets bouquet`** reads `facebook/bouquet`
  `benchmark_data/<level>/<split>/<src>-<tgt>.parquet` for fra↔mos and
  eng↔mos (854 sentences, 198 paragraphs per direction in `test`) and keeps
  `domain` in the predictions. Paragraphs get `--paragraph-max-tokens` 384.
- References are scored as published. `mt-training` normalizes them
  (’ « » → ' ") because NLLB has no tokens for those characters; Hy-MT has
  them, so scores are not directly comparable with `mt-training` Bouquet
  numbers.
- Run Bouquet into its own `--output-dir`: `summary.md` only covers the sets
  of one invocation and would replace an earlier summary in the same folder.

## 2026-10-27 (first server and smoke eval)

- **vLLM 0.31.0 runs HunYuanDenseV1 through its Transformers backend**
  ("`torch.compile` is turned on, but the model … does not support it").
  Its dynamic RoPE (`dynamic_frequency_update`) branches on sequence length
  and crashed CUDA graph capture at startup ("operation not permitted when
  stream is capturing"). The server passes `--enforce-eager`.
- **LoRA works on that backend**: `--enable-lora --max-lora-rank 64
  --lora-modules mos-dyu=/outputs/.../checkpoint-2400`.
- **vLLM 0.31 removed `--disable-log-requests`** (request logs are off by
  default); passing it fails at argument parsing.
- **`scripts/run_inference.py` sends `stop_token_ids: [127960]`**, a token id
  that doesn't exist in HY-MT1.5 (eos is 120020,
  `<｜hy_place▁holder▁no▁2｜>`). `evaluate_translation.py` relies on the
  model's generation config instead.
- The endpoint is public, so the server requires `VLLM_API_KEY` from the
  Modal secret `hy-mt-vllm`; the client reads it from `.env`.
- **FLORES+ Dyula (`dyu_Latn`) uses a French-based orthography** (`ô`, `é`:
  "dôgôtôrôya", "diabéti") while the training data and the model use the
  standard `ɔ`/`ɛ` ("Dɔgɔtɔrɔ"). Character-level scores into Dyula on FLORES+
  understate quality; compare into-dyu FLORES+ scores across models only, and
  add a Dyula set in the standard orthography.
- **The base model can't write Mooré**: for fra→mos it returns French, yet
  scores ~17-18 chrF++ on FLORES+ from names and numbers. That is the floor,
  not a weak translation.
- Smoke eval (20 rows per direction) showed one dyu→mos repetition loop; the
  summary's length ratio flags these.
