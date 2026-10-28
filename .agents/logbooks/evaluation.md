# Evaluation (`scripts/evaluate_translation.py`, `train/modal_vllm_server.py`)

`modal_vllm_server.py` serves the base model (`hy-mt`) and LoRA adapters
(one served name each) with vLLM on Modal. `evaluate_translation.py` sends
FLORES+ `devtest` (10 directions) and moore-web `test` (fra↔mos) to it with
the training prompt, and scores chrF++ / BLEU per direction.

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
