#!/usr/bin/env python3
"""Score models served by an OpenAI-compatible server on mos/dyu/fra/eng directions.

Sets: FLORES+ `devtest` for the 10 trained directions, moore-web `test`
for fra->mos and mos->fra, and Bouquet `test` (sentence and paragraph level)
for mos<->fra and mos<->eng. Prompts use the same instruction and language
names as the training mix. Writes one Bouquet-style prediction JSONL per
model, set and direction, plus `summary.json` and `summary.md` (chrF++ / BLEU).

    uv run --extra eval python scripts/evaluate_translation.py \
      --base-url https://<workspace>--hy-mt-vllm-serve.modal.run/v1 \
      --models hy-mt mos-dyu --output-dir eval/results/run1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from sacrebleu.metrics import BLEU, CHRF
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_translation_data import (  # noqa: E402
    LANG_NAMES,
    MOORE_WEB,
    clean_text,
    flores_texts_by_id,
    instruction_for_version,
    read_hf_dataset,
    to_sharegpt,
    validation_directions,
)


def flores_rows(dataset: str, split: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    texts = {lang: flores_texts_by_id(dataset, split, lang) for lang in LANG_NAMES}
    rows = {}
    for source_lang, target_lang in validation_directions():
        source_texts, target_texts = texts[source_lang], texts[target_lang]
        rows[(source_lang, target_lang)] = [
            {"id": str(row_id), "src_text": source_texts[row_id], "ref_text": target_texts[row_id]}
            for row_id in source_texts
            if row_id in target_texts
        ]
    return rows


def moore_web_rows(split: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    rows = {("fra", "mos"): [], ("mos", "fra"): []}
    for row in read_hf_dataset(MOORE_WEB, split=split, config="mos-fra"):
        moore, french = clean_text(row["moore"]), clean_text(row["french"])
        if not moore or not french:
            continue
        rows[("fra", "mos")].append({"id": row["id"], "src_text": french, "ref_text": moore})
        rows[("mos", "fra")].append({"id": row["id"], "src_text": moore, "ref_text": french})
    return rows


BOUQUET = "facebook/bouquet"
BOUQUET_DIRECTIONS = [("fra", "mos"), ("mos", "fra"), ("eng", "mos"), ("mos", "eng")]
BOUQUET_CODES = {"mos": "mos_Latn", "fra": "fra_Latn", "eng": "eng_Latn"}


def bouquet_rows(level: str, split: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    """One Bouquet benchmark file per direction; references are scored as published."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    rows = {}
    for source_lang, target_lang in BOUQUET_DIRECTIONS:
        path = hf_hub_download(
            BOUQUET,
            f"benchmark_data/{level}/{split}/{BOUQUET_CODES[source_lang]}-{BOUQUET_CODES[target_lang]}.parquet",
            repo_type="dataset",
        )
        rows[(source_lang, target_lang)] = [
            {"id": row.uniq_id, "domain": row.domain, "src_text": row.src_text, "ref_text": row.tgt_text}
            for row in pd.read_parquet(path).itertuples()
        ]
    return rows


def prompt_messages(text: str, source_lang: str, target_lang: str, args: argparse.Namespace) -> list[dict[str, str]]:
    instruction = instruction_for_version(args.instruction_version, args.instruction)
    sharegpt = to_sharegpt(
        text, "", LANG_NAMES[source_lang], LANG_NAMES[target_lang], instruction, args.instruction_version
    )
    return sharegpt["messages"][:-1]


async def translate(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    repetition_penalty: float,
) -> str:
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    extra_body={"repetition_penalty": repetition_penalty},
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # retry transient server/network errors
                if attempt == 2:
                    print(f"\nRequest failed after 3 attempts: {exc}", file=sys.stderr)
                    return ""
                await asyncio.sleep(2 ** attempt)
    return ""


def read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    hypotheses = [row["mt_text"] for row in predictions]
    references = [[row["ref_text"] for row in predictions]]
    hypothesis_chars = sum(len(text) for text in hypotheses)
    reference_chars = sum(len(text) for text in references[0])
    return {
        "rows": len(predictions),
        "chrf++": round(CHRF(word_order=2).corpus_score(hypotheses, references).score, 2),
        "bleu": round(BLEU().corpus_score(hypotheses, references).score, 2),
        "empty": sum(1 for text in hypotheses if not text),
        "length_ratio": round(hypothesis_chars / reference_chars, 3) if reference_chars else None,
    }


async def run_direction(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    model: str,
    set_name: str,
    direction: tuple[str, str],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    source_lang, target_lang = direction
    max_tokens = args.paragraph_max_tokens if "paragraph" in set_name else args.max_tokens
    path = args.output_dir / model / set_name / f"{source_lang}_{target_lang}.jsonl"
    if path.exists() and not args.overwrite:
        predictions = read_predictions(path)
    else:
        tasks = [
            translate(
                client,
                semaphore,
                model,
                prompt_messages(row["src_text"], source_lang, target_lang, args),
                max_tokens,
                args.repetition_penalty,
            )
            for row in rows
        ]
        translations = await tqdm_asyncio.gather(*tasks, desc=f"{model} {set_name} {source_lang}->{target_lang}")
        predictions = [
            {"id": row["id"], "src_lang": source_lang, "tgt_lang": target_lang, **row, "mt_text": translation}
            for row, translation in zip(rows, translations, strict=True)
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return {"model": model, "set": set_name, "direction": f"{source_lang}->{target_lang}", **score(predictions)}


def write_summary(results: list[dict[str, Any]], models: list[str], output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = []
    for set_name in dict.fromkeys(result["set"] for result in results):
        by_key = {(r["model"], r["direction"]): r for r in results if r["set"] == set_name}
        directions = list(dict.fromkeys(r["direction"] for r in results if r["set"] == set_name))
        lines += [f"## {set_name}", "", "chrF++ (BLEU), length ratio of output to reference.", ""]
        header = "| Direction | Rows | " + " | ".join(models)
        if len(models) == 2:
            header += f" | {models[1]} − {models[0]}"
        lines += [header + " |", "| --- | ---: |" + " --- |" * (len(models) + (len(models) == 2))]
        for direction in directions:
            cells = [by_key[(model, direction)] for model in models]
            row = f"| {direction} | {cells[0]['rows']} | " + " | ".join(
                f"{c['chrf++']:.2f} ({c['bleu']:.2f}), {c['length_ratio']}" for c in cells
            )
            if len(models) == 2:
                row += f" | {cells[1]['chrf++'] - cells[0]['chrf++']:+.2f}"
            lines.append(row + " |")
        averages = [
            sum(by_key[(model, d)]["chrf++"] for d in directions) / len(directions) for model in models
        ]
        row = "| **average** | | " + " | ".join(f"**{a:.2f}**" for a in averages)
        if len(models) == 2:
            row += f" | **{averages[1] - averages[0]:+.2f}**"
        lines += [row + " |", ""]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


async def main(args: argparse.Namespace) -> None:
    sets = {}
    if "flores" in args.sets:
        sets[f"flores_{args.flores_split}"] = flores_rows(args.flores_dataset, args.flores_split)
    if "moore-web" in args.sets:
        sets[f"moore_web_{args.moore_web_split}"] = moore_web_rows(args.moore_web_split)
    if "bouquet" in args.sets:
        for level in args.bouquet_levels:
            sets[f"bouquet_{level}_{args.bouquet_split}"] = bouquet_rows(level, args.bouquet_split)
    if args.max_examples:
        sets = {
            name: {direction: rows[: args.max_examples] for direction, rows in directions.items()}
            for name, directions in sets.items()
        }

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=300)
    available = {model.id async for model in client.models.list()}
    missing = [model for model in args.models if model not in available]
    if missing:
        raise SystemExit(f"Model(s) not served: {', '.join(missing)}. Served: {', '.join(sorted(available))}.")

    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    for model in args.models:
        for set_name, directions in sets.items():
            for direction, rows in directions.items():
                results.append(await run_direction(client, semaphore, args, model, set_name, direction, rows))
    write_summary(results, args.models, args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, ending in /v1.")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--models", nargs="+", default=["hy-mt"], help="Served model names; first is the baseline.")
    parser.add_argument("--sets", nargs="+", choices=("flores", "moore-web", "bouquet"), default=["flores", "moore-web"])
    parser.add_argument("--flores-dataset", default="openlanguagedata/flores_plus")
    parser.add_argument("--flores-split", default="devtest")
    parser.add_argument("--moore-web-split", default="test")
    parser.add_argument("--bouquet-split", default="test")
    parser.add_argument(
        "--bouquet-levels",
        nargs="+",
        choices=("sentence_level", "paragraph_level"),
        default=["sentence_level", "paragraph_level"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--instruction", help="Instruction template override ({source_lang}, {target_lang}).")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--paragraph-max-tokens",
        type=int,
        default=384,
        help="Bouquet paragraph references reach ~270 Mooré tokens.",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-examples", type=int, help="Rows per direction, for a quick check.")
    parser.add_argument("--overwrite", action="store_true", help="Re-translate even if predictions exist.")
    return parser.parse_args()


if __name__ == "__main__":
    # Importing mix_translation_data already loaded `.env` (HF_TOKEN, VLLM_API_KEY).
    asyncio.run(main(parse_args()))
