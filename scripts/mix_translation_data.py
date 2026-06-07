#!/usr/bin/env python3
"""Build mixed fr/en -> mos training JSONL from weighted data buckets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BucketSpec:
    display_name: str
    source_columns: list[str]
    target_columns: list[str]
    default_source_lang: str
    default_target_lang: str


@dataclass(frozen=True)
class TranslationExample:
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str


FR_MOS_ORIGINAL = "fr_mos_original"
FR_MOS_ROUNDTRIP = "fr_mos_roundtrip"
EN_MOS = "en_mos"
FR_MOS_SYNTHETIC = "fr_mos_synthetic"


MIXES = {
    "default": {
        FR_MOS_ORIGINAL: 0.35,
        FR_MOS_ROUNDTRIP: 0.15,
        EN_MOS: 0.45,
        FR_MOS_SYNTHETIC: 0.05,
    },
    "stage1": {
        FR_MOS_ORIGINAL: 0.25,
        FR_MOS_ROUNDTRIP: 0.15,
        EN_MOS: 0.60,
        FR_MOS_SYNTHETIC: 0.00,
    },
    "stage2": {
        FR_MOS_ORIGINAL: 0.40,
        FR_MOS_ROUNDTRIP: 0.15,
        EN_MOS: 0.45,
        FR_MOS_SYNTHETIC: 0.00,
    },
    "stage3": {
        FR_MOS_ORIGINAL: 0.50,
        FR_MOS_ROUNDTRIP: 0.10,
        EN_MOS: 0.40,
        FR_MOS_SYNTHETIC: 0.00,
    },
}

BUCKET_SPECS = {
    FR_MOS_ORIGINAL: BucketSpec(
        display_name="fr -> mos original/high-quality",
        source_columns=["source_text", "french"],
        target_columns=["target_text", "moore"],
        default_source_lang="French",
        default_target_lang="Moore",
    ),
    FR_MOS_ROUNDTRIP: BucketSpec(
        display_name="fr -> mos round-tripped or synthetic",
        source_columns=["source_text", "french_backtranslated", "french"],
        target_columns=["target_text", "moore"],
        default_source_lang="French",
        default_target_lang="Moore",
    ),
    EN_MOS: BucketSpec(
        display_name="en -> mos",
        source_columns=["source_text", "eng_Latn"],
        target_columns=["target_text", "mos_Latn", "moore"],
        default_source_lang="English",
        default_target_lang="Moore",
    ),
    FR_MOS_SYNTHETIC: BucketSpec(
        display_name="NLLB translated-French -> mos synthetic",
        source_columns=[
            "source_text",
            "eng_Latn_to_fra_Latn",
            "fra_Latn",
            "french",
        ],
        target_columns=["target_text", "mos_Latn", "moore"],
        default_source_lang="French",
        default_target_lang="Moore",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mix translation buckets into ShareGPT-style JSONL. Local JSONL paths work "
            "without extra dependencies. Hugging Face dataset ids require `uv sync --extra data`."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=sorted(MIXES), default="default")
    parser.add_argument("--total-examples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--instruction",
        help=(
            "Optional instruction template for the user message. Supports "
            "{source_lang} and {target_lang}; the source text is appended after a blank line. "
            "For Hy-MT 1.5 use: 'Translate the following segment into {target_lang}, "
            "without additional explanation.'"
        ),
    )
    parser.add_argument(
        "--fr-mos-original",
        dest=FR_MOS_ORIGINAL,
        help=f"Input for {BUCKET_SPECS[FR_MOS_ORIGINAL].display_name}.",
    )
    parser.add_argument(
        "--fr-mos-roundtrip",
        dest=FR_MOS_ROUNDTRIP,
        help=f"Input for {BUCKET_SPECS[FR_MOS_ROUNDTRIP].display_name}.",
    )
    parser.add_argument(
        "--en-mos",
        dest=EN_MOS,
        help=f"Input for {BUCKET_SPECS[EN_MOS].display_name}.",
    )
    parser.add_argument(
        "--fr-mos-synthetic",
        dest=FR_MOS_SYNTHETIC,
        help=f"Input for {BUCKET_SPECS[FR_MOS_SYNTHETIC].display_name}.",
    )
    parser.add_argument("--a", dest=FR_MOS_ORIGINAL, help=argparse.SUPPRESS)
    parser.add_argument("--b", dest=FR_MOS_ROUNDTRIP, help=argparse.SUPPRESS)
    parser.add_argument("--c", dest=EN_MOS, help=argparse.SUPPRESS)
    parser.add_argument("--d", dest=FR_MOS_SYNTHETIC, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate source/target pairs per bucket.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def read_hf_dataset(name: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(name, split="train")
    return [dict(row) for row in dataset]


def load_rows(spec: str) -> list[dict[str, Any]]:
    if spec.startswith("hf:"):
        return read_hf_dataset(spec.removeprefix("hf:"))
    path = Path(spec)
    if path.exists():
        return read_jsonl(path)
    return read_hf_dataset(spec)


def required_value(row: dict[str, Any], column: str) -> Any:
    try:
        return row[column]
    except KeyError as exc:
        available = ", ".join(sorted(row))
        raise KeyError(
            f"Missing column {exc.args[0]!r}. Available columns: {available}. "
            "Expected data contract columns: source_text, target_text, source_lang, target_lang."
        ) from exc


def required_text(row: dict[str, Any], column: str, label: str | None = None) -> str:
    value = required_value(row, column)
    if value is None:
        raise ValueError(f"Column {(label or column)!r} must not be null.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Column {(label or column)!r} must not be empty.")
    return text


def find_text_column(row: dict[str, Any], columns: list[str]) -> str | None:
    for column in columns:
        if column in row:
            return column
    return None


def required_text_from_aliases(
    row: dict[str, Any],
    bucket_name: str,
    columns: list[str],
    field_name: str,
) -> str:
    column = find_text_column(row, columns)
    if column:
        return required_text(row, column, field_name)
    available = ", ".join(sorted(row))
    accepted = ", ".join(columns)
    raise KeyError(
        f"Missing {field_name!r} for bucket {bucket_name!r}. "
        f"Accepted columns: {accepted}. Available columns: {available}."
    )


def language_value(row: dict[str, Any], column: str, default: str) -> str:
    if column in row:
        return required_text(row, column)
    return default


def normalize_example(row: dict[str, Any], bucket_name: str) -> TranslationExample:
    spec = BUCKET_SPECS[bucket_name]
    return TranslationExample(
        source_text=required_text_from_aliases(
            row,
            bucket_name,
            spec.source_columns,
            "source_text",
        ),
        target_text=required_text_from_aliases(
            row,
            bucket_name,
            spec.target_columns,
            "target_text",
        ),
        source_lang=language_value(row, "source_lang", spec.default_source_lang),
        target_lang=language_value(row, "target_lang", spec.default_target_lang),
    )


def dedupe_examples(examples: Iterable[TranslationExample]) -> list[TranslationExample]:
    seen = set()
    unique = []
    for example in examples:
        key = (example.source_text, example.target_text, example.source_lang, example.target_lang)
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def load_bucket(args: argparse.Namespace, bucket_name: str) -> list[TranslationExample]:
    input_spec = getattr(args, bucket_name)
    if not input_spec:
        return []
    rows = load_rows(input_spec)
    examples = []
    for row_number, row in enumerate(rows, start=1):
        try:
            examples.append(normalize_example(row, bucket_name))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Bucket {bucket_name!r} row {row_number}: {exc}") from exc
    if args.dedupe:
        examples = dedupe_examples(examples)
    if not examples:
        raise ValueError(f"Bucket {bucket_name!r} has no usable examples.")
    return examples


def allocation(
    weights: dict[str, float],
    available: dict[str, list[TranslationExample]],
    total: int,
) -> dict[str, int]:
    active_weights = {bucket: weight for bucket, weight in weights.items() if weight > 0 and available.get(bucket)}
    if not active_weights:
        raise ValueError("No active buckets. Provide at least one bucket input with a non-zero stage weight.")

    weight_sum = sum(active_weights.values())
    raw_counts = {bucket: (weight / weight_sum) * total for bucket, weight in active_weights.items()}
    counts = {bucket: int(value) for bucket, value in raw_counts.items()}
    remainder = total - sum(counts.values())

    order = sorted(raw_counts, key=lambda bucket: raw_counts[bucket] - counts[bucket], reverse=True)
    for bucket in order[:remainder]:
        counts[bucket] += 1
    return counts


def sample_pairs(
    rng: random.Random,
    pairs: list[TranslationExample],
    count: int,
) -> list[TranslationExample]:
    if count <= len(pairs):
        return rng.sample(pairs, count)
    sampled = list(pairs)
    sampled.extend(rng.choice(pairs) for _ in range(count - len(pairs)))
    rng.shuffle(sampled)
    return sampled


def build_prompt(source: str, source_lang: str, target_lang: str, instruction: str | None) -> str:
    if instruction:
        rendered_instruction = instruction.format(source_lang=source_lang, target_lang=target_lang)
        return f"{rendered_instruction}\n\n{source}"
    return source


def to_sharegpt(
    source: str,
    target: str,
    source_lang: str,
    target_lang: str,
    instruction: str | None,
) -> dict[str, Any]:
    prompt = build_prompt(source, source_lang, target_lang, instruction)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    buckets = {bucket_name: load_bucket(args, bucket_name) for bucket_name in BUCKET_SPECS}
    counts = allocation(MIXES[args.stage], buckets, args.total_examples)

    examples = []
    for bucket_name, count in counts.items():
        for example in sample_pairs(rng, buckets[bucket_name], count):
            examples.append(
                to_sharegpt(
                    example.source_text,
                    example.target_text,
                    example.source_lang,
                    example.target_lang,
                    args.instruction,
                )
            )

    rng.shuffle(examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {args.output}")
    for bucket_name, spec in BUCKET_SPECS.items():
        if bucket_name in counts:
            print(f"{bucket_name} ({spec.display_name}): {counts[bucket_name]}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
