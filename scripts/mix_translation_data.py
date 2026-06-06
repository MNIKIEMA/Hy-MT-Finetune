#!/usr/bin/env python3
"""Build mixed fr/en -> mos training JSONL from weighted data buckets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


MIXES = {
    "default": {"a": 0.35, "b": 0.15, "c": 0.45, "d": 0.05},
    "stage1": {"a": 0.25, "b": 0.15, "c": 0.60, "d": 0.00},
    "stage2": {"a": 0.40, "b": 0.15, "c": 0.45, "d": 0.00},
    "stage3": {"a": 0.50, "b": 0.10, "c": 0.40, "d": 0.00},
}

BUCKET_HELP = {
    "a": "fr -> mos original/high-quality",
    "b": "fr -> mos round-tripped or synthetic",
    "c": "en -> mos",
    "d": "NLLB translated-French -> mos synthetic",
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
    parser.add_argument("--a", help=f"Bucket A input: {BUCKET_HELP['a']}")
    parser.add_argument("--b", help=f"Bucket B input: {BUCKET_HELP['b']}")
    parser.add_argument("--c", help=f"Bucket C input: {BUCKET_HELP['c']}")
    parser.add_argument("--d", help=f"Bucket D input: {BUCKET_HELP['d']}")
    parser.add_argument("--dedupe", action="store_true", help="Drop duplicate source/target pairs per bucket.")
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


def required_text(row: dict[str, Any], column: str) -> str:
    value = required_value(row, column)
    if value is None:
        raise ValueError(f"Column {column!r} must not be null.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Column {column!r} must not be empty.")
    return text


def normalize_example(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        required_text(row, "source_text"),
        required_text(row, "target_text"),
        required_text(row, "source_lang"),
        required_text(row, "target_lang"),
    )


def dedupe_examples(examples: Iterable[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    seen = set()
    unique = []
    for source, target, source_lang, target_lang in examples:
        key = (source, target, source_lang, target_lang)
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, target, source_lang, target_lang))
    return unique


def load_bucket(args: argparse.Namespace, bucket: str) -> list[tuple[str, str, str, str]]:
    spec = getattr(args, bucket)
    if not spec:
        return []
    rows = load_rows(spec)
    examples = []
    for row_number, row in enumerate(rows, start=1):
        try:
            examples.append(normalize_example(row))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Bucket {bucket.upper()} row {row_number}: {exc}") from exc
    if args.dedupe:
        examples = dedupe_examples(examples)
    if not examples:
        raise ValueError(f"Bucket {bucket.upper()} has no usable examples.")
    return examples


def allocation(
    weights: dict[str, float],
    available: dict[str, list[tuple[str, str, str, str]]],
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
    pairs: list[tuple[str, str, str, str]],
    count: int,
) -> list[tuple[str, str, str, str]]:
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

    buckets = {bucket: load_bucket(args, bucket) for bucket in ("a", "b", "c", "d")}
    counts = allocation(MIXES[args.stage], buckets, args.total_examples)

    examples = []
    for bucket, count in counts.items():
        for source, target, source_lang, target_lang in sample_pairs(rng, buckets[bucket], count):
            examples.append(to_sharegpt(source, target, source_lang, target_lang, args.instruction))

    rng.shuffle(examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {args.output}")
    for bucket in ("a", "b", "c", "d"):
        if bucket in counts:
            print(f"{bucket.upper()} ({BUCKET_HELP[bucket]}): {counts[bucket]}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
