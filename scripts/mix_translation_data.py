#!/usr/bin/env python3
"""Build bidirectional mos/dyu/fra/eng training and validation JSONL.

Training pairs: mos-fra, mos-eng, dyu-fra, dyu-eng, dyu-mos (fra-eng is out of scope).
Each pair gets a fixed share of the mix; within a pair, each source gets a fixed share.
Every sampled row is assigned one direction, biased toward generating mos/dyu.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

LANG_NAMES = {
    "mos": "Mossi",
    "dyu": "Dyula",
    "fra": "French",
    "eng": "English",
}
LOCAL_LANGS = {"mos", "dyu"}
FLORES_CODES = {
    "mos": "mos_Latn",
    "dyu": "dyu_Latn",
    "fra": "fra_Latn",
    "eng": "eng_Latn",
}

INSTRUCTION_V2 = (
    "Translate the following {source_lang} text into {target_lang}, "
    "output only the translation result without additional explanation:"
)
INSTRUCTION_V1 = "Translate the following segment into {target_lang}, without additional explanation."

# Pairs are written (local language, other language).
PAIR_WEIGHTS = {
    ("mos", "fra"): 0.25,
    ("mos", "eng"): 0.20,
    ("dyu", "fra"): 0.22,
    ("dyu", "eng"): 0.16,
    ("dyu", "mos"): 0.17,
}

MOORE_WEB = "madoss/moore-web-parallel"
FASO_WEB = "madoss/faso-web-parallel"
NLLB_BURKINA = "burkimbia/nllb-burkina-parallel"

# Starting thresholds, chosen from the score distributions of each dataset.
MOORE_WEB_MIN_LASER = 0.70
MOORE_WEB_MIN_COMET_QE = 0.52
FASO_MIN_LASER = 0.75
FASO_PIVOT_MIN_LASER = 0.70
FASO_PIVOT_MIN_DIRECT_LASER = 0.75
NLLB_MIN_SCORE = 1.06
NLLB_MIN_LID_PROB = 0.9
NLLB_MIN_LENGTH_RATIO = 0.5


def score_at_least(row: dict[str, Any], column: str, minimum: float) -> bool:
    value = row.get(column)
    return value is not None and value >= minimum


def score_missing_or_at_least(row: dict[str, Any], column: str, minimum: float) -> bool:
    value = row.get(column)
    return value is None or value >= minimum


def moore_web_reviewed(row: dict[str, Any]) -> bool:
    return bool(row["reviewed"])


def moore_web_aligned(row: dict[str, Any]) -> bool:
    # `news` is excluded: its Mooré side is a summary-style rewrite, not a translation.
    return (
        not row["reviewed"]
        and row["source"] in {"conseils", "lexicon", "digital-defs"}
        and score_at_least(row, "laser_score", MOORE_WEB_MIN_LASER)
        and score_missing_or_at_least(row, "comet_qe", MOORE_WEB_MIN_COMET_QE)
    )


def moore_web_lexicon(row: dict[str, Any]) -> bool:
    return row["source"] in {"lexicon_entries", "digital-terms"}


def faso_sentences(row: dict[str, Any]) -> bool:
    if row["unit_type"] == "gloss":
        return False
    if row["review_status"] == "structural":
        return True
    return score_at_least(row, "laser_score", FASO_MIN_LASER)


def faso_lexicon(row: dict[str, Any]) -> bool:
    return row["unit_type"] == "gloss"


def faso_pivot(row: dict[str, Any]) -> bool:
    return score_at_least(row, "laser_score", FASO_PIVOT_MIN_LASER) and score_at_least(
        row, "direct_laser_score", FASO_PIVOT_MIN_DIRECT_LASER
    )


def nllb_filter(src_lid: str, trg_lid: str) -> Callable[[dict[str, Any]], bool]:
    """NLLB Dyula rows are often Bambara, so both sides must match the expected language."""

    def keep(row: dict[str, Any]) -> bool:
        return (
            row["score"] >= NLLB_MIN_SCORE
            and row["length_ratio"] >= NLLB_MIN_LENGTH_RATIO
            and row["src_lid"] == src_lid
            and row["trg_lid"] == trg_lid
            and row["src_lid_prob"] >= NLLB_MIN_LID_PROB
            and row["trg_lid_prob"] >= NLLB_MIN_LID_PROB
        )

    return keep


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str
    pair: tuple[str, str]
    local_column: str
    other_column: str
    weight: float
    keep: Callable[[dict[str, Any]], bool]
    pivot_column: str | None = None


# Order matters: when deduplicating within a pair, the earlier source keeps the row.
SOURCES = [
    SourceSpec("moore_web_reviewed", MOORE_WEB, "mos-fra", ("mos", "fra"), "moore", "french", 0.30, moore_web_reviewed),
    SourceSpec("moore_web_aligned", MOORE_WEB, "mos-fra", ("mos", "fra"), "moore", "french", 0.35, moore_web_aligned),
    SourceSpec(
        "nllb_fra_mos_pivot", NLLB_BURKINA, "fra-mos", ("mos", "fra"), "trg_text", "src_text", 0.25,
        nllb_filter("fra_Latn", "mos_Latn"), pivot_column="pivot_text",
    ),
    SourceSpec("moore_web_lexicon", MOORE_WEB, "mos-fra", ("mos", "fra"), "moore", "french", 0.10, moore_web_lexicon),
    SourceSpec(
        "nllb_eng_mos", NLLB_BURKINA, "eng-mos", ("mos", "eng"), "trg_text", "src_text", 1.0,
        nllb_filter("eng_Latn", "mos_Latn"),
    ),
    SourceSpec(
        "faso_dyu_fra_sentences", FASO_WEB, "dyu-fra", ("dyu", "fra"), "source_text", "target_text", 0.45,
        faso_sentences,
    ),
    SourceSpec(
        "nllb_fra_dyu", NLLB_BURKINA, "fra-dyu", ("dyu", "fra"), "trg_text", "src_text", 0.45,
        nllb_filter("fra_Latn", "dyu_Latn"),
    ),
    SourceSpec(
        "faso_dyu_fra_lexicon", FASO_WEB, "dyu-fra", ("dyu", "fra"), "source_text", "target_text", 0.10,
        faso_lexicon,
    ),
    SourceSpec(
        "nllb_eng_dyu", NLLB_BURKINA, "eng-dyu", ("dyu", "eng"), "trg_text", "src_text", 0.85,
        nllb_filter("eng_Latn", "dyu_Latn"),
    ),
    SourceSpec(
        "faso_dyu_eng_examples", FASO_WEB, "dyu-eng", ("dyu", "eng"), "source_text", "target_text", 0.05,
        faso_sentences,
    ),
    SourceSpec(
        "faso_dyu_eng_lexicon", FASO_WEB, "dyu-eng", ("dyu", "eng"), "source_text", "target_text", 0.10,
        faso_lexicon,
    ),
    SourceSpec(
        "faso_dyu_mos_pivot", FASO_WEB, "dyu-mos", ("dyu", "mos"), "source_text", "target_text", 0.75,
        faso_pivot, pivot_column="pivot_text",
    ),
    SourceSpec(
        "nllb_mos_dyu_pivot", NLLB_BURKINA, "mos-dyu", ("dyu", "mos"), "trg_text", "src_text", 0.25,
        nllb_filter("mos_Latn", "dyu_Latn"), pivot_column="pivot_text",
    ),
]


@dataclass(frozen=True)
class PairExample:
    pair: tuple[str, str]
    local_text: str
    other_text: str


def add_common_instruction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--instruction-version",
        choices=("v1", "v2"),
        default="v2",
        help="Use v1 system-message instruction style or v2 user-message instruction style.",
    )
    parser.add_argument(
        "--instruction",
        help=(
            "Optional override for the instruction template. Supports "
            "{source_lang} and {target_lang}."
        ),
    )


def add_mix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-examples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--into-local-ratio",
        type=float,
        default=0.6,
        help="Share of fra/eng pairs translated into mos/dyu. dyu-mos is always split 50/50.",
    )
    parser.add_argument(
        "--pair-weight",
        action="append",
        default=[],
        metavar="PAIR=WEIGHT",
        help="Override a pair weight, e.g. --pair-weight dyu-mos=0.25. Weights are renormalized.",
    )
    parser.add_argument(
        "--flores-dataset",
        default="openlanguagedata/flores_plus",
        help="FLORES+ sentences (dev and devtest) are removed from training data.",
    )
    add_common_instruction_args(parser)


def add_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path("eval"))
    parser.add_argument("--flores-dataset", default="openlanguagedata/flores_plus")
    parser.add_argument("--flores-split", default="dev")
    parser.add_argument(
        "--combined-max-per-direction",
        type=int,
        default=100,
        help="Rows per direction in eval/validation.jsonl, the file used during training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    add_common_instruction_args(parser)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build bidirectional mos/dyu/fra/eng ShareGPT-style JSONL from Hugging Face datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mix_parser = subparsers.add_parser("mix", help="Build weighted training JSONL.")
    add_mix_args(mix_parser)

    validation_parser = subparsers.add_parser("validation", help="Build validation JSONL.")
    add_validation_args(validation_parser)

    return parser.parse_args()


def read_hf_dataset(name: str, split: str = "train", config: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if config:
        dataset = load_dataset(name, config, split=split)
    else:
        dataset = load_dataset(name, split=split)
    return [dict(row) for row in dataset]


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_text(text: str) -> str:
    """Matching key for dedupe and decontamination: case, punctuation and spacing are ignored."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


def parse_pair_weights(overrides: list[str]) -> dict[tuple[str, str], float]:
    weights = dict(PAIR_WEIGHTS)
    names = {pair_name(pair): pair for pair in weights}
    for override in overrides:
        name, _, value = override.partition("=")
        if name not in names or not value:
            raise ValueError(f"Invalid --pair-weight {override!r}. Pairs: {', '.join(names)}.")
        weights[names[name]] = float(value)
    return weights


def flores_rows(dataset: str, split: str, lang: str) -> list[dict[str, Any]]:
    return read_hf_dataset(dataset, split=split, config=FLORES_CODES[lang])


def flores_texts_by_id(dataset: str, split: str, lang: str) -> dict[Any, str]:
    rows = flores_rows(dataset, split, lang)
    return {row.get("id", index): row["text"].strip() for index, row in enumerate(rows)}


def held_out_texts(flores_dataset: str) -> set[str]:
    """Normalized sentences from every evaluation set, so they never reach training data."""
    texts = set()
    for split in ("validation", "test"):
        for row in read_hf_dataset(MOORE_WEB, split=split, config="mos-fra"):
            texts.update(normalize_text(row[column]) for column in ("moore", "french") if row[column])
    for split in ("dev", "devtest"):
        for lang in FLORES_CODES:
            texts.update(normalize_text(text) for text in flores_texts_by_id(flores_dataset, split, lang).values())
    texts.discard("")
    return texts


def load_source(
    spec: SourceSpec,
    rows: list[dict[str, Any]],
    held_out: set[str],
    seen: set[tuple[str, str, str]],
) -> tuple[list[PairExample], Counter]:
    stats = Counter()
    examples = []
    for row in rows:
        if not spec.keep(row):
            continue
        stats["kept_by_filter"] += 1
        local_text = clean_text(row.get(spec.local_column))
        other_text = clean_text(row.get(spec.other_column))
        if not local_text or not other_text:
            stats["empty"] += 1
            continue

        texts = [local_text, other_text]
        if spec.pivot_column and row.get(spec.pivot_column):
            texts.append(row[spec.pivot_column])
        if any(normalize_text(text) in held_out for text in texts):
            stats["held_out"] += 1
            continue

        key = (pair_name(spec.pair), normalize_text(local_text), normalize_text(other_text))
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        examples.append(PairExample(spec.pair, local_text, other_text))
    stats["usable"] = len(examples)
    return examples, stats


def load_sources(flores_dataset: str) -> dict[str, list[PairExample]]:
    held_out = held_out_texts(flores_dataset)
    print(f"Held-out sentences (moore-web validation/test, FLORES+ dev/devtest): {len(held_out)}")

    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: dict[tuple[str, str], set[tuple[str, str, str]]] = {pair: set() for pair in PAIR_WEIGHTS}
    pools = {}
    for spec in SOURCES:
        key = (spec.dataset, spec.config)
        if key not in cache:
            cache[key] = read_hf_dataset(spec.dataset, config=spec.config)
        pools[spec.name], stats = load_source(spec, cache[key], held_out, seen[spec.pair])
        print(
            f"{spec.name}: {stats['usable']} usable "
            f"(filter kept {stats['kept_by_filter']}, held-out {stats['held_out']}, "
            f"duplicates {stats['duplicate']}, empty {stats['empty']})"
        )
    return pools


def allocation(weights: dict[Any, float], total: int) -> dict[Any, int]:
    active_weights = {key: weight for key, weight in weights.items() if weight > 0}
    if not active_weights:
        raise ValueError("No active buckets with a non-zero weight.")

    weight_sum = sum(active_weights.values())
    raw_counts = {key: (weight / weight_sum) * total for key, weight in active_weights.items()}
    counts = {key: int(value) for key, value in raw_counts.items()}
    remainder = total - sum(counts.values())

    order = sorted(raw_counts, key=lambda key: raw_counts[key] - counts[key], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def sample_pairs(rng: random.Random, pairs: list[PairExample], count: int) -> list[PairExample]:
    if count <= len(pairs):
        return rng.sample(pairs, count)
    sampled = list(pairs)
    sampled.extend(rng.choice(pairs) for _ in range(count - len(pairs)))
    rng.shuffle(sampled)
    return sampled


def orient(rng: random.Random, example: PairExample, into_local_ratio: float) -> tuple[str, str, str, str]:
    """Return (source_lang, target_lang, source_text, target_text) for one sampled direction."""
    local_lang, other_lang = example.pair
    ratio = 0.5 if other_lang in LOCAL_LANGS else into_local_ratio
    if rng.random() < ratio:
        return other_lang, local_lang, example.other_text, example.local_text
    return local_lang, other_lang, example.local_text, example.other_text


def to_sharegpt(
    source: str,
    target: str,
    source_lang: str,
    target_lang: str,
    instruction: str | None,
    instruction_version: str,
) -> dict[str, Any]:
    messages = []
    if instruction:
        rendered_instruction = instruction.format(source_lang=source_lang, target_lang=target_lang)
        if instruction_version == "v1":
            messages.append({"role": "system", "content": rendered_instruction})
            prompt = source
        else:
            prompt = f"{rendered_instruction}\n\n{source}"
    else:
        prompt = source

    messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    )
    return {"messages": messages}


def instruction_for_version(version: str, override: str | None) -> str:
    if override is not None:
        return override
    return INSTRUCTION_V1 if version == "v1" else INSTRUCTION_V2


def translation_row(
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str,
    instruction: str,
    instruction_version: str,
) -> dict[str, Any]:
    return to_sharegpt(
        source_text,
        target_text,
        LANG_NAMES[source_lang],
        LANG_NAMES[target_lang],
        instruction,
        instruction_version,
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def run_mix(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    instruction = instruction_for_version(args.instruction_version, args.instruction)
    pair_counts = allocation(parse_pair_weights(args.pair_weight), args.total_examples)
    pools = load_sources(args.flores_dataset)

    examples = []
    directions = Counter()
    print("\nSampled rows:")
    for pair, pair_count in pair_counts.items():
        specs = [spec for spec in SOURCES if spec.pair == pair and pools[spec.name]]
        if not specs:
            raise ValueError(f"Pair {pair_name(pair)} has no usable examples.")
        source_counts = allocation({spec.name: spec.weight for spec in specs}, pair_count)
        print(f"{pair_name(pair)}: {pair_count}")
        for name, count in source_counts.items():
            pool = pools[name]
            print(f"  {name}: {count} from {len(pool)} ({count / len(pool):.2f}x)")
            for example in sample_pairs(rng, pool, count):
                source_lang, target_lang, source_text, target_text = orient(rng, example, args.into_local_ratio)
                directions[f"{source_lang}->{target_lang}"] += 1
                examples.append(
                    translation_row(
                        source_lang, target_lang, source_text, target_text, instruction, args.instruction_version
                    )
                )

    rng.shuffle(examples)
    write_jsonl(args.output, examples)
    print("\nDirections:")
    for direction, count in sorted(directions.items()):
        print(f"  {direction}: {count}")
    print(f"Wrote {len(examples)} examples to {args.output}")


def validation_directions() -> list[tuple[str, str]]:
    return [direction for pair in PAIR_WEIGHTS for direction in (pair, pair[::-1])]


def build_flores_validation(
    args: argparse.Namespace, instruction: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    texts = {lang: flores_texts_by_id(args.flores_dataset, args.flores_split, lang) for lang in FLORES_CODES}
    rows = {}
    for source_lang, target_lang in validation_directions():
        source_texts, target_texts = texts[source_lang], texts[target_lang]
        shared_ids = [row_id for row_id in source_texts if row_id in target_texts]
        rows[(source_lang, target_lang)] = [
            translation_row(
                source_lang,
                target_lang,
                source_texts[row_id],
                target_texts[row_id],
                instruction,
                args.instruction_version,
            )
            for row_id in shared_ids
        ]
    return rows


def build_moore_web_validation(
    args: argparse.Namespace, instruction: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    dataset = read_hf_dataset(MOORE_WEB, split="validation", config="mos-fra")
    rows = {("fra", "mos"): [], ("mos", "fra"): []}
    for row in dataset:
        moore, french = clean_text(row["moore"]), clean_text(row["french"])
        if not moore or not french:
            continue
        rows[("fra", "mos")].append(translation_row("fra", "mos", french, moore, instruction, args.instruction_version))
        rows[("mos", "fra")].append(translation_row("mos", "fra", moore, french, instruction, args.instruction_version))
    return rows


def run_validation(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    instruction = instruction_for_version(args.instruction_version, args.instruction)

    sets = {
        f"flores_{args.flores_split}": build_flores_validation(args, instruction),
        "moore_web_validation": build_moore_web_validation(args, instruction),
    }
    combined = []
    for set_name, directions in sets.items():
        for (source_lang, target_lang), rows in directions.items():
            path = args.output_dir / set_name / f"{source_lang}_{target_lang}.jsonl"
            count = write_jsonl(path, rows)
            print(f"Wrote {count} examples to {path}")
            combined.extend(rng.sample(rows, min(args.combined_max_per_direction, len(rows))))

    rng.shuffle(combined)
    path = args.output_dir / "validation.jsonl"
    count = write_jsonl(path, combined)
    print(f"Wrote {count} examples to {path}")


def main() -> None:
    args = parse_args()
    if args.command == "validation":
        run_validation(args)
    else:
        run_mix(args)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
