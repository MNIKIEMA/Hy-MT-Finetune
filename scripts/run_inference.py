import argparse
import asyncio
import json
import os
from pathlib import Path

from datasets import DownloadConfig, load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

INSTRUCTION_V2 = (
    "Translate the following {source_lang} text into {target_lang}, "
    "output only the translation result without additional explanation:"
)
INSTRUCTION_V1 = "Translate the following segment into {target_lang}, without additional explanation."


def instruction_for_version(version, override):
    if override is not None:
        return override
    return INSTRUCTION_V1 if version == "v1" else INSTRUCTION_V2


def value_from_row(row, field, default):
    return row[field] if field else default


def ensure_mapped_fields_exist(dataset, fields):
    available = set(dataset.column_names)
    missing = {field for field in fields if field and field not in available}
    if missing:
        raise ValueError(
            f"Mapped column(s) not found: {', '.join(sorted(missing))}. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )


def build_messages(text, source_lang, target_lang, instruction, instruction_version):
    messages = []
    if instruction:
        rendered_instruction = instruction.format(source_lang=source_lang, target_lang=target_lang)
        if instruction_version == "v1":
            messages.append({"role": "system", "content": rendered_instruction})
            prompt = text
        else:
            prompt = f"{rendered_instruction}\n\n{text}"
    else:
        prompt = text

    messages.append({"role": "user", "content": prompt})
    return messages


async def translate_example(
    client, semaphore, model_name, text, source_lang, target_lang, instruction, instruction_version
):
    async with semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=build_messages(text, source_lang, target_lang, instruction, instruction_version),
            temperature=0.0,
            top_p=0.9,
            extra_body={
                "top_k": 20,
                "repetition_penalty": 1.05,
                "stop_token_ids": [127960],
            },
        )

    return response.choices[0].message.content.strip()


async def translate_safely(*args):
    try:
        return await translate_example(*args)
    except Exception as exc:
        return exc


async def main(args):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    instruction = instruction_for_version(args.instruction_version, args.instruction)

    # Load dataset
    ds = load_dataset(
        args.dataset,
        download_config=DownloadConfig(token=os.environ.get("HF_TOKEN")),
        split=args.split,
    )

    print(f"Loaded {len(ds)} examples")
    print(
        "Prompt layout: "
        f"{args.instruction_version} ({args.source_lang} -> {args.target_lang})"
    )
    ensure_mapped_fields_exist(
        ds,
        [
            args.source_field,
            args.source_lang_field,
            args.target_lang_field,
            args.reference_field,
            args.id_field,
        ],
    )

    tasks = [
        translate_safely(
            client,
            semaphore,
            args.model_path,
            example[args.source_field],
            value_from_row(example, args.source_lang_field, args.source_lang),
            value_from_row(example, args.target_lang_field, args.target_lang),
            instruction,
            args.instruction_version,
        )
        for example in ds
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Translating")
    translations = []
    for result in results:
        if isinstance(result, Exception):
            print(f"\nError processing example: {result}")
            translations.append("")
        else:
            translations.append(result)

    suffix = ".jsonl" if args.output_format == "bouquet" else ".csv"
    output_path = args.output_path if args.output_path.endswith(suffix) else args.output_path + suffix
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if args.output_format == "bouquet":
        with Path(output_path).open("w", encoding="utf-8") as output_file:
            for example, translation in zip(ds, translations, strict=True):
                row = {
                    "src_lang": value_from_row(example, args.source_lang_field, args.source_lang),
                    "tgt_lang": value_from_row(example, args.target_lang_field, args.target_lang),
                    "src_text": example[args.source_field],
                }
                if args.reference_field:
                    row["ref_text"] = example[args.reference_field]
                row["mt_text"] = translation
                if args.id_field:
                    row["id"] = example[args.id_field]
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        ds = ds.add_column("translation", translations)
        ds.to_csv(output_path)

    print(f"\nDataset with translations saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch translation inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="hy-mt",
        help="Model name served on the OpenAI-compatible server",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./translated_dataset",
        help="Where to save the updated dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="burkimbia/mt-benchmark-public",
        help="Hugging Face dataset id to translate",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--source-field", type=str, default="source_text", help="Dataset column containing source text.")
    parser.add_argument(
        "--source-lang-field",
        help="Optional dataset column for the source language; otherwise --source-lang is used.",
    )
    parser.add_argument(
        "--target-lang-field",
        help="Optional dataset column for the target language; otherwise --target-lang is used.",
    )
    parser.add_argument(
        "--reference-field",
        help="Optional dataset column to write as ref_text in Bouquet JSONL output.",
    )
    parser.add_argument(
        "--id-field",
        help="Optional dataset column to preserve as id in Bouquet JSONL output.",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="French",
        help="Source language label used in the v2 instruction.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="Moore",
        help="Target language label used in the instruction.",
    )
    parser.add_argument(
        "--instruction-version",
        choices=("v1", "v2"),
        default="v2",
        help="Use Hy-MT 1.x system-message layout or Hy-MT2 user-message layout.",
    )
    parser.add_argument(
        "--instruction",
        help=(
            "Optional instruction template override. Supports "
            "{source_lang} and {target_lang}."
        ),
    )
    parser.add_argument("--base-url", default="http://localhost:8021/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--output-format",
        choices=("csv", "bouquet"),
        default="csv",
        help="Write the existing CSV output or Bouquet-compatible JSONL.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent requests to the inference server.",
    )

    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    asyncio.run(main(args))
