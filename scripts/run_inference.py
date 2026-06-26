import argparse
import os

from datasets import DownloadConfig, load_dataset
from openai import OpenAI
from tqdm import tqdm


INSTRUCTION_V2 = (
    "Translate the following {source_lang} text into {target_lang}, "
    "output only the translation result without additional explanation:"
)
INSTRUCTION_V1 = "Translate the following segment into {target_lang}, without additional explanation."


def instruction_for_version(version, override):
    if override is not None:
        return override
    return INSTRUCTION_V1 if version == "v1" else INSTRUCTION_V2


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


def translate_example(client, model_name, text, source_lang, target_lang, instruction, instruction_version):
    response = client.chat.completions.create(
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


def translate_batch(client, model_name, texts, source_lang, target_lang, instruction, instruction_version):
    """
    Sends a batch of texts in a single API call.
    Assumes your server supports batched chat completions.
    """

    # Build batch messages
    messages = []
    for text in texts:
        messages.append(build_messages(text, source_lang, target_lang, instruction, instruction_version))

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,  # <-- batched messages
        temperature=0.1,
        top_p=0.9,
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1.05,
            "stop_token_ids": [127960],
        },
    )

    # Extract outputs
    translations = []
    for choice in response.choices:
        translations.append(choice.message.content.strip())

    return translations


def main(args):
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
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

    translations = []

    for example in tqdm(ds):
        source_text = example[args.source_field]

        try:
            translation = translate_example(
                client,
                args.model_path,
                source_text,
                args.source_lang,
                args.target_lang,
                instruction,
                args.instruction_version,
            )
        except Exception as e:
            print(f"\nError processing example: {e}")
            translation = ""

        translations.append(translation)

    ds = ds.add_column("translation", translations)

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    if not args.output_path.endswith(".csv"):
        args.output_path += ".csv"

    ds.to_csv(args.output_path)

    print(f"\nDataset with translations saved to {args.output_path}")


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
    parser.add_argument("--source-field", type=str, default="source_text")
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
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    main(args)
