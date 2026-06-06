import argparse
import os

from datasets import DownloadConfig, load_dataset
from openai import OpenAI
from tqdm import tqdm


def translate_example(client, model_name, text):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Translate the following French text to Moore.",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        temperature=0.0,
        top_p=0.9,
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1.05,
            "stop_token_ids": [127960],
        },
    )

    return response.choices[0].message.content.strip()


def translate_batch(client, model_name, texts):
    """
    Sends a batch of texts in a single API call.
    Assumes your server supports batched chat completions.
    """

    # Build batch messages
    messages = []
    for text in texts:
        messages.append(
            [
                {
                    "role": "system",
                    "content": "Translate the following French text to Moore.",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ]
        )

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
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8021/v1")

    # Load dataset
    ds = load_dataset(
        "burkimbia/mt-benchmark-public",
        download_config=DownloadConfig(token=os.environ["HF_TOKEN"]),
        split="train",
    )

    print(f"Loaded {len(ds)} examples")

    translations = []

    for example in tqdm(ds):
        french_text = example["source_text"]

        try:
            translation = translate_example(client, args.model_path, french_text)
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
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    main(args)
