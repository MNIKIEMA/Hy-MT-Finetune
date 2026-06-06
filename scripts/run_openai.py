import argparse
import asyncio
import os

from datasets import DownloadConfig, load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


async def translate_example(client, semaphore, model_name, text):
    async with semaphore:
        response = await client.chat.completions.create(
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


async def main(args):
    client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8021/v1")
    semaphore = asyncio.Semaphore(args.concurrency)

    ds = load_dataset(
        "burkimbia/mt-benchmark-public",
        download_config=DownloadConfig(token=os.environ["HF_TOKEN"]),
        split="train",
    )
    print(f"Loaded {len(ds)} examples")

    tasks = [translate_example(client, semaphore, args.model_path, example["source_text"]) for example in ds]

    results = await tqdm_asyncio.gather(*tasks, desc="Translating")

    translations = []
    for r in results:
        if isinstance(r, Exception):
            print(f"\nError: {r}")
            translations.append("")
        else:
            translations.append(r)

    ds = ds.add_column("translation", translations)

    output_path = args.output_path
    if not output_path.endswith(".csv"):
        output_path += ".csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ds.to_csv(output_path)
    print(f"\nDataset with translations saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async translation inference")
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
        "--concurrency",
        type=int,
        default=32,
        help="Max number of concurrent requests to vLLM",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
