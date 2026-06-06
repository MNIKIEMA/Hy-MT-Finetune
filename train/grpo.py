import logging
import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import transformers
import wandb
from dotenv import load_dotenv
from peft import LoraConfig
from sacrebleu.metrics import CHRF
from trl import GRPOConfig, GRPOTrainer

load_dotenv()

# ---------------------------------------------------------------------------
# MetricX-24 reward helper
# ---------------------------------------------------------------------------


class MetricXRewardModel:
    """
    Wraps google/metricx-24-hybrid-large-v2p6.

    MetricX is a *reference-based* quality-estimation model that produces
    segment-level scores in [0, 25] where **lower is better** (it predicts
    MQM error scores).  We negate and normalise to [-1, 0] so it behaves
    like a reward where higher is better.
    """

    MODEL_ID = "google/metricx-24-hybrid-large-v2p6"
    # MetricX uses a T5-style tokenizer; the checkpoint is hosted under the
    # google-metricx organisation on HF.

    def __init__(self, device: str | None = None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading MetricX-24 on {self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
    ) -> list[float]:
        """
        Returns a reward ∈ [-1, 0] per segment.
        MetricX input format (hybrid):
          "source: <src> hypothesis: <hyp> reference: <ref>"
        """
        inputs_text = [
            f"source: {src} hypothesis: {hyp} reference: {ref}"
            for src, hyp, ref in zip(sources, hypotheses, references, strict=True)
        ]
        enc = self.tokenizer(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        # MetricX outputs a scalar via the decoder; the model is a regression
        # T5 where the score is the first generated token's value.
        outputs = self.model.generate(
            **enc,
            max_new_tokens=1,
        )
        # Decode to float
        raw_scores = []
        for token_ids in outputs:
            try:
                val = float(self.tokenizer.decode(token_ids, skip_special_tokens=True).strip())
            except ValueError:
                val = 25.0  # worst case
            raw_scores.append(val)

        # Negate and normalise from [0, 25] → [−1, 0]
        rewards = [-s / 25.0 for s in raw_scores]
        return rewards


class SSACometRewardModel:
    """
    Wraps McGill-NLP/ssa-comet-mtl via the `comet` library.

    SSA-COMET is a reference-based MT quality-estimation model that produces
    segment-level scores in [0, 1] where **higher is better**.  We keep the
    scores as-is since they already behave like a reward.
    """

    MODEL_ID = "McGill-NLP/ssa-comet-mtl"

    def __init__(self, device: str | None = None):
        from comet import download_model, load_from_checkpoint

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gpus = 1 if "cuda" in self.device else 0

        logging.info(f"Loading SSA-COMET on {self.device} …")
        model_path = download_model(self.MODEL_ID)
        self.model = load_from_checkpoint(model_path)

    def score(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
    ) -> list[float]:
        """
        Returns a reward ∈ [0, 1] per segment (higher is better).
        SSA-COMET input format:
          [{"src": ..., "mt": ..., "ref": ...}, ...]
        """
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references, strict=True)
        ]

        model_output = self.model.predict(
            data,
            batch_size=8,
            gpus=self.gpus,
        )

        # model_output.scores is a list of per-segment floats
        return model_output.scores


# ---------------------------------------------------------------------------
# Reward functions for GRPOTrainer
# ---------------------------------------------------------------------------

_chrf_metric = CHRF()
_metricx: MetricXRewardModel | None = None  # lazy init


def _get_metricx() -> MetricXRewardModel:
    global _metricx
    if _metricx is None:
        _metricx = MetricXRewardModel()
    return _metricx


def reward_chrf(
    completions: list[str],
    references: list[str],
    **kwargs,
) -> list[float]:
    """
    chrF++ reward in [0, 1].

    GRPOTrainer passes `completions` (model outputs) and any extra fields
    from the dataset row via **kwargs.  We expect `references` to be the
    Moore ground-truth string.
    """
    rewards = []
    for hyp, ref in zip(completions, references, strict=True):
        score = _chrf_metric.sentence_score(hyp, [ref]).score / 100.0
        rewards.append(score)
    return rewards


def reward_metricx(
    completions: list[str],
    references: list[str],
    prompts: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """
    MetricX-24 reward ∈ [-1, 0].

    `prompts` here is the French source sentence extracted from the
    GRPOTrainer prompt list (see dataset below).
    """
    sources = prompts if prompts is not None else [""] * len(completions)
    return _get_metricx().score(sources, completions, references)


def _ngram_counts(tokens: list[str], n: int) -> dict:
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts: dict = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def reward_no_repetition(
    completions: list[str],
    ngram_sizes: tuple[int, ...] = (3, 4),
    **kwargs,
) -> list[float]:
    """
    Penalises repeated n-grams in the completion.

    For each n in `ngram_sizes` we compute the fraction of n-grams that are
    duplicates (i.e. appear more than once).  The penalty is the *maximum*
    duplicate fraction across all n sizes, so even a single heavily repeated
    4-gram is caught.

    Returns a reward ∈ [-1, 0]:
      0.0  → no repetition at all
      -1.0 → every n-gram is a repeat (degenerate looping output)

    Using [-1, 0] keeps it consistent with MetricX and makes it easy to
    weight alongside quality signals without inflating the scale.
    """
    rewards = []
    for text in completions:
        tokens = text.split()  # word-level; fast and language-agnostic
        if len(tokens) < max(ngram_sizes):
            rewards.append(0.0)  # too short to judge
            continue

        max_dup_fraction = 0.0
        for n in ngram_sizes:
            counts = _ngram_counts(tokens, n)
            total = sum(counts.values())
            duplicated = sum(c - 1 for c in counts.values() if c > 1)
            dup_fraction = duplicated / total if total > 0 else 0.0
            max_dup_fraction = max(max_dup_fraction, dup_fraction)

        rewards.append(-max_dup_fraction)  # ∈ [-1, 0]
    return rewards


def reward_combined(
    completions: list[str],
    references: list[str],
    prompts: list[str] | None = None,
    chrf_weight: float = 0.4,
    metricx_weight: float = 0.4,
    repetition_weight: float = 0.2,
    ngram_sizes: tuple[int, ...] = (3, 4),
    **kwargs,
) -> list[float]:
    """
    Weighted combination of three signals, all normalised to [0, 1]:

      combined = chrf_weight    * chrF
               + metricx_weight * (1 + metricx)   # shift [-1,0] → [0,1]
               + repetition_weight * (1 + rep)     # shift [-1,0] → [0,1]

    Default weights sum to 1.0.  Adjust via CLI flags.
    """
    chrf_r = reward_chrf(completions, references)
    mx_r = reward_metricx(completions, references, prompts=prompts)
    rep_r = reward_no_repetition(completions, ngram_sizes=ngram_sizes)

    mx_norm = [r + 1.0 for r in mx_r]  # [-1,0] → [0,1]
    rep_norm = [r + 1.0 for r in rep_r]  # [-1,0] → [0,1]

    return [
        chrf_weight * c + metricx_weight * m + repetition_weight * r
        for c, m, r in zip(chrf_r, mx_norm, rep_norm, strict=True)
    ]


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------


def print_args(args, name="arguments"):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"------------------------ {name} ------------------------", flush=True)
        for arg in sorted(vars(args)):
            dots = "." * (48 - len(arg))
            print(f"  {arg} {dots} {getattr(args, arg)}", flush=True)
        print(f"-------------------- end of {name} ---------------------", flush=True)


@dataclass
class ModelArguments:
    use_flash_attn: bool = field(default=False)
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    train_attention_params_only: bool = field(default=False)


@dataclass
class DataArguments:
    model_size: Literal["0.5B", "1.8B", "4B", "7B"] = field(default="1.8B")
    max_seq_length: int = field(default=512)
    use_dummy_data: bool = field(default=False)
    chrf_weight: float = field(
        default=0.4,
        metadata={"help": "Weight for chrF in the combined reward."},
    )
    metricx_weight: float = field(
        default=0.4,
        metadata={"help": "Weight for MetricX in the combined reward."},
    )
    repetition_weight: float = field(
        default=0.2,
        metadata={"help": "Weight for the repetition penalty in the combined reward."},
    )
    repetition_ngram_sizes: str = field(
        default="3,4",
        metadata={"help": "Comma-separated n-gram sizes for repetition detection (e.g. '3,4')."},
    )
    reward_fn: Literal["chrf", "metricx", "repetition", "combined"] = field(
        default="combined",
        metadata={"help": "Which reward function to use."},
    )


@dataclass
class MyGRPOConfig(GRPOConfig):
    """Thin wrapper so we can add extra CLI fields alongside GRPOConfig."""

    tokenizer_name_or_path: str | None = field(default=None)
    model_name_or_path: str | None = field(default=None)
    hub_model_id: str | None = field(default=None)
    hub_private_repo: bool = field(default=False)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "Translate the following French text to Moore."


def build_prompt(french: str) -> str:
    """Return a plain chat-style prompt string (formatted later by tokenizer)."""
    return french  # raw source; chat template applied inside GRPOTrainer


def make_grpo_dataset(tokenizer, data_args: DataArguments):
    """
    Returns HuggingFace Dataset objects ready for GRPOTrainer.

    Each example must have:
      - "prompt"     : list[dict] (chat messages, *without* the assistant turn)
      - "reference"  : str        (the ground-truth Moore translation)
    """
    from datasets import Dataset as HFDataset

    if data_args.use_dummy_data:
        dummy = [
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Bonjour le monde."},
                ],
                "reference": "Laafi, a to dũni.",
                "source": "Bonjour le monde.",
            }
        ] * 200
        train_ds = HFDataset.from_list(dummy[:180])
        eval_ds = HFDataset.from_list(dummy[180:])
        return train_ds, eval_ds

    train_ds, val_ds, _ = load_all_splits("huggingface", "burkimbia/fr_mos_data_cleaned")

    def _format(examples):
        return {
            "prompt": [
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": src}]
                for src in examples["french"]
            ],
            "reference": examples["moore"],
            "source": examples["french"],
        }

    return (
        train_ds.map(_format, batched=True, remove_columns=train_ds.column_names),
        val_ds.map(_format, batched=True, remove_columns=val_ds.column_names),
    )


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, MyGRPOConfig))
    model_args, data_args, grpo_config = parser.parse_args_into_dataclasses()

    print_args(model_args, "model arguments")
    print_args(data_args, "data arguments")
    print_args(grpo_config, "GRPO config")

    # W&B -----------------------------------------------------------------
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "hy-mt-grpo"),
            name=grpo_config.run_name,
            config={
                "model_name_or_path": grpo_config.model_name_or_path,
                "use_lora": model_args.use_lora,
                "lora_rank": model_args.lora_rank,
                "reward_fn": data_args.reward_fn,
                "chrf_weight": data_args.chrf_weight,
                "learning_rate": grpo_config.learning_rate,
                "num_train_epochs": grpo_config.num_train_epochs,
            },
        )

    # Tokenizer ------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        grpo_config.tokenizer_name_or_path or grpo_config.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model ----------------------------------------------------------------
    if not grpo_config.model_name_or_path or not os.path.exists(grpo_config.model_name_or_path):
        raise FileNotFoundError(
            f"Model path '{grpo_config.model_name_or_path}' is invalid or does not exist."
        )

    init_kwargs: dict = {"trust_remote_code": True}
    if model_args.use_flash_attn:
        init_kwargs["attn_implementation"] = "flash_attention_2"
    if grpo_config.bf16:
        init_kwargs["torch_dtype"] = torch.bfloat16
    elif grpo_config.fp16:
        init_kwargs["torch_dtype"] = torch.float16

    model = transformers.AutoModelForCausalLM.from_pretrained(grpo_config.model_name_or_path, **init_kwargs)

    if model_args.train_attention_params_only:
        for name, param in model.named_parameters():
            if "self_attn" not in name:
                param.requires_grad = False

    peft_config = None
    if model_args.use_lora:
        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Datasets -------------------------------------------------------------
    train_dataset, eval_dataset = make_grpo_dataset(tokenizer, data_args)

    # Reward function selection --------------------------------------------
    ngram_sizes = tuple(int(x) for x in data_args.repetition_ngram_sizes.split(","))

    if data_args.reward_fn == "chrf":
        reward_funcs = [reward_chrf]

    elif data_args.reward_fn == "metricx":
        reward_funcs = [reward_metricx]

    elif data_args.reward_fn == "repetition":

        def _rep_only(completions, **kw):
            return reward_no_repetition(completions, ngram_sizes=ngram_sizes)

        reward_funcs = [_rep_only]

    else:  # "combined"

        def _combined(completions, references, prompts=None, **kw):
            return reward_combined(
                completions,
                references,
                prompts=prompts,
                chrf_weight=data_args.chrf_weight,
                metricx_weight=data_args.metricx_weight,
                repetition_weight=data_args.repetition_weight,
                ngram_sizes=ngram_sizes,
            )

        # Tip: pass [reward_chrf, reward_metricx, reward_no_repetition]
        # for per-signal W&B logging at the cost of separate reward scales.
        reward_funcs = [_combined]

    # GRPO trainer ---------------------------------------------------------
    # GRPOTrainer forwards every column that isn't "prompt" as a keyword
    # argument to the reward function — so `reference` and `source` are
    # automatically available inside reward_chrf / reward_metricx.
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)

    # Optionally push to hub
    if grpo_config.push_to_hub and grpo_config.hub_model_id:
        trainer.push_to_hub(grpo_config.hub_model_id, private=grpo_config.hub_private_repo)


if __name__ == "__main__":
    train()
