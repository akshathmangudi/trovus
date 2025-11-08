from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset
from huggingface_hub.utils import HfHubHTTPError
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

try:
    from trl import SFTTrainer
except ImportError as exc:  # pragma: no cover - handled during runtime
    raise RuntimeError(
        "The `trl` package is required for supervised fine-tuning. "
        "Install it with `pip install trl`."
    ) from exc

from peft import LoraConfig

from trovus.evaluate.utils import (
    DATASET_REGISTRY,
    DatasetSpec,
    dump_json,
    ensure_model_on_disk,
    expand_path,
    normalize_dataset_key,
    resolve_dataset_spec,
    timestamped_run_dir,
)


DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

console = Console()


@dataclass
class SFTMethodConfig:
    dataset: str = "gsm8k"
    dataset_split: Optional[str] = None
    eval_split: Optional[str] = None
    output_dir: str = "./runs"
    cache_dir: Optional[str] = None
    epochs: float = 1.0
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: Optional[int] = None
    max_steps: Optional[int] = None
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: DEFAULT_LORA_TARGET_MODULES.copy())
    use_4bit: bool = False
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42
    push_to_hub: bool = False
    report_to: Optional[str] = "none"
    force_download_model: bool = False
    merge_lora_after_training: bool = False
    custom_prompts: Optional[str] = None


@dataclass
class SFTExecutionResult:
    run_dir: str
    model_path: str
    tokenizer_path: str
    metrics: Dict
    compute_summary: Dict


def _load_registered_dataset(spec: DatasetSpec, dataset_split: Optional[str], eval_split: Optional[str]) -> DatasetDict:
    split_train = dataset_split or spec.train_split
    split_eval = eval_split or spec.eval_split

    load_kwargs: Dict = {}
    if spec.config_name:
        load_kwargs["name"] = spec.config_name

    try:
        dataset = load_dataset(spec.hub_id, **load_kwargs)
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Unable to download dataset '{spec.hub_id}'. Ensure you have network access and "
            f"the dataset is public."
        ) from exc

    if split_train not in dataset:
        raise KeyError(f"Split '{split_train}' not available in dataset '{spec.hub_id}'.")

    if split_eval and split_eval not in dataset:
        console.print(
            f"[yellow]Warning:[/yellow] requested eval split '{split_eval}' not found. "
            "Skipping evaluation dataset."
        )
        split_eval = None

    if split_eval:
        return DatasetDict({"train": dataset[split_train], "eval": dataset[split_eval]})
    return DatasetDict({"train": dataset[split_train]})


def _format_example(dataset_name: str, example: Dict) -> str:
    key = normalize_dataset_key(dataset_name)
    if key == "gsm8k":
        question = example.get("question", "").strip()
        answer = example.get("answer", "").strip()
        return f"Question:\n{question}\n\nAnswer:\n{answer}"
    if key == "math":
        problem = example.get("problem", "").strip()
        solution = example.get("solution", "").strip()
        return f"Problem:\n{problem}\n\nSolution:\n{solution}"
    if key == "arc_challenge":
        question = example.get("question", "").strip()
        options = example.get("choices", {}).get("text", [])
        labels = example.get("choices", {}).get("label", [])
        formatted_options = "\n".join(f"{label}. {text}" for label, text in zip(labels, options))
        answer = example.get("answerKey", "").strip()
        return (
            f"Question:\n{question}\n\nOptions:\n{formatted_options}\n\n"
            f"Answer:\n{answer}"
        )
    raise KeyError(f"Formatting for dataset '{dataset_name}' is not implemented yet.")


def _prepare_sft_datasets(
    dataset_name: str,
    config: SFTMethodConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[Dict[str, int], DatasetDict]:
    spec = resolve_dataset_spec(dataset_name)
    dataset = _load_registered_dataset(spec, config.dataset_split, config.eval_split)

    def map_fn(example: Dict) -> Dict:
        text = _format_example(dataset_name, example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=config.max_seq_length,
            add_special_tokens=True,
        )
        return {
            "text": text,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "num_tokens": len(tokenized["input_ids"]),
        }

    dataset = dataset.map(map_fn, remove_columns=[col for col in dataset["train"].column_names if col not in {"text"}])

    train_tokens = sum(dataset["train"]["num_tokens"]) if "num_tokens" in dataset["train"].features else 0
    summary: Dict[str, int] = {
        "train_examples": len(dataset["train"]),
        "train_tokens": int(train_tokens),
    }

    if "eval" in dataset:
        summary["eval_examples"] = len(dataset["eval"])
        if "num_tokens" in dataset["eval"].features:
            summary["eval_tokens"] = int(sum(dataset["eval"]["num_tokens"]))

    dataset = dataset.remove_columns(
        [col for col in dataset["train"].column_names if col not in {"text", "input_ids", "attention_mask", "num_tokens"}]
    )

    return summary, dataset


def _build_lora_config(config: SFTMethodConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _bitsandbytes_config(config: SFTMethodConfig) -> Optional[BitsAndBytesConfig]:
    if not config.use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16" if config.bf16 else "float16",
    )


def run_sft_pipeline(model_id: str, config: SFTMethodConfig) -> SFTExecutionResult:
    output_base = expand_path(config.output_dir) or Path("./runs")
    cache_dir = expand_path(config.cache_dir)
    models_dir = Path("./models")

    run_dir = timestamped_run_dir(output_base, model_id, "sft")

    model_path = ensure_model_on_disk(
        model_id=model_id,
        models_dir=models_dir,
        cache_dir=cache_dir,
        force_download=config.force_download_model,
    )

    console.print(f"[blue]Loading tokenizer from[/blue] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = _bitsandbytes_config(config)

    console.print(f"[blue]Loading base model from[/blue] {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.config.use_cache = False

    dataset_tokens_summary, dataset = _prepare_sft_datasets(config.dataset, config, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps or -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps or config.save_steps,
        evaluation_strategy="steps" if "eval" in dataset else "no",
        save_total_limit=3,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=None if not config.report_to or config.report_to == "none" else [config.report_to],
        seed=config.seed,
        push_to_hub=config.push_to_hub,
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        args=training_args,
        peft_config=_build_lora_config(config),
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
    )

    console.print("[green]Starting supervised fine-tuning run...[/green]")
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(run_dir)

    if config.merge_lora_after_training:
        console.print("[cyan]Merging LoRA adapters into the base model weights...[/cyan]")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(run_dir / "merged_model")

    metrics = train_result.metrics if train_result else {}
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    compute_summary = _estimate_compute_summary(config, dataset_tokens_summary, trainer)
    dump_json(metrics, Path(run_dir) / "training_metrics.json")
    dump_json(compute_summary, Path(run_dir) / "compute_summary.json")
    dump_json(
        {
            "model_id": model_id,
            "model_path": str(model_path),
            "config": asdict(config),
            "dataset_stats": dataset_tokens_summary,
        },
        Path(run_dir) / "run_config.json",
    )

    return SFTExecutionResult(
        run_dir=str(run_dir),
        model_path=str(model_path),
        tokenizer_path=str(run_dir),
        metrics=metrics,
        compute_summary=compute_summary,
    )


def _estimate_compute_summary(
    config: SFTMethodConfig,
    dataset_tokens_summary: Dict[str, int],
    trainer: SFTTrainer,
) -> Dict:
    tokens = dataset_tokens_summary.get("train_tokens", 0)
    steps = trainer.state.global_step if trainer.state else 0
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    epoch_count = trainer.state.epoch if trainer.state and trainer.state.epoch is not None else config.epochs
    approx_flops_per_token = _approximate_flops_per_token(trainer.model)

    return {
        "train_tokens": tokens,
        "effective_batch_size": effective_batch,
        "epochs": epoch_count,
        "global_steps": steps,
        "approximate_flops": tokens * approx_flops_per_token if tokens and approx_flops_per_token else None,
    }


def _approximate_flops_per_token(model) -> Optional[float]:
    try:
        hidden = getattr(model.config, "hidden_size")
        num_layers = getattr(model.config, "num_hidden_layers")
        vocab = getattr(model.config, "vocab_size")
    except AttributeError:
        return None

    if not hidden or not num_layers:
        return None

    # Simple transformer FLOPs estimate: 6 * N_layers * hidden_size^2
    return 6 * num_layers * (hidden**2) + 2 * hidden * vocab

