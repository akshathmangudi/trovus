import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from rich.console import Console

from trovus.download import ModelDownloader


console = Console()


def _safe_model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def _looks_like_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    indicator_files = {"config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin"}
    for file_name in indicator_files:
        if (path / file_name).exists():
            return True
    # Some repos place files in subdirectories — check one level deep.
    for child in path.iterdir():
        if child.is_dir():
            for file_name in indicator_files:
                if (child / file_name).exists():
                    return True
    return False


def _candidate_model_paths(model_id: str, models_dir: Path) -> Iterable[Path]:
    # Some users may download directly into the models directory root.
    yield models_dir
    safe_name = _safe_model_dir_name(model_id)
    yield models_dir / safe_name
    if "/" in model_id:
        org, name = model_id.split("/", 1)
        yield models_dir / org / name
        yield models_dir / name
    yield models_dir / model_id


def resolve_local_model_path(model_id: str, models_dir: Path) -> Optional[Path]:
    models_dir = models_dir.expanduser().resolve()
    if not models_dir.exists():
        return None
    for candidate in _candidate_model_paths(model_id, models_dir):
        if _looks_like_model_dir(candidate):
            return candidate
    return None


def ensure_model_on_disk(
    model_id: str,
    models_dir: Path,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    existing_path = None if force_download else resolve_local_model_path(model_id, models_dir)
    if existing_path:
        console.print(f"[green]✓ Found existing model weights at[/green] {existing_path}")
        return existing_path

    console.print(f"[cyan]Downloading model weights for[/cyan] {model_id}")
    downloader = ModelDownloader(cache_dir=str(cache_dir) if cache_dir else None)
    download_target = models_dir / _safe_model_dir_name(model_id)
    download_target.parent.mkdir(parents=True, exist_ok=True)
    path = downloader.download_model(
        repo_id=model_id,
        output_dir=str(download_target),
        include_patterns=None,
        exclude_patterns=None,
        specific_files=None,
        revision="main",
        force_download=force_download,
        resume=True,
    )
    if not path:
        raise RuntimeError(f"Failed to download model {model_id}")
    return Path(path)


def timestamped_run_dir(base_dir: Path, model_id: str, method_name: str) -> Path:
    safe_model = _safe_model_dir_name(model_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / safe_model / method_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


@dataclass
class DatasetSpec:
    hub_id: str
    config_name: Optional[str]
    train_split: str
    eval_split: Optional[str]
    description: str


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        hub_id="gsm8k",
        config_name="main",
        train_split="train",
        eval_split="test",
        description="Grade-school math word problems requiring short multi-step reasoning.",
    ),
    "math": DatasetSpec(
        hub_id="hendrycks/competition_math",
        config_name=None,
        train_split="train",
        eval_split="test",
        description="Competition-level mathematics problems spanning algebra, geometry, and more.",
    ),
    "arc_challenge": DatasetSpec(
        hub_id="ai2_arc",
        config_name="ARC-Challenge",
        train_split="train",
        eval_split="validation",
        description="ARC Challenge benchmark of multiple-choice science reasoning questions.",
    ),
}


def normalize_dataset_key(name: str) -> str:
    return name.lower().replace("-", "_")


def resolve_dataset_spec(dataset_name: str) -> DatasetSpec:
    key = normalize_dataset_key(dataset_name)
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{dataset_name}' is not registered. "
            f"Available options: {', '.join(sorted(DATASET_REGISTRY.keys()))}"
        )
    return DATASET_REGISTRY[key]


def expand_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()

