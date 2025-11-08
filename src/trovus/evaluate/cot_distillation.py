from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class CoTDistillationConfig:
    dataset: str
    output_dir: Path
    teacher_model: Optional[str] = None
    notes: Optional[str] = None


def run_cot_distillation_stub(config: CoTDistillationConfig) -> None:
    """
    Placeholder implementation for Chain-of-Thought distillation.

    Once implemented, this method will:
      * Source rationales from a large teacher model (or cached traces),
      * Align them with the registered student dataset,
      * Run a supervised fine-tuning pass that emphasises rationale fidelity,
      * Emit the same artifact structure as SFT (metrics, compute summary, adapters).
    """

    console.print(
        "[yellow]Chain-of-Thought distillation is not implemented yet.[/yellow] "
        "Configuration has been recorded for future use."
    )

    console.print(f"  Dataset: {config.dataset}")
    if config.teacher_model:
        console.print(f"  Teacher model: {config.teacher_model}")
    if config.notes:
        console.print(f"  Notes: {config.notes}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    placeholder = config.output_dir / "TODO.txt"
    placeholder.write_text(
        "CoT Distillation pipeline is not yet implemented.\n"
        "Please check back once the research team adds teacher rationale generation.\n",
        encoding="utf-8",
    )

