from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class RLTrainingConfig:
    dataset: str
    output_dir: Path
    reward_model: Optional[str] = None
    notes: Optional[str] = None


def run_rl_training_stub(config: RLTrainingConfig) -> None:
    """
    Placeholder for Reinforcement Learning based teacher signal fine-tuning.

    The final implementation will integrate:
      * Reward modelling / process reward model loading,
      * PPO, GRPO, or other small-model friendly RL algorithms,
      * On-policy sampling pipelines with budget tracking,
      * Consistent artifact emission for efficiency frontier analysis.
    """

    console.print(
        "[yellow]RL-assisted fine-tuning is not implemented yet.[/yellow] "
        "Configuration has been recorded for future development."
    )

    console.print(f"  Dataset: {config.dataset}")
    if config.reward_model:
        console.print(f"  Reward model: {config.reward_model}")
    if config.notes:
        console.print(f"  Notes: {config.notes}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    placeholder = config.output_dir / "TODO.txt"
    placeholder.write_text(
        "RL-assisted training pipeline is not yet implemented.\n"
        "Future work will attach reward models and PPO/GRPO loops here.\n",
        encoding="utf-8",
    )

