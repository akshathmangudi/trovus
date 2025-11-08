"""
Evaluation utilities for Trovus teacher-signal experiments.

This package currently provides the supervised fine-tuning (SFT) pipeline used by
the ``trovus evaluate`` CLI command, as well as placeholders for future
Chain-of-Thought Distillation and Reinforcement Learning methods.
"""

from .sft_runner import (
    DEFAULT_LORA_TARGET_MODULES,
    SFTExecutionResult,
    SFTMethodConfig,
    run_sft_pipeline,
)
from .cot_distillation import CoTDistillationConfig, run_cot_distillation_stub
from .rl_trainer import RLTrainingConfig, run_rl_training_stub

__all__ = [
    "DEFAULT_LORA_TARGET_MODULES",
    "SFTMethodConfig",
    "SFTExecutionResult",
    "run_sft_pipeline",
    "CoTDistillationConfig",
    "run_cot_distillation_stub",
    "RLTrainingConfig",
    "run_rl_training_stub",
]

