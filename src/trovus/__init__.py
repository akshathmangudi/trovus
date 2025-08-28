"""
Trovus: Small Language Model Efficiency Frontier Research Library

A CLI-based toolkit for exploring and mapping efficiency frontiers of 
fine-tuning techniques on small language models.

Developed by the Trovus Research Team.
"""

__version__ = "0.1.0"
__author__ = "Trovus Research Team"
__email__ = "akshathmangudi@gmail.com"

# Import main functionality for easy access
from .cli import main

__all__ = ["main", "__version__"]
