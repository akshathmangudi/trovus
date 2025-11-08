#!/usr/bin/env python3
"""
Trovus Download Module

Handles downloading model weights from HuggingFace Hub with flexible options.
Supports selective downloads, model card integration, and efficient caching.

Authors: Trovus Research Team
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import re

from huggingface_hub import snapshot_download, hf_hub_download, HfApi, hf_hub_url
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich import print as rprint

console = Console()

# Note: HuggingFace Hub's default tqdm already shows:
# - File counting progress (e.g., "Fetching 4 files")
# - Per-file download progress with speed and time remaining
# We don't need to customize it - the default behavior is good


class ModelDownloader:
    """
    Handles downloading models from HuggingFace Hub with various options.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model downloader.
        
        Args:
            cache_dir: Custom cache directory. If None, uses HF default.
        """
        self.api = HfApi()
        self.cache_dir = cache_dir
        self.console = Console()
    
    def parse_model_identifier(self, identifier: str) -> Dict[str, Any]:
        """
        Parse a model identifier to extract repo_id and other metadata.
        
        Args:
            identifier: Model identifier (e.g., "microsoft/DialoGPT-medium", "gpt2")
            
        Returns:
            Dict with parsed information
        """
        # Handle shorthand models (expand to full repo paths)
        shorthand_mapping = {
            "gpt2": "openai-community/gpt2",
            "distilbert": "distilbert-base-uncased",
            "bert": "google-bert/bert-base-uncased",
        }
        
        if identifier in shorthand_mapping:
            identifier = shorthand_mapping[identifier]
        
        # Validate format
        if "/" not in identifier and identifier not in shorthand_mapping.values():
            # Assume it's a model name, try to find it
            console.print(f"[yellow]Warning: '{identifier}' doesn't follow 'owner/model' format. Searching...[/yellow]")
            return {"repo_id": identifier, "needs_search": True}
        
        return {"repo_id": identifier, "needs_search": False}
    
    def get_model_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model information from HuggingFace Hub.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            Model information dictionary or None if error
        """
        try:
            # Get model info - try with files_metadata if available, otherwise without
            try:
                model_info = self.api.model_info(repo_id, files_metadata=True)
            except TypeError:
                # If files_metadata parameter doesn't exist, try without it
                model_info = self.api.model_info(repo_id)
            
            # Calculate approximate size from files
            total_size = 0
            file_info = []
            
            if hasattr(model_info, 'siblings') and model_info.siblings:
                for file in model_info.siblings:
                    # Try multiple ways to get file size
                    file_size = None
                    
                    # Method 1: Direct size attribute (most common)
                    if hasattr(file, 'size'):
                        size_val = file.size
                        if size_val is not None and (isinstance(size_val, (int, float)) and size_val > 0):
                            file_size = int(size_val)
                    
                    # Method 2: Check for size_bytes attribute
                    if file_size is None and hasattr(file, 'size_bytes'):
                        size_val = file.size_bytes
                        if size_val is not None and (isinstance(size_val, (int, float)) and size_val > 0):
                            file_size = int(size_val)
                    
                    # Method 3: Check LFS pointer file size (for large files)
                    if file_size is None and hasattr(file, 'lfs') and file.lfs:
                        if hasattr(file.lfs, 'size'):
                            size_val = file.lfs.size
                            if size_val is not None and (isinstance(size_val, (int, float)) and size_val > 0):
                                file_size = int(size_val)
                        elif hasattr(file.lfs, 'size_bytes'):
                            size_val = file.lfs.size_bytes
                            if size_val is not None and (isinstance(size_val, (int, float)) and size_val > 0):
                                file_size = int(size_val)
                    
                    # Get filename
                    filename = getattr(file, 'rfilename', None) or getattr(file, 'filename', None)
                    
                    if filename:
                        # Always add file info, even if size is unknown
                        file_info.append({
                            "filename": filename,
                            "size": file_size if file_size is not None else 0
                        })
                        
                        # Add to total only if we have a valid size
                        if file_size is not None and file_size > 0:
                            total_size += file_size
            
            return {
                "repo_id": repo_id,
                "total_size": total_size,
                "files": file_info,
                "tags": getattr(model_info, 'tags', []),
                "library_name": getattr(model_info, 'library_name', None),
                "model_info": model_info
            }
            
        except RepositoryNotFoundError:
            console.print(f"[red]Error: Model '{repo_id}' not found on HuggingFace Hub.[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Error fetching model info: {e}[/red]")
            return None
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
    
    def filter_files(self, files: List[Dict], include_patterns: Optional[List[str]] = None, 
                    exclude_patterns: Optional[List[str]] = None, specific_files: Optional[List[str]] = None) -> List[str]:
        """
        Filter files based on patterns and specific file requests.
        
        Args:
            files: List of file dictionaries
            include_patterns: Patterns to include (e.g., ["*.safetensors"])
            exclude_patterns: Patterns to exclude (e.g., ["*.bin"])
            specific_files: Specific files to download
            
        Returns:
            List of filtered filenames
        """
        import fnmatch
        
        if specific_files:
            return specific_files
        
        filtered_files = []
        
        for file_info in files:
            filename = file_info["filename"]
            
            # Apply include patterns
            if include_patterns:
                if not any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns):
                    continue
            
            # Apply exclude patterns
            if exclude_patterns:
                if any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns):
                    continue
            
            filtered_files.append(filename)
        
        return filtered_files
    
    def download_model(self, repo_id: str, 
                      output_dir: Optional[str] = None,
                      include_patterns: Optional[List[str]] = None,
                      exclude_patterns: Optional[List[str]] = None,
                      specific_files: Optional[List[str]] = None,
                      revision: str = "main",
                      force_download: bool = False,
                      resume: bool = True) -> Optional[str]:
        """
        Download a model from HuggingFace Hub.
        
        Args:
            repo_id: Repository ID to download
            output_dir: Local directory to download to
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            specific_files: Specific files to download
            revision: Git revision to download
            force_download: Force re-download even if cached
            resume: Resume interrupted downloads
            
        Returns:
            Path to downloaded model directory or None if error
        """
        # Get model info first
        model_info = self.get_model_info(repo_id)
        if not model_info:
            return None
        
        # Show download info
        console.print(f"\n[bold green]Downloading {repo_id}[/bold green]")
        
        # Filter files if needed
        filtered_files = []
        filtered_size = 0
        
        if include_patterns or exclude_patterns or specific_files:
            filtered_files = self.filter_files(
                model_info['files'], 
                include_patterns, 
                exclude_patterns, 
                specific_files
            )
            
            # Calculate filtered size by matching filenames
            # Create a dict for quick lookup
            file_dict = {f["filename"]: f for f in model_info['files']}
            
            for filename in filtered_files:
                if filename in file_dict:
                    file_size = file_dict[filename]["size"]
                    if file_size > 0:
                        filtered_size += file_size
            
            # Show total size if we have it
            if model_info['total_size'] > 0:
                console.print(f"Total size: {self.format_size(model_info['total_size'])}")
            
            # Show filtered size and file count
            if filtered_size > 0:
                console.print(f"Filtered size: {self.format_size(filtered_size)}")
            else:
                console.print(f"Filtered size: [yellow]Unknown (size info unavailable)[/yellow]")
            
            console.print(f"Files to download: {len(filtered_files)}")
        else:
            # No filters, show total size
            if model_info['total_size'] > 0:
                console.print(f"Total size: {self.format_size(model_info['total_size'])}")
            else:
                console.print(f"Total size: [yellow]Unknown (size info unavailable)[/yellow]")
            
            console.print(f"Files to download: {len(model_info['files'])}")
        
        # Confirm download if large (only if we know the size)
        if model_info['total_size'] > 1_000_000_000:  # 1GB
            if not Confirm.ask(f"Model is {self.format_size(model_info['total_size'])}. Continue download?"):
                console.print("[yellow]Download cancelled.[/yellow]")
                return None
        
        try:
            console.print("")  # Add spacing before progress bars
            
            # Use snapshot_download for full downloads or filtered downloads
            # HuggingFace Hub's default tqdm will show:
            # - File counting progress (e.g., "Fetching 4 files")
            # - Per-file download progress with speed and time remaining
            if specific_files and len(specific_files) == 1:
                # Single file download
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=specific_files[0],
                    revision=revision,
                    cache_dir=self.cache_dir,
                    local_dir=output_dir,
                    force_download=force_download,
                    resume_download=resume
                )
            else:
                # Multiple files or full repo download
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=self.cache_dir,
                    local_dir=output_dir,
                    allow_patterns=include_patterns,
                    ignore_patterns=exclude_patterns,
                    force_download=force_download,
                    resume_download=resume
                )
            
            console.print(f"\n[bold green]✓ Successfully downloaded to: {downloaded_path}[/bold green]")
            return downloaded_path
            
        except Exception as e:
            console.print(f"[red]Error downloading model: {e}[/red]")
            return None


def download_command(args):
    """Handle the download subcommand."""
    downloader = ModelDownloader(cache_dir=args.cache_dir)
    
    # Parse model identifier
    model_info = downloader.parse_model_identifier(args.model)
    
    if model_info["needs_search"]:
        console.print(f"[yellow]Model '{args.model}' needs disambiguation. Please use full 'owner/model' format.[/yellow]")
        return
    
    repo_id = model_info["repo_id"]
    
    # Prepare download options
    include_patterns = args.include if args.include else None
    exclude_patterns = args.exclude if args.exclude else None
    specific_files = args.files if args.files else None
    
    # Handle preset modes
    if args.minimal:
        # Include all common model weight formats + essential configs
        include_patterns = [
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "*.safetensors", "*.bin", "*.h5", "*.model"
        ]
        exclude_patterns = ["*.msgpack", "*.onnx", "*.tflite"]  # Exclude less common formats
        console.print("[blue]Using minimal download mode (configs + model weights)[/blue]")
    elif args.research_mode:
        # Include all model weights but exclude very large or unnecessary files
        include_patterns = [
            "*.safetensors", "*.bin", "*.h5", "*.model", "*.json", 
            "*.txt", "*.py", "*.md"  # Include configs, tokenizers, and docs
        ]
        exclude_patterns = [
            "*.msgpack", "*.onnx", "*.tflite", "*.gguf",  # Exclude specialized formats
            "*tf_model*", "*flax_model*"  # Exclude framework-specific duplicates when possible
        ]
        console.print("[blue]Using research mode (all weights + configs, exclude specialized formats)[/blue]")
    
    # Download the model
    downloaded_path = downloader.download_model(
        repo_id=repo_id,
        output_dir=args.output_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        specific_files=specific_files,
        revision=args.revision,
        force_download=args.force,
        resume=not args.no_resume
    )
    
    if downloaded_path:
        console.print(f"\n[bold]Model downloaded successfully![/bold]")
        console.print(f"Location: {downloaded_path}")


def download_from_card_command(args):
    """Handle downloading from a model card."""
    console.print("[yellow]Model card download functionality coming soon![/yellow]")
    # TODO: Implement model card parsing and download


def add_download_subparser(subparsers):
    """Add download-related subcommands to the argument parser."""
    
    # Main download command
    download_parser = subparsers.add_parser(
        'download', 
        help='Download model weights from HuggingFace Hub',
        description='Download models with flexible filtering options'
    )
    
    download_parser.add_argument(
        'model',
        help='Model identifier (e.g., microsoft/DialoGPT-medium, Qwen/Qwen3-0.6B)'
    )
    
    download_parser.add_argument(
        '--output-dir', '-o',
        help='Local directory to download model to'
    )
    
    download_parser.add_argument(
        '--cache-dir',
        help='Custom cache directory (default: HuggingFace cache)'
    )
    
    download_parser.add_argument(
        '--include',
        nargs='+',
        help='File patterns to include (e.g., *.safetensors *.json)'
    )
    
    download_parser.add_argument(
        '--exclude',
        nargs='+',
        help='File patterns to exclude (e.g., *.bin *.msgpack)'
    )
    
    download_parser.add_argument(
        '--files',
        nargs='+',
        help='Specific files to download'
    )
    
    download_parser.add_argument(
        '--revision',
        default='main',
        help='Git revision to download (default: main)'
    )
    
    download_parser.add_argument(
        '--minimal',
        action='store_true',
        help='Download only essential files (configs + model weights: safetensors/bin/h5/model)'
    )
    
    download_parser.add_argument(
        '--research-mode',
        action='store_true',
        help='Download optimized for research (all weights + configs, exclude specialized formats)'
    )
    
    download_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if cached'
    )
    
    download_parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resuming interrupted downloads'
    )
    
    download_parser.set_defaults(func=download_command)
    
    # Download from card command
    card_parser = subparsers.add_parser(
        'download-card',
        help='Download model from a model card file',
        description='Download models using information from saved model cards'
    )
    
    card_parser.add_argument(
        'card_path',
        help='Path to the model card file'
    )
    
    card_parser.add_argument(
        '--output-dir', '-o',
        help='Local directory to download model to'
    )
    
    card_parser.set_defaults(func=download_from_card_command)
