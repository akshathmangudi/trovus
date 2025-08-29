#!/usr/bin/env python3
"""
Trovus Model Manager

Handles listing, removing, and managing downloaded models.
Provides information about local model cache and storage.

Authors: Trovus Research Team
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich import print as rprint

console = Console()


class ModelManager:
    """
    Manages local model cache and provides model information.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            cache_dir: Custom cache directory. If None, uses HF default.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(HF_HUB_CACHE)
        self.console = Console()
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        size = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
    
    def scan_local_models(self) -> List[Dict[str, Any]]:
        """
        Scan local cache for downloaded models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            cache_info = scan_cache_dir(self.cache_dir)
            models = []
            
            for repo in cache_info.repos:
                models.append({
                    "repo_id": repo.repo_id,
                    "repo_type": repo.repo_type,
                    "size_on_disk": repo.size_on_disk,
                    "nb_files": repo.nb_files,
                    "last_accessed": repo.last_accessed,
                    "last_modified": repo.last_modified,
                    "refs": list(repo.refs),
                    "local_path": str(repo.repo_path)
                })
            
            return models
            
        except Exception as e:
            console.print(f"[red]Error scanning cache: {e}[/red]")
            return []
    
    def list_models(self, show_details: bool = False):
        """
        Display list of downloaded models.
        
        Args:
            show_details: Whether to show detailed information
        """
        models = self.scan_local_models()
        
        if not models:
            console.print("[yellow]No models found in cache.[/yellow]")
            console.print(f"Cache directory: {self.cache_dir}")
            return
        
        # Create table
        table = Table(title="Downloaded Models")
        
        if show_details:
            table.add_column("Repository", style="cyan", no_wrap=True)
            table.add_column("Type", style="magenta")
            table.add_column("Size", style="green")
            table.add_column("Files", justify="right", style="blue")
            table.add_column("Last Accessed", style="yellow")
            table.add_column("Refs", style="dim")
        else:
            table.add_column("Repository", style="cyan", no_wrap=True)
            table.add_column("Type", style="magenta")
            table.add_column("Size", style="green")
            table.add_column("Last Accessed", style="yellow")
        
        # Add model data
        total_size = 0
        for model in models:
            total_size += model["size_on_disk"]
            
            last_accessed = "Never"
            if model["last_accessed"]:
                # Handle both datetime objects and timestamps
                if isinstance(model["last_accessed"], (int, float)):
                    last_accessed = datetime.fromtimestamp(model["last_accessed"]).strftime("%Y-%m-%d")
                else:
                    last_accessed = model["last_accessed"].strftime("%Y-%m-%d")
            
            if show_details:
                refs_str = ", ".join(model["refs"][:3])  # Show first 3 refs
                if len(model["refs"]) > 3:
                    refs_str += "..."
                
                table.add_row(
                    model["repo_id"],
                    model["repo_type"],
                    self.format_size(model["size_on_disk"]),
                    str(model["nb_files"]),
                    last_accessed,
                    refs_str
                )
            else:
                table.add_row(
                    model["repo_id"],
                    model["repo_type"],
                    self.format_size(model["size_on_disk"]),
                    last_accessed
                )
        
        console.print(table)
        console.print(f"\n[bold]Total: {len(models)} models, {self.format_size(total_size)}[/bold]")
        console.print(f"Cache directory: {self.cache_dir}")
    
    def get_model_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific downloaded model.
        
        Args:
            repo_id: Repository ID to get info for
            
        Returns:
            Model information dictionary or None if not found
        """
        models = self.scan_local_models()
        
        for model in models:
            if model["repo_id"] == repo_id:
                return model
        
        return None
    
    def show_model_info(self, repo_id: str):
        """
        Display detailed information about a specific model.
        
        Args:
            repo_id: Repository ID to show info for
        """
        model = self.get_model_info(repo_id)
        
        if not model:
            console.print(f"[red]Model '{repo_id}' not found in local cache.[/red]")
            self._suggest_similar_models(repo_id)
            return
        
        console.print(f"\n[bold cyan]Model Information: {repo_id}[/bold cyan]")
        console.print(f"Type: {model['repo_type']}")
        console.print(f"Size on disk: {self.format_size(model['size_on_disk'])}")
        console.print(f"Number of files: {model['nb_files']}")
        console.print(f"Local path: {model['local_path']}")
        
        if model['last_accessed']:
            if isinstance(model['last_accessed'], (int, float)):
                last_accessed_str = datetime.fromtimestamp(model['last_accessed']).strftime('%Y-%m-%d %H:%M:%S')
            else:
                last_accessed_str = model['last_accessed'].strftime('%Y-%m-%d %H:%M:%S')
            console.print(f"Last accessed: {last_accessed_str}")
            
        if model['last_modified']:
            if isinstance(model['last_modified'], (int, float)):
                last_modified_str = datetime.fromtimestamp(model['last_modified']).strftime('%Y-%m-%d %H:%M:%S')
            else:
                last_modified_str = model['last_modified'].strftime('%Y-%m-%d %H:%M:%S')
            console.print(f"Last modified: {last_modified_str}")
        
        console.print(f"Available refs: {', '.join(model['refs'])}")
        
        # List files in the model directory
        try:
            model_path = Path(model['local_path'])
            if model_path.exists():
                files = list(model_path.rglob('*'))
                files = [f for f in files if f.is_file()]
                
                if files:
                    console.print(f"\n[bold]Files ({len(files)}):[/bold]")
                    
                    # Group files by type
                    file_types = {}
                    for file_path in files:
                        ext = file_path.suffix.lower()
                        if ext not in file_types:
                            file_types[ext] = []
                        file_types[ext].append(file_path)
                    
                    for ext, files_list in sorted(file_types.items()):
                        console.print(f"  {ext or 'no extension'}: {len(files_list)} files")
                        
                        # Show largest files for this type
                        files_with_size = []
                        for f in files_list:
                            try:
                                size = f.stat().st_size
                                files_with_size.append((f, size))
                            except:
                                continue
                        
                        files_with_size.sort(key=lambda x: x[1], reverse=True)
                        for f, size in files_with_size[:3]:  # Show top 3 largest
                            console.print(f"    - {f.name}: {self.format_size(size)}")
        except Exception as e:
            console.print(f"[yellow]Could not list files: {e}[/yellow]")
    
    def _suggest_similar_models(self, repo_id: str):
        """Suggest similar models based on partial matches."""
        models = self.scan_local_models()
        suggestions = []
        
        for model in models:
            if repo_id.lower() in model["repo_id"].lower():
                suggestions.append(model["repo_id"])
        
        if suggestions:
            console.print(f"\nDid you mean one of these?")
            for suggestion in suggestions[:5]:  # Show up to 5 suggestions
                console.print(f"  - {suggestion}")
    
    def remove_model(self, repo_id: str, force: bool = False):
        """
        Remove a downloaded model from local cache.
        
        Args:
            repo_id: Repository ID to remove
            force: Skip confirmation prompt
        """
        model = self.get_model_info(repo_id)
        
        if not model:
            console.print(f"[red]Model '{repo_id}' not found in local cache.[/red]")
            self._suggest_similar_models(repo_id)
            return
        
        size_str = self.format_size(model['size_on_disk'])
        
        if not force:
            if not Confirm.ask(f"Remove '{repo_id}' ({size_str})?"):
                console.print("[yellow]Removal cancelled.[/yellow]")
                return
        
        try:
            model_path = Path(model['local_path'])
            if model_path.exists():
                shutil.rmtree(model_path)
                console.print(f"[green]✓ Successfully removed '{repo_id}' ({size_str})[/green]")
            else:
                console.print(f"[yellow]Model path not found: {model_path}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error removing model: {e}[/red]")
    
    def cache_info(self):
        """Display cache directory information and statistics."""
        console.print(f"[bold]Cache Information[/bold]")
        console.print(f"Cache directory: {self.cache_dir}")
        console.print(f"Directory exists: {'Yes' if self.cache_dir.exists() else 'No'}")
        
        if self.cache_dir.exists():
            try:
                total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
                console.print(f"Total cache size: {self.format_size(total_size)}")
            except Exception as e:
                console.print(f"[yellow]Could not calculate cache size: {e}[/yellow]")
        
        models = self.scan_local_models()
        console.print(f"Number of models: {len(models)}")
        
        if models:
            total_model_size = sum(m['size_on_disk'] for m in models)
            console.print(f"Total model size: {self.format_size(total_model_size)}")
            
            # Show breakdown by type
            type_counts = {}
            for model in models:
                repo_type = model['repo_type']
                if repo_type not in type_counts:
                    type_counts[repo_type] = 0
                type_counts[repo_type] += 1
            
            console.print("\nBreakdown by type:")
            for repo_type, count in sorted(type_counts.items()):
                console.print(f"  {repo_type}: {count} models")


def list_command(args):
    """Handle the list subcommand."""
    manager = ModelManager(cache_dir=args.cache_dir)
    manager.list_models(show_details=args.details)


def info_command(args):
    """Handle the info subcommand."""
    manager = ModelManager(cache_dir=args.cache_dir)
    manager.show_model_info(args.model)


def remove_command(args):
    """Handle the remove subcommand."""
    manager = ModelManager(cache_dir=args.cache_dir)
    manager.remove_model(args.model, force=args.force)


def cache_info_command(args):
    """Handle the cache-info subcommand."""
    manager = ModelManager(cache_dir=args.cache_dir)
    manager.cache_info()


def add_management_subparsers(subparsers):
    """Add model management subcommands to the argument parser."""
    
    # List models command
    list_parser = subparsers.add_parser(
        'list', 
        help='List downloaded models',
        description='Show all models in local cache'
    )
    
    list_parser.add_argument(
        '--cache-dir',
        help='Custom cache directory to scan'
    )
    
    list_parser.add_argument(
        '--details', '-d',
        action='store_true',
        help='Show detailed information'
    )
    
    list_parser.set_defaults(func=list_command)
    
    # Model info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show detailed information about a model',
        description='Display comprehensive information about a downloaded model'
    )
    
    info_parser.add_argument(
        'model',
        help='Model identifier to show info for'
    )
    
    info_parser.add_argument(
        '--cache-dir',
        help='Custom cache directory to scan'
    )
    
    info_parser.set_defaults(func=info_command)
    
    # Remove model command
    remove_parser = subparsers.add_parser(
        'remove',
        help='Remove a downloaded model',
        description='Delete a model from local cache'
    )
    
    remove_parser.add_argument(
        'model',
        help='Model identifier to remove'
    )
    
    remove_parser.add_argument(
        '--cache-dir',
        help='Custom cache directory'
    )
    
    remove_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    remove_parser.set_defaults(func=remove_command)
    
    # Cache info command
    cache_parser = subparsers.add_parser(
        'cache-info',
        help='Show cache directory information',
        description='Display cache statistics and information'
    )
    
    cache_parser.add_argument(
        '--cache-dir',
        help='Custom cache directory to scan'
    )
    
    cache_parser.set_defaults(func=cache_info_command)
