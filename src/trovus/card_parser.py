#!/usr/bin/env python3
"""
Trovus Model Card Parser

Parses model cards to extract download information and metadata.
Supports YAML frontmatter and markdown parsing.

Authors: Trovus Research Team
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console

console = Console()


class ModelCardParser:
    """
    Parses model cards to extract download and model information.
    """
    
    def __init__(self):
        self.console = Console()
    
    def parse_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        Parse YAML frontmatter from markdown content.
        
        Args:
            content: Markdown content with potential frontmatter
            
        Returns:
            Dictionary of frontmatter data
        """
        # Look for YAML frontmatter (between --- markers)
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(frontmatter_pattern, content, re.DOTALL)
        
        if match:
            try:
                frontmatter_yaml = match.group(1)
                return yaml.safe_load(frontmatter_yaml) or {}
            except yaml.YAMLError as e:
                console.print(f"[yellow]Warning: Could not parse YAML frontmatter: {e}[/yellow]")
                return {}
        
        return {}
    
    def extract_repo_id_from_title(self, content: str) -> Optional[str]:
        """
        Extract repository ID from model card title or headers.
        
        Args:
            content: Markdown content
            
        Returns:
            Repository ID if found
        """
        # Look for patterns like "# Model Card: owner/model" or "# owner/model"
        patterns = [
            r'#\s*Model Card:\s*([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)',
            r'#\s*([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)',
            r'Model:\s*([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                return match.group(1)
        
        return None
    
    def extract_download_preferences(self, content: str) -> Dict[str, Any]:
        """
        Extract download preferences from model card content.
        
        Args:
            content: Markdown content
            
        Returns:
            Dictionary of download preferences
        """
        preferences = {}
        
        # Look for download size information
        size_pattern = r'download[_\s]*size[:\s]*([0-9.]+\s*[KMGT]?B)'
        size_match = re.search(size_pattern, content, re.IGNORECASE)
        if size_match:
            preferences['expected_size'] = size_match.group(1)
        
        # Look for recommended file formats
        if 'safetensors' in content.lower():
            preferences['prefer_safetensors'] = True
        
        # Look for model type/category for download optimization
        if 'small language model' in content.lower() or 'efficiency' in content.lower():
            preferences['optimization_target'] = 'efficiency'
        
        return preferences
    
    def parse_card_file(self, card_path: str) -> Dict[str, Any]:
        """
        Parse a model card file and extract all relevant information.
        
        Args:
            card_path: Path to the model card file
            
        Returns:
            Dictionary containing parsed information
        """
        card_file = Path(card_path)
        
        if not card_file.exists():
            console.print(f"[red]Error: Model card file not found: {card_file}[/red]")
            return {}
        
        try:
            content = card_file.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"[red]Error reading model card: {e}[/red]")
            return {}
        
        # Parse frontmatter
        frontmatter = self.parse_frontmatter(content)
        
        # Extract repo ID from various sources
        repo_id = None
        
        # 1. Check frontmatter for explicit repo_id
        if 'repo_id' in frontmatter:
            repo_id = frontmatter['repo_id']
        
        # 2. Check frontmatter for model name/id patterns
        elif 'model_name' in frontmatter:
            repo_id = frontmatter['model_name']
        elif 'id' in frontmatter:
            repo_id = frontmatter['id']
        
        # 3. Extract from filename if it follows pattern
        elif '_' in card_file.stem:
            # Convert filename like "Qwen_Qwen3-0.6B_card" to "Qwen/Qwen3-0.6B"
            parts = card_file.stem.replace('_card', '').replace('_', '/')
            if '/' in parts:
                repo_id = parts
        
        # 4. Extract from content
        if not repo_id:
            repo_id = self.extract_repo_id_from_title(content)
        
        # Extract download preferences
        download_prefs = self.extract_download_preferences(content)
        
        # Combine all information
        result = {
            'card_path': str(card_file),
            'repo_id': repo_id,
            'frontmatter': frontmatter,
            'download_preferences': download_prefs,
            'has_repo_id': bool(repo_id)
        }
        
        # Add any additional metadata from frontmatter
        if frontmatter:
            # Common fields that might be useful for downloads
            useful_fields = [
                'library_name', 'license', 'pipeline_tag', 'base_model',
                'tags', 'datasets', 'language', 'model_type'
            ]
            
            for field in useful_fields:
                if field in frontmatter:
                    result[field] = frontmatter[field]
        
        return result
    
    def suggest_download_options(self, card_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest optimal download options based on model card information.
        
        Args:
            card_info: Parsed model card information
            
        Returns:
            Dictionary of suggested download options
        """
        suggestions = {}
        
        # Default to research-friendly options
        suggestions['include_patterns'] = ['*.safetensors', '*.json']
        suggestions['exclude_patterns'] = ['*.bin', '*.msgpack', '*.h5']
        
        # Adjust based on model card preferences
        prefs = card_info.get('download_preferences', {})
        
        if prefs.get('prefer_safetensors'):
            suggestions['minimal_download'] = True
            suggestions['reason'] = 'Model card indicates safetensors preference'
        
        if prefs.get('optimization_target') == 'efficiency':
            suggestions['research_mode'] = True
            suggestions['reason'] = 'Optimized for efficiency research'
        
        # Suggest based on library
        library = card_info.get('library_name')
        if library == 'transformers':
            suggestions['essential_files'] = [
                'config.json', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', '*.safetensors'
            ]
        
        return suggestions


def parse_card_command(args):
    """Handle parsing a model card and showing information."""
    parser = ModelCardParser()
    
    card_info = parser.parse_card_file(args.card_path)
    
    if not card_info:
        console.print("[red]Could not parse model card.[/red]")
        return
    
    console.print(f"\n[bold cyan]Model Card Analysis: {Path(args.card_path).name}[/bold cyan]")
    
    if card_info['has_repo_id']:
        console.print(f"Repository ID: [green]{card_info['repo_id']}[/green]")
    else:
        console.print("[yellow]Warning: No repository ID found in card[/yellow]")
    
    # Show frontmatter summary
    if card_info['frontmatter']:
        console.print("\n[bold]Frontmatter fields:[/bold]")
        for key, value in card_info['frontmatter'].items():
            if isinstance(value, (list, dict)):
                console.print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                console.print(f"  {key}: {value}")
    
    # Show download preferences
    if card_info['download_preferences']:
        console.print("\n[bold]Download preferences:[/bold]")
        for key, value in card_info['download_preferences'].items():
            console.print(f"  {key}: {value}")
    
    # Show suggestions
    suggestions = parser.suggest_download_options(card_info)
    if suggestions:
        console.print("\n[bold]Suggested download options:[/bold]")
        for key, value in suggestions.items():
            if isinstance(value, list):
                console.print(f"  {key}: {', '.join(value)}")
            else:
                console.print(f"  {key}: {value}")


def add_card_parser_subcommands(subparsers):
    """Add model card parsing subcommands."""
    
    # Parse card command
    parse_parser = subparsers.add_parser(
        'parse-card',
        help='Parse and analyze a model card file',
        description='Extract download information from a model card'
    )
    
    parse_parser.add_argument(
        'card_path',
        help='Path to the model card file'
    )
    
    parse_parser.set_defaults(func=parse_card_command)
