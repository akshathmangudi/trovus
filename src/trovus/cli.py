#!/usr/bin/env python3
"""
Trovus CLI - HuggingFace Model Search Engine

A command-line interface for searching and retrieving model cards from HuggingFace Hub.
This serves as the foundation for building efficiency frontier research on small language models.

Authors: Trovus Research Team
"""

import sys
from typing import List, Optional

from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from rapidfuzz import fuzz, process


# Initialize Rich console for beautiful output
console = Console()


class ModelSearchEngine:
    """
    HuggingFace model search engine for finding and retrieving model information.
    """
    
    def __init__(self):
        self.api = HfApi()
        self.console = Console()
    
    def preprocess_query(self, query: str) -> str:
        """
        Clean and normalize search query without hardcoded mappings.
        
        Args:
            query: Raw search query
            
        Returns:
            Cleaned query string
        """
        # Just clean whitespace and normalize - let HuggingFace + fuzzy matching handle the rest
        return query.strip()

    def filter_small_language_models(self, results: List[dict], query: str, preference: str = "") -> List[dict]:
        """
        Apply intelligent filtering for small language models using fuzzy matching.
        
        Args:
            results: List of model dictionaries  
            query: Original search query
            preference: User preference ("base", "instruct", etc.)
            
        Returns:
            Filtered and sorted list of models
        """
        if not results:
            return []
        
        # Extract model names for fuzzy matching
        model_names = [model['id'] for model in results]
        
        # Use fuzzy matching to find best matches
        fuzzy_matches = process.extract(
            query, 
            model_names, 
            scorer=fuzz.partial_ratio,
            limit=len(results)
        )
        
        # Create a mapping of model names to scores
        score_map = {match[0]: match[1] for match in fuzzy_matches}
        
        # More permissive threshold for fuzzy matching
        filtered_models = []
        for model in results:
            model_id = model['id']
            fuzzy_score = score_map.get(model_id, 0)
            
            # Lower threshold to catch more relevant models (40 instead of 60)
            if fuzzy_score >= 40:
                model['fuzzy_score'] = fuzzy_score
                filtered_models.append(model)
        
        # If no good fuzzy matches, try keyword matching
        if not filtered_models:
            query_terms = query.lower().split()
            for model in results:
                model_id_lower = model['id'].lower()
                # Check if any query term appears in model ID
                if any(term in model_id_lower for term in query_terms):
                    model['fuzzy_score'] = 30  # Base score for keyword match
                    filtered_models.append(model)
        
        # Sort by fuzzy score (descending) then by downloads (descending)
        filtered_models.sort(
            key=lambda x: (x['fuzzy_score'], x['downloads']), 
            reverse=True
        )
        
        # Apply additional small LM heuristics with user preference
        prioritized_models = []
        for model in filtered_models:
            model_id = model['id'].lower()
            tags = [tag.lower() for tag in model.get('tags', [])]
            
            # Boost small model indicators
            size_indicators = ['270m', '0.5b', '1b', '1.5b', '3b', '7b', 'small', 'mini', 'tiny']
            has_size_indicator = any(indicator in model_id for indicator in size_indicators)
            
            # Check model type
            instruct_indicators = ['instruct', 'chat', 'assistant']
            is_instruct = any(indicator in model_id or indicator in tags for indicator in instruct_indicators)
            is_base = not is_instruct
            
            # Calculate priority score with user preference
            priority = model['fuzzy_score']
            if has_size_indicator:
                priority += 10
            
            # Apply user preference
            preference_lower = preference.lower()
            if preference_lower == "base" and is_base:
                priority += 20  # Strong boost for base models
            elif preference_lower == "instruct" and is_instruct:
                priority += 20  # Strong boost for instruct models
            elif preference_lower == "chat" and is_instruct:
                priority += 20  # Chat models are usually instruct variants
            elif not preference:  # No preference specified
                if is_instruct:
                    priority += 5  # Default slight boost for instruct models
            
            model['priority_score'] = priority
            prioritized_models.append(model)
        
        # Final sort by priority score
        prioritized_models.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return prioritized_models

    def show_search_suggestions(self) -> None:
        """Show helpful search suggestions when no results are found."""
        self.console.print("\n[dim]💡 Try these popular small language models:[/dim]")
        suggestions = [
            "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b",
            "phi-3.5-mini", "phi-3-mini", 
            "smollm-135m", "smollm-360m", "smollm-1.7b",
            "gemma-2b", "llama-3.2-1b", "llama-3.2-3b"
        ]
        
        for i, suggestion in enumerate(suggestions[:6], 1):
            self.console.print(f"  [cyan]{i}.[/cyan] {suggestion}")
        
        self.console.print("\n[dim]Or try searching with terms like: 'instruct', 'chat', 'small', 'mini'[/dim]")

    def search_models(self, query: str, limit: int = 5, filter_options: Optional[dict] = None, preference: str = "") -> List[dict]:
        """
        Search for models on HuggingFace Hub using advanced filtering.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            filter_options: Additional filtering options
            preference: User preference like "base", "instruct", etc.
            
        Returns:
            List of model information dictionaries
        """
        try:
            # Clean the query but don't over-preprocess
            cleaned_query = self.preprocess_query(query)
            
            # Use HuggingFace's advanced search parameters
            search_params = {
                'search': cleaned_query,
                'limit': limit * 5,  # Get more results for better fuzzy filtering
                'sort': 'downloads',  # Sort by popularity
                'direction': -1,  # Descending order
                'task': 'text-generation',  # Focus on language models
                'library': 'transformers',  # Standard ML library
            }
            
            # Add additional filters if provided
            if filter_options:
                search_params.update(filter_options)
            
            # Search models using HuggingFace API with advanced parameters
            models = list(self.api.list_models(**search_params))
            
            # Convert to dictionaries with relevant information
            results = []
            for model in models:
                model_info = {
                    'id': model.id,
                    'author': model.author or 'Unknown',
                    'downloads': getattr(model, 'downloads', 0) or 0,
                    'likes': getattr(model, 'likes', 0) or 0,
                    'tags': getattr(model, 'tags', []) or [],
                }
                results.append(model_info)
            
            # Apply smart filtering for small language models using original query and preference
            filtered_results = self.filter_small_language_models(results, query, preference)
            
            # Return top results
            return filtered_results[:limit]
            
        except Exception as e:
            self.console.print(f"[red]Error searching models: {e}[/red]")
            return []
    
    def get_suggestions(self, query: str, limit: int = 3) -> List[dict]:
        """
        Get suggested models when no exact matches are found.
        
        Args:
            query: Original search query
            limit: Number of suggestions to return
            
        Returns:
            List of suggested model dictionaries
        """
        try:
            # Cast a wider net with relaxed search parameters
            search_params = {
                'search': query.strip(),
                'limit': 50,  # Get many results for better suggestions
                'sort': 'downloads',
                'direction': -1,
                'task': 'text-generation',
                'library': 'transformers',
            }
            
            models = list(self.api.list_models(**search_params))
            
            if not models:
                # If still no results, try with just the first word
                first_word = query.strip().split()[0] if query.strip() else ""
                if first_word:
                    search_params['search'] = first_word
                    models = list(self.api.list_models(**search_params))
            
            # Convert to our format
            results = []
            for model in models:
                model_info = {
                    'id': model.id,
                    'author': model.author or 'Unknown',
                    'downloads': getattr(model, 'downloads', 0) or 0,
                    'likes': getattr(model, 'likes', 0) or 0,
                    'tags': getattr(model, 'tags', []) or [],
                }
                results.append(model_info)
            
            # Use fuzzy matching to find closest suggestions
            if results:
                model_names = [model['id'] for model in results]
                fuzzy_matches = process.extract(
                    query, 
                    model_names, 
                    scorer=fuzz.partial_ratio,
                    limit=limit * 2  # Get more for filtering
                )
                
                # Filter suggestions and add fuzzy scores
                suggestions = []
                for match in fuzzy_matches:
                    match_name, score = match[0], match[1]  # Explicit unpacking
                    if score > 20:  # Very permissive for suggestions
                        model = next((m for m in results if m['id'] == match_name), None)
                        if model:
                            model['fuzzy_score'] = score
                            
                            # Add preference scoring for base vs instruct models
                            model_id_lower = model['id'].lower()
                            
                            # Boost base models (non-instruct/chat variants)
                            is_instruct = any(variant in model_id_lower for variant in ['instruct', 'chat', 'assistant'])
                            is_base = not is_instruct
                            
                            # Calculate preference score
                            preference_score = score
                            if is_base:
                                preference_score += 15  # Strong preference for base models
                            
                            model['preference_score'] = preference_score
                            suggestions.append(model)
                
                # Sort by preference score first (base models preferred), then by downloads
                suggestions.sort(key=lambda x: (x['preference_score'], x['downloads']), reverse=True)
                return suggestions[:limit]
            
            return []
            
        except Exception as e:
            self.console.print(f"[dim red]Error getting suggestions: {e}[/dim red]")
            return []

    def display_results(self, results: List[dict], query: str = "", search_engine=None) -> None:
        """
        Display search results in a formatted table.
        
        Args:
            results: List of model information dictionaries
            query: Original search query for suggestions
            search_engine: Reference to ModelSearchEngine for saving model cards
        """
        if not results:
            self.console.print("[yellow]No models found.[/yellow]")
            
            # Show "Did you mean?" suggestions
            if query:
                suggestions = self.get_suggestions(query, limit=10)  # Increased to 10
                if suggestions:
                    self.console.print("\n[dim]💡 Did you mean?[/dim]")
                    
                    suggestion_table = Table(show_header=False, box=None, padding=(0, 1))
                    suggestion_table.add_column("", style="cyan", no_wrap=True)
                    suggestion_table.add_column("", style="magenta")
                    suggestion_table.add_column("", style="dim blue", justify="right")
                    suggestion_table.add_column("", style="dim green", justify="center")
                    
                    for idx, suggestion in enumerate(suggestions, 1):
                        # Add indicator for base models
                        model_type = ""
                        model_id_lower = suggestion['id'].lower()
                        if not any(variant in model_id_lower for variant in ['instruct', 'chat', 'assistant']):
                            model_type = "📍"  # Base model indicator
                        
                        suggestion_table.add_row(
                            f"{idx}.",
                            suggestion['id'],
                            f"↓ {suggestion['downloads']:,}",
                            model_type
                        )
                    
                    self.console.print(suggestion_table)
                    self.console.print("[dim]📍 = Base model (non-instruct) | Try searching for one of these models specifically.[/dim]")
                    
                    # Ask if user wants to save a model card from suggestions
                    if search_engine:
                        save_choice = Prompt.ask(
                            "\nWould you like to save a model card from these suggestions?", 
                            choices=["yes", "no"], 
                            default="no"
                        )

                        if save_choice == "yes":
                            model_choice = Prompt.ask(
                                "Enter suggestion number (1-10)", 
                                choices=[str(i) for i in range(1, len(suggestions) + 1)]
                            )
                            
                            selected_model = suggestions[int(model_choice) - 1]
                            model_id = selected_model['id']
                            
                            self.console.print(f"\n[dim]Fetching model card for: {model_id}[/dim]")
                            card_content = search_engine.get_model_card(model_id)
                            
                            if card_content:
                                search_engine.save_model_card(model_id, card_content)
                else:
                    self.show_search_suggestions()
            else:
                self.show_search_suggestions()
            return
        
        # Create a rich table for results
        table = Table(title="🤗 HuggingFace Model Search Results")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Model ID", style="magenta")
        table.add_column("Author", style="green")
        table.add_column("Downloads", justify="right", style="blue")
        table.add_column("Likes", justify="right", style="red")
        
        # Add rows to the table
        for idx, model in enumerate(results, 1):
            table.add_row(
                str(idx),
                model['id'],
                model['author'],
                str(model['downloads']),
                str(model['likes'])
            )
        
        self.console.print(table)
    
    def get_model_card(self, model_id: str) -> Optional[str]:
        """
        Retrieve and return model card content.
        
        Args:
            model_id: HuggingFace model identifier
            
        Returns:
            Model card content as string, or None if not found
        """
        try:
            # Import ModelCard from huggingface_hub
            from huggingface_hub import ModelCard
            
            # Load the model card using the proper HuggingFace method
            card = ModelCard.load(model_id)
            
            # Return the content (includes metadata header and text)
            if card.content:
                return card.content
            elif card.text:
                # Fallback to text only if content is empty
                return card.text
            else:
                return "Model card exists but has no content."
                
        except Exception as e:
            # If ModelCard.load fails, try the old method as fallback
            try:
                model_info = self.api.model_info(model_id)
                
                # Try to get README content from model files
                files = []
                if hasattr(model_info, 'siblings') and model_info.siblings:
                    files = [f.rfilename for f in model_info.siblings if f.rfilename.lower() == 'readme.md']
                if files:
                    # Download README.md file content
                    from huggingface_hub import hf_hub_download
                    readme_path = hf_hub_download(repo_id=model_id, filename="README.md", repo_type="model")
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return f"No README.md found for {model_id}."
                    
            except Exception as e2:
                self.console.print(f"[red]Error fetching model card for {model_id}: {e} | Fallback error: {e2}[/red]")
                return None
    
    def save_model_card(self, model_id: str, content: str, output_dir: str = "model_cards") -> bool:
        """
        Save model card to file.
        
        Args:
            model_id: HuggingFace model identifier
            content: Model card content
            output_dir: Directory to save the card
            
        Returns:
            True if saved successfully, False otherwise
        """
        import os
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename from model ID (replace / with _)
            filename = f"{model_id.replace('/', '_')}_card.md"
            filepath = os.path.join(output_dir, filename)
            
            # Write content to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Model Card: {model_id}\n\n")
                f.write(content)
            
            self.console.print(f"[green]✓ Model card saved to: {filepath}[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error saving model card: {e}[/red]")
            return False


def interactive_search():
    """
    Run an interactive model search session.
    """
    search_engine = ModelSearchEngine()
    
    console.print("\n[bold blue]🔍 Trovus HuggingFace Model Search Engine[/bold blue]")
    console.print("[dim]Type 'quit' or 'exit' to end the session.[/dim]")
    console.print("[dim]Add preferences like: 'qwen 270m base' or 'phi mini instruct'[/dim]\n")
    
    current_query = ""
    current_preference = ""
    
    while True:
        try:
            # Get search query from user
            if current_query:
                prompt_text = f"Search models (current: '{current_query}' {current_preference}): "
            else:
                prompt_text = "Search models: "
            
            user_input = prompt(prompt_text)
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            
            if not user_input.strip():
                if current_query:
                    # Re-run current search if user just presses enter
                    query_to_search = current_query
                    preference_to_use = current_preference
                else:
                    continue
            else:
                # Parse input for preferences
                input_lower = user_input.lower()
                preference_keywords = ['base', 'instruct', 'chat']
                
                found_preference = ""
                query_parts = user_input.split()
                
                # Look for preference keywords
                for keyword in preference_keywords:
                    if keyword in input_lower:
                        found_preference = keyword
                        # Remove preference from query
                        query_parts = [part for part in query_parts if part.lower() != keyword]
                        break
                
                new_query = " ".join(query_parts).strip()
                
                # Determine if this is a refinement or new search
                if found_preference and not new_query:
                    # Just adding/changing preference to existing query
                    current_preference = found_preference
                    query_to_search = current_query
                    preference_to_use = current_preference
                elif new_query and not found_preference and current_query:
                    # Check if it's a refinement (preference keyword only) or new search
                    if input_lower in preference_keywords:
                        # It's a preference refinement
                        current_preference = input_lower
                        query_to_search = current_query
                        preference_to_use = current_preference
                    else:
                        # It's a completely new search - reset context
                        current_query = new_query
                        current_preference = ""
                        query_to_search = current_query
                        preference_to_use = current_preference
                else:
                    # Update context normally
                    if new_query:
                        current_query = new_query
                    if found_preference:
                        current_preference = found_preference
                    
                    query_to_search = current_query
                    preference_to_use = current_preference
            
            if not query_to_search:
                continue
            
            # Show what we're searching for
            if preference_to_use:
                console.print(f"\n[dim]Searching for: {query_to_search} [cyan]({preference_to_use} preference)[/cyan][/dim]")
            else:
                console.print(f"\n[dim]Searching for: {query_to_search}[/dim]")
            
            # Search for models
            results = search_engine.search_models(query_to_search, limit=5, preference=preference_to_use)
            
            # Display results with query for suggestions
            search_engine.display_results(results, query_to_search, search_engine)
            
            if results:
                # Ask user if they want to save a model card
                save_choice = Prompt.ask(
                    "\nWould you like to save a model card?", 
                    choices=["y", "n"], 
                    default="n"
                )
                
                if save_choice == "y":
                    model_choice = Prompt.ask(
                        "Enter model number (1-5)", 
                        choices=[str(i) for i in range(1, len(results) + 1)]
                    )
                    
                    selected_model = results[int(model_choice) - 1]
                    model_id = selected_model['id']
                    
                    console.print(f"\n[dim]Fetching model card for: {model_id}[/dim]")
                    card_content = search_engine.get_model_card(model_id)
                    
                    if card_content:
                        search_engine.save_model_card(model_id, card_content)
            
            console.print("\n" + "─" * 50)
            if current_query and current_preference:
                console.print(f"[dim]Current context: '{current_query}' with '{current_preference}' preference[/dim]")
            elif current_query:
                console.print(f"[dim]Current context: '{current_query}' (no preference set)[/dim]")
            else:
                console.print("[dim]No active search context[/dim]")
            console.print("[dim]• Type 'base', 'instruct', or 'chat' to refine current search[/dim]")
            console.print("[dim]• Enter new query to start fresh search | Press Enter to re-run current search[/dim]\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def main():
    """
    Main entry point for the Trovus CLI.
    """
    if len(sys.argv) > 1:
        # Handle command-line arguments (for future expansion)
        command = sys.argv[1]
        if command in ['-h', '--help']:
            console.print("[bold]Trovus - HuggingFace Model Search Engine[/bold]")
            console.print("\nUsage: python -m trovus")
            console.print("       trovus  (if installed as package)")
            console.print("\nCommands:")
            console.print("  Interactive search mode (default)")
            return
    
    # Run interactive search
    interactive_search()


if __name__ == "__main__":
    main()
