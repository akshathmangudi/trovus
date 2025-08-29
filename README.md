# Trovus

The official repository of the paper, "Reasoning on a Budget: How Teacher Signals Shape Efficiency Frontiers for Small Language Models"

### Search for Models
```bash
# Interactive search
trovus search

# Example searches:
# - "microsoft deberta"
# - "qwen 270m base"  
# - "phi mini instruct"
```

### Download Models
```bash
# Download with research mode (recommended)
trovus download microsoft/deberta-v3-small --output-dir ./models --research-mode

# Download minimal files only
trovus download Qwen/Qwen3-0.6B --minimal

# Custom file patterns
trovus download microsoft/phi-3-mini --include "*.safetensors" "*.json" --exclude "*.bin"
```

### Model Card Management
```bash
# Parse a model card for download information
trovus parse-card ./model_cards/microsoft_deberta-v3-small_card.md

# The search interface can save model cards automatically
```

### Cache Management
```bash
# List downloaded models
trovus list

# Show cache information
trovus cache-info

# Get detailed model info
trovus info microsoft/deberta-v3-small

# Remove models
trovus remove microsoft/deberta-v3-small
```

## Command Reference

### Search Commands
- `trovus search` - Interactive model search with fuzzy matching
- `trovus parse-card <file>` - Extract information from model cards

### Download Commands  
- `trovus download <model>` - Download models with flexible options
  - `--output-dir` - Custom download directory
  - `--research-mode` - Optimized for research (all weights + configs, exclude specialized formats)
  - `--minimal` - Essential files only (configs + model weights: safetensors/bin/h5/model)
  - `--include` - File patterns to include
  - `--exclude` - File patterns to exclude
  - `--force` - Force re-download

### Management Commands
- `trovus list` - List cached models with sizes and dates
- `trovus info <model>` - Detailed information about a specific model
- `trovus cache-info` - Overall cache statistics
- `trovus remove <model>` - Remove models from cache

## Download Modes

### Research Mode (`--research-mode`)
Downloads all model weights and configurations while excluding specialized formats:
- ✅ Includes: `*.safetensors`, `*.bin`, `*.h5`, `*.model`, `*.json`, `*.txt`, `*.py`, `*.md`
- ❌ Excludes: `*.msgpack`, `*.onnx`, `*.tflite`, `*.gguf`, framework-specific duplicates

### Minimal Mode (`--minimal`)  
Downloads only essential files needed to use the model:
- ✅ Includes: Config files + all model weight formats (`*.safetensors`, `*.bin`, `*.h5`, `*.model`)
- ❌ Excludes: Less common formats (`*.msgpack`, `*.onnx`, `*.tflite`)

## Examples

```bash
# Search and download workflow
trovus search
# Type: "microsoft deberta"
# Save model card when prompted
trovus parse-card ./model_cards/microsoft_deberta-v3-small_card.md
trovus download microsoft/deberta-v3-small --output-dir ./models --research-mode

# Quick download for inference
trovus download Qwen/Qwen3-0.6B --minimal --output-dir ./models

# Custom download with specific files
trovus download microsoft/phi-3-mini \
  --include "*.safetensors" "config.json" "tokenizer*" \
  --output-dir ./models

# Check what you have downloaded
trovus list
trovus cache-info
```

To-do: 
- [X] Implement the search retriever + model card downloader. 
- [X] Convert "python -m trovus" into a universal command: "using trovus" 
- [X] Include ability to download model weights. 
- [ ] Implement the first set of fine-tuning techniques and employ them on the first model. 
- [ ] Will add more steps once step 3 is complete.