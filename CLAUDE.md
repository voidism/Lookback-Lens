# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lookback Lens is a research codebase for detecting and mitigating contextual hallucinations in Large Language Models using only attention maps. The project implements a hallucination detection method based on "lookback ratios" - the ratio of attention weights on input context versus newly generated tokens.

## Core Architecture

### Pipeline Steps (Sequential Workflow)
1. **step01_extract_attns.py**: Extracts attention weights and computes lookback ratios from LLM outputs
2. **step02_eval_gpt4o.py**: Generates GPT-4o annotations for hallucination labels  
3. **step03_lookback_lens.py**: Trains logistic regression classifiers on lookback ratio features
4. **step04_run_decoding.py**: Performs inference with optional classifier-guided sampling

### Key Components
- **generation.py**: LLM class for model loading and inference with attention extraction
- **eval_exact_match.py**: Evaluation utilities for NQ dataset exact match scoring
- **lookback_lens_demo.ipynb**: Interactive demo notebook with full pipeline
- **extract_attentions.py**: Qwen3 model attention extraction for target field analysis
- **visualize_attentions.py**: Attention heatmap visualization and analysis tools

### Data Structure
- **data/**: Contains datasets (NQ-Open, CNN/DM, XSum) in JSONL format
- **classifiers/**: Pre-trained hallucination detection models (.pkl files)
- **transformers-4.32.0/**: Modified Transformers library with attention extraction support

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install modified transformers with attention extraction
pip install -e ./transformers-4.32.0

# Decompress NQ dataset
gzip -d data/nq-open-10_total_documents_gold_at_4.jsonl.gz
```

### Model Access Requirements
Most commands require HuggingFace authentication for LLaMA2 access:
- Login: `huggingface-cli login`
- Or use: `--auth_token <hf_auth_token>` parameter

### Core Pipeline Commands

#### Extract Attention Weights (Step 01)
```bash
python step01_extract_attns.py --model-name meta-llama/Llama-2-7b-chat-hf --data-path data/nq-open-10_total_documents_gold_at_4.jsonl --output-path lookback-ratio-nq-7b.pt --auth_token <token>
python step01_extract_attns.py --model-name meta-llama/Llama-2-7b-chat-hf --data-path data/cnndm-1000.jsonl --output-path lookback-ratio-cnndm-7b.pt --auth_token <token>
```

#### Generate GPT-4o Annotations (Step 02)
```bash
OPENAI_API_KEY={your_key} python step02_eval_gpt4o.py --hyp lookback-ratio-nq-7b.pt --ref data/nq-open-10_total_documents_gold_at_4.jsonl --out anno-nq-7b.jsonl
OPENAI_API_KEY={your_key} python step02_eval_gpt4o.py --hyp lookback-ratio-cnndm-7b.pt --ref data/xsum-1000.jsonl --out anno-cnndm-7b.jsonl
```

#### Train Lookback Lens Classifiers (Step 03)
```bash
# Predefined span mode
python step03_lookback_lens.py --anno_1 anno-nq-7b.jsonl --anno_2 anno-cnndm-7b.jsonl --lookback_ratio_1 lookback-ratio-nq-7b.pt --lookback_ratio_2 lookback-ratio-cnndm-7b.pt --auth_token <token>

# Sliding window mode
python step03_lookback_lens.py --anno_1 anno-nq-7b.jsonl --anno_2 anno-cnndm-7b.jsonl --lookback_ratio_1 lookback-ratio-nq-7b.pt --lookback_ratio_2 lookback-ratio-cnndm-7b.pt --sliding_window 8 --auth_token <token>
```

#### Run Inference (Step 04)
```bash
# Greedy decoding
python step04_run_decoding.py --model_name meta-llama/Llama-2-7b-chat-hf --data_path data/nq-open-10_total_documents_gold_at_4.jsonl --output_path output-nq-greedy.jsonl --num_gpus 1 --auth_token <token>

# Classifier-guided decoding  
python step04_run_decoding.py --model_name meta-llama/Llama-2-7b-chat-hf --data_path data/nq-open-10_total_documents_gold_at_4.jsonl --output_path output-nq-guided.jsonl --num_gpus 1 --do_sample --guiding_classifier classifiers/classifier_anno-cnndm-7b_sliding_window_8.pkl --chunk_size 8 --num_candidates 8 --auth_token <token>

# Parallel inference (for speed)
python step04_run_decoding.py --parallel --total_shard 4 --shard_id 0 [other params]
```

### Evaluation Commands
```bash
# NQ exact match evaluation
python eval_exact_match.py --hyp output-nq-greedy.jsonl --ref data/nq-open-10_total_documents_gold_at_4.jsonl

# XSum GPT-4o evaluation  
OPENAI_API_KEY={your_key} python step02_eval_gpt4o.py --hyp output-xsum-greedy.jsonl --ref data/xsum-1000.jsonl --out eval-results.jsonl
```

### Qwen3 Attention Analysis Commands

#### Extract Qwen3 Attention Weights
```bash
# Extract attention weights from Qwen3 model with thinking mode
python extract_attentions.py --model-name Qwen/Qwen3-14B --data-path data/wandb_gemini.csv --output-path qwen3_attentions.pt --jsonl-output qwen3_records.log --num-samples 10 --max-new-tokens 25600

# With custom parameters
python extract_attentions.py --model-name Qwen/Qwen3-14B --data-path data/wandb_gemini.csv --num-samples 5 --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 --max-memory 40
```

#### Visualize Attention Heatmaps
```bash
# Create attention visualization heatmaps
python visualize_attentions.py --data-path qwen3_attentions.pt --output-dir attention_visualizations --tokenizer-name Qwen/Qwen3-14B

# Limit number of samples to visualize
python visualize_attentions.py --data-path qwen3_attentions.pt --output-dir attention_visualizations --max-samples 5
```

## Key Technical Details

- **Modified Transformers**: Uses custom transformers-4.32.0 with attention extraction capabilities
- **Multi-GPU Support**: Step 04 supports distributed inference with device_map="auto"
- **Memory Requirements**: LLaMA inference requires high-memory environments (>40GB recommended)
- **Lookback Ratio Calculation**: Attention weights ratio between context tokens vs generated tokens
- **Transfer Learning**: Classifiers trained on one dataset/task can transfer to others
- **Qwen3 Support**: Extract and visualize attention weights from Qwen3-14B with thinking mode enabled
- **Target Field Analysis**: Focus on specific text segments (marked by ###) for attention analysis
- **Attention Visualization**: Generate heatmaps showing attention patterns across prompt, thinking, and response sections

## Data Formats

- **Lookback Ratios**: PyTorch tensors with shape (examples, layers, heads, tokens)
- **Annotations**: JSONL files with hallucination binary labels
- **Datasets**: JSONL format with fields like 'question', 'context', 'answer', 'summary'
- **Classifiers**: Scikit-learn LogisticRegression models saved as pickle files
- **Qwen3 Attention Data**: PyTorch .pt files containing attention weights, target fields, and token information
- **Wandb Gemini CSV**: Input data with columns 'prompt', 'conv_num' for Qwen3 analysis
- **Visualization Output**: PNG heatmaps showing attention patterns and summary statistics