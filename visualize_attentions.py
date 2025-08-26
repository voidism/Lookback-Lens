import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re
import glob
from typing import List, Tuple, Optional
from transformers import AutoTokenizer
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def discover_sample_files(data_dir: str):
    """Discover all sample_*.pt files in the directory"""
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data path {data_dir} is not a directory. Expected results_extraction/ directory.")
    
    sample_files = glob.glob(os.path.join(data_dir, "sample_*.pt"))
    if not sample_files:
        raise ValueError(f"No sample_*.pt files found in {data_dir}")
    
    # Sort files by sample index
    sample_files.sort(key=lambda x: int(os.path.basename(x)[7:10]))  # sample_XXX.pt -> XXX
    return sample_files

def load_attention_data(data_dir: str):
    """Load attention data from individual sample_*.pt files in directory"""
    print(f"Loading attention data from directory: {data_dir}")
    
    sample_files = discover_sample_files(data_dir)
    print(f"Found {len(sample_files)} sample files")
    
    data = []
    for sample_file in sample_files:
        try:
            sample_data = torch.load(sample_file, map_location='cpu')
            # Each file contains a list with one sample
            if isinstance(sample_data, list) and len(sample_data) == 1:
                data.append(sample_data[0])
            else:
                print(f"Warning: Unexpected data format in {sample_file}")
                data.append(sample_data)
        except Exception as e:
            print(f"Error loading {sample_file}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(data)} samples")
    return data

def find_section_boundaries(model_completion: str, tokenizer, full_tokens: List[str]) -> Tuple[int, int, int, int]:
    """
    Find boundaries for different sections in the generated text:
    - Prompt end
    - Thinking start/end (<think>...</think>)
    - Response start
    """
    # Find thinking tags
    thinking_start_pattern = r'<think>'
    thinking_end_pattern = r'</think>'
    
    thinking_start_match = re.search(thinking_start_pattern, model_completion)
    thinking_end_match = re.search(thinking_end_pattern, model_completion)
    
    # Initialize boundaries
    prompt_end = 0  # This will be set based on context length
    thinking_start = -1
    thinking_end = -1
    response_start = len(full_tokens)
    
    if thinking_start_match:
        thinking_start_pos = thinking_start_match.start()
        # Find token position for thinking start
        prefix_text = model_completion[:thinking_start_pos]
        thinking_start = len(tokenizer.encode(prefix_text, add_special_tokens=False))
    
    if thinking_end_match:
        thinking_end_pos = thinking_end_match.end()
        # Find token position for thinking end
        prefix_text = model_completion[:thinking_end_pos]
        thinking_end = len(tokenizer.encode(prefix_text, add_special_tokens=False))
        response_start = thinking_end
    
    return prompt_end, thinking_start, thinking_end, response_start

def get_token_strings(tokenizer, model_completion_ids: List[int], context: str) -> Tuple[List[str], int]:
    """
    Get token strings for visualization
    Returns: (all_token_strings, context_length)
    """
    # Tokenize context to get prompt length
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    context_length = len(context_tokens)
    
    # Get all tokens (context + generated)
    context_token_strings = [tokenizer.decode([token_id]) for token_id in context_tokens]
    generated_token_strings = [tokenizer.decode([token_id]) for token_id in model_completion_ids]
    
    all_token_strings = context_token_strings + generated_token_strings
    
    return all_token_strings, context_length

def apply_attention_optimization(attention_matrix: np.ndarray, title_suffix: str = "", is_mean_version: bool = False) -> Tuple[np.ndarray, str]:
    """
    Apply optimization strategy, with mean version forced to match max version behavior
    
    Args:
        is_mean_version: If True, force mean version to use same strategy as max version
        
    Returns:
        Tuple of (processed_matrix, colorbar_label, cmap[, vmin, vmax])
    """
    # Analyze Y-axis variation
    y_variance_mean = 0  # Default value for single row case
    if attention_matrix.shape[0] > 1:
        y_variance = np.var(attention_matrix, axis=1)  # variance across X-axis for each Y
        y_variance_mean = np.mean(y_variance)
        y_variance_max = np.max(y_variance)
        print(f"Y-axis variation{title_suffix}: mean_var={y_variance_mean:.8f}, max_var={y_variance_max:.8f}")
        
        # Additional diagnostics for Y-axis differences
        row_correlations = []
        for i in range(attention_matrix.shape[0]):
            for j in range(i+1, attention_matrix.shape[0]):
                corr = np.corrcoef(attention_matrix[i], attention_matrix[j])[0, 1]
                row_correlations.append(corr)
        if row_correlations:
            print(f"Target field token correlations{title_suffix}: mean={np.mean(row_correlations):.4f}, range=[{np.min(row_correlations):.4f}, {np.max(row_correlations):.4f}]")
    
    # Enhanced normalization strategy for weak Y-axis variation
    vmin, vmax = np.percentile(attention_matrix, [1, 99])  # Use more extreme percentiles
    
    # Make both versions use the same colormap strategy
    if is_mean_version:
        print(f"Forcing mean version to use same colormap as max version{title_suffix}")
        # Force mean version to use plasma colormap like max version
        return attention_matrix, 'Attention Weight', 'plasma', vmin, vmax
    
    # Original adaptive strategy for max version  
    # Option 1: For very small values, use log scale with stronger enhancement
    if vmax < 0.01:
        log_matrix = np.log10(attention_matrix + 1e-8)
        return log_matrix, 'Log10(Attention Weight + 1e-8)', 'viridis'
    # Option 2: Use row-wise normalization to enhance Y-axis differences
    elif attention_matrix.shape[0] > 1 and y_variance_mean < 1e-6:
        print(f"Applying row-wise normalization to enhance Y-axis variation{title_suffix}")
        normalized_matrix = np.zeros_like(attention_matrix)
        for i in range(attention_matrix.shape[0]):
            row = attention_matrix[i]
            row_min, row_max = np.percentile(row, [1, 99])  # Use extreme percentiles for better contrast
            if row_max > row_min:
                normalized_matrix[i] = (row - row_min) / (row_max - row_min)
            else:
                normalized_matrix[i] = row
        return normalized_matrix, 'Row-wise Normalized Attention', 'RdYlBu_r', 0, 1
    else:
        # Use percentile normalization with enhanced colormap
        return attention_matrix, 'Attention Weight', 'plasma', vmin, vmax

def create_attention_heatmap(attention_matrix: np.ndarray, 
                           all_tokens: List[str],
                           field_tokens: List[str],
                           boundaries: Tuple[int, int, int, int],
                           title: str = "Attention Heatmap",
                           figsize: Tuple[int, int] = (20, 8),
                           interpolation_factor: int = 4,
                           smooth_sigma: float = 0.8) -> plt.Figure:
    """
    Create attention heatmap with section boundaries marked and continuous interpolation
    """
    prompt_end, thinking_start, thinking_end, response_start = boundaries
    
    # Apply interpolation to create continuous visualization
    interpolated_matrix = interpolate_attention_matrix(attention_matrix, 
                                                      interpolation_factor=interpolation_factor,
                                                      smooth_sigma=smooth_sigma)
    
    # Use all tokens without sampling
    display_tokens = all_tokens
    original_seq_len = len(all_tokens)
    interpolated_seq_len = interpolated_matrix.shape[1]
    
    # Scale boundaries to interpolated coordinates
    scale_factor = interpolated_seq_len / original_seq_len
    prompt_end_scaled = int(prompt_end * scale_factor)
    thinking_start_scaled = int(thinking_start * scale_factor) if thinking_start > 0 else -1
    thinking_end_scaled = int(thinking_end * scale_factor) if thinking_end > 0 else -1
    response_start_scaled = int(response_start * scale_factor)
    
    # Clean token strings for display
    clean_tokens = []
    for token in display_tokens:
        # Replace special characters and limit length
        clean_token = token.replace('\n', '\\n').replace('\t', '\\t')
        if len(clean_token) > 10:
            clean_token = clean_token[:7] + '...'
        clean_tokens.append(clean_token)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply consistent optimization strategy
    print(f"Attention matrix stats: min={np.min(interpolated_matrix):.6f}, max={np.max(interpolated_matrix):.6f}, mean={np.mean(interpolated_matrix):.6f}")
    
    # Get the title suffix for diagnostics and version info
    title_suffix = " (from interpolated matrix)"
    is_mean_version = False
    print(f"Debug: Title is '{title}'")
    if "Max" in title:
        title_suffix = " (Max pooled)"
        print("Debug: Detected MAX version")
    elif "Mean" in title or "Continuous" in title:
        title_suffix = " (Mean pooled)"
        is_mean_version = True
        print("Debug: Detected MEAN version - will force consistent coloring")
        
    # Apply optimization and get display parameters
    optimization_result = apply_attention_optimization(interpolated_matrix, title_suffix, is_mean_version)
    
    if len(optimization_result) == 3:
        display_matrix, cbar_label, cmap = optimization_result
        im = ax.imshow(display_matrix, cmap=cmap, aspect='auto', interpolation='bilinear')
    else:  # len == 5
        display_matrix, cbar_label, cmap, vmin, vmax = optimization_result
        im = ax.imshow(display_matrix, cmap=cmap, aspect='auto', interpolation='bilinear',
                      vmin=vmin, vmax=vmax)
    
    # Set labels
    ax.set_xlabel('Input Tokens (Context + Thinking + Response)', fontsize=12)
    ax.set_ylabel('Target Field Tokens', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set tick labels - map original token positions to interpolated coordinates
    max_labels = 50 if len(clean_tokens) <= 50 else 20
    
    if len(clean_tokens) <= max_labels:
        # Show all tokens
        token_indices = range(len(clean_tokens))
        selected_tokens = clean_tokens
    else:
        # Show evenly spaced tokens
        step = len(clean_tokens) // max_labels
        token_indices = range(0, len(clean_tokens), step)
        selected_tokens = [clean_tokens[i] for i in token_indices]
    
    # Map original token positions to interpolated coordinates
    interpolated_positions = [int(i * scale_factor) for i in token_indices]
    
    ax.set_xticks(interpolated_positions)
    ax.set_xticklabels(selected_tokens, rotation=45, ha='right', fontsize=8)
    
    if len(field_tokens) <= 20:
        ax.set_yticks(range(len(field_tokens)))
        ax.set_yticklabels(field_tokens, fontsize=10)
    
    # Add section boundary lines - only show Prompt End and Thinking End
    if prompt_end_scaled > 0:
        ax.axvline(x=prompt_end_scaled-0.5, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='Prompt End')
    
    if thinking_end_scaled > 0:
        ax.axvline(x=thinking_end_scaled-0.5, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='Thinking End')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def interpolate_attention_matrix(attention_matrix: np.ndarray, interpolation_factor: int = 4, 
                                smooth_sigma: float = 0.8) -> np.ndarray:
    """
    Interpolate attention matrix along x-axis (sequence dimension) to create continuous visualization
    
    Args:
        attention_matrix: [field_length, seq_len] attention matrix
        interpolation_factor: factor to increase resolution (default: 4x)
        smooth_sigma: gaussian smoothing parameter (0 = no smoothing)
    
    Returns:
        np.ndarray: [field_length, seq_len * interpolation_factor] interpolated matrix
    """
    if attention_matrix.size == 0:
        return attention_matrix
    
    # Convert to float32 to ensure compatibility with scipy functions
    attention_matrix = attention_matrix.astype(np.float32)
    
    field_length, seq_len = attention_matrix.shape
    
    # Create new x coordinates with higher resolution
    original_x = np.arange(seq_len)
    new_x = np.linspace(0, seq_len - 1, seq_len * interpolation_factor)
    
    # Interpolate each row (each target field token's attention)
    interpolated_matrix = np.zeros((field_length, len(new_x)))
    
    for i in range(field_length):
        attention_row = attention_matrix[i, :]
        
        # Apply gaussian smoothing before interpolation if specified
        if smooth_sigma > 0:
            attention_row = gaussian_filter1d(attention_row, sigma=smooth_sigma)
        
        # Create interpolation function
        f_interp = interp1d(original_x, attention_row, kind='cubic', 
                           bounds_error=False, fill_value='extrapolate')
        
        # Interpolate to new x coordinates
        interpolated_matrix[i, :] = f_interp(new_x)
        
        # Ensure non-negative values (attention weights should be >= 0)
        interpolated_matrix[i, :] = np.maximum(interpolated_matrix[i, :], 0)
    
    return interpolated_matrix

def average_attention_across_heads(field_attentions: List, method: str = 'mean') -> np.ndarray:
    """
    Average attention weights across heads and layers
    
    Args:
        field_attentions: List[tensor] - [layer][heads, field_length, seq_len]
        method: 'mean' or 'max' for aggregation method
        
    Returns:
        np.ndarray: [field_length, seq_len] - averaged attention matrix
    """
    if not field_attentions:
        return np.array([])
    
    print(f"Processing {len(field_attentions)} layers of attention data")
    
    # Convert tensors to numpy arrays for processing
    layer_attentions = []
    
    for layer_idx, layer_attention in enumerate(field_attentions):
        # layer_attention shape: [heads, field_length, seq_len]
        if hasattr(layer_attention, 'numpy'):
            layer_attn_np = layer_attention.numpy()
        elif hasattr(layer_attention, 'cpu'):
            layer_attn_np = layer_attention.cpu().numpy()
        else:
            layer_attn_np = np.array(layer_attention)
            
        print(f"Layer {layer_idx} attention shape: {layer_attn_np.shape}")
        
        # Average across heads: [heads, field_length, seq_len] -> [field_length, seq_len]
        if method == 'mean':
            head_averaged = np.mean(layer_attn_np, axis=0)
        elif method == 'max':
            head_averaged = np.max(layer_attn_np, axis=0)
        else:
            head_averaged = np.mean(layer_attn_np, axis=0)
            
        layer_attentions.append(head_averaged)
    
    # Stack all layers: List[[field_length, seq_len]] -> [num_layers, field_length, seq_len]
    stacked_attentions = np.stack(layer_attentions, axis=0)
    print(f"Stacked attention shape: {stacked_attentions.shape}")
    
    # Average across layers: [num_layers, field_length, seq_len] -> [field_length, seq_len]
    if method == 'mean':
        final_attention = np.mean(stacked_attentions, axis=0)
    elif method == 'max':
        final_attention = np.max(stacked_attentions, axis=0)
    else:
        final_attention = np.mean(stacked_attentions, axis=0)
    
    print(f"Final attention matrix shape: {final_attention.shape}")
    
    return final_attention

def visualize_sample(sample_data: dict, tokenizer, output_dir: str, sample_idx: int):
    """Visualize attention for a single sample"""
    
    print(f"\nProcessing sample {sample_idx}")
    
    # Extract data
    context = sample_data['context']
    model_completion = sample_data['model_completion']
    target_field = sample_data['target_field']
    field_attentions = sample_data['field_attentions']
    model_completion_ids = sample_data['model_completion_ids']
    
    if not field_attentions or not target_field:
        print(f"Skipping sample {sample_idx} - no valid attention data or target field")
        return
    
    print(f"Target field: '{target_field}'")
    print(f"Attention data has {len(field_attentions)} layers")
    if field_attentions:
        field_length = field_attentions[0].shape[1]  # [heads, field_length, seq_len]
        print(f"Target field has {field_length} tokens")
    
    # Get token strings
    all_tokens, context_length = get_token_strings(tokenizer, model_completion_ids, context)
    field_tokens = tokenizer.encode(target_field, add_special_tokens=False)
    field_token_strings = [tokenizer.decode([token_id]) for token_id in field_tokens]
    
    print(f"Total tokens: {len(all_tokens)}, Context length: {context_length}")
    
    # Find section boundaries
    boundaries = find_section_boundaries(model_completion, tokenizer, all_tokens)
    boundaries = (context_length, boundaries[1], boundaries[2], boundaries[3])  # Set prompt_end to context_length
    
    print(f"Boundaries - Prompt: {boundaries[0]}, Think start: {boundaries[1]}, Think end: {boundaries[2]}, Response: {boundaries[3]}")
    
    # Average attention across heads and layers
    attention_matrix = average_attention_across_heads(field_attentions, method='mean')
    
    if attention_matrix.size == 0:
        print(f"Skipping sample {sample_idx} - empty attention matrix")
        return
    
    print(f"Attention matrix shape: {attention_matrix.shape}")
    
    # Create visualization
    title = f"Sample {sample_idx}: Attention from Target Field to Previous Tokens (Continuous)"
    fig = create_attention_heatmap(
        attention_matrix=attention_matrix,
        all_tokens=all_tokens,
        field_tokens=field_token_strings,
        boundaries=boundaries,
        title=title,
        figsize=(25, 10),
        interpolation_factor=4,
        smooth_sigma=0.8
    )
    
    # Save figure
    output_path = os.path.join(output_dir, f"attention_heatmap_sample_{sample_idx}.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved heatmap to {output_path}")
    
    # Also create a simplified version with max pooling
    attention_matrix_max = average_attention_across_heads(field_attentions, method='max')
    title_max = f"Sample {sample_idx}: Max Attention from Target Field to Previous Tokens (Continuous)"
    fig_max = create_attention_heatmap(
        attention_matrix=attention_matrix_max,
        all_tokens=all_tokens,
        field_tokens=field_token_strings,
        boundaries=boundaries,
        title=title_max,
        figsize=(25, 10),
        interpolation_factor=4,
        smooth_sigma=0.8
    )
    
    output_path_max = os.path.join(output_dir, f"attention_heatmap_sample_{sample_idx}_max.png")
    fig_max.savefig(output_path_max, dpi=150, bbox_inches='tight')
    plt.close(fig_max)
    
    print(f"Saved max attention heatmap to {output_path_max}")

def create_summary_statistics(data: List[dict], output_dir: str):
    """Create summary statistics and overview visualizations"""
    
    valid_samples = [sample for sample in data if sample.get('field_attentions') and sample.get('target_field')]
    
    print(f"\nCreating summary statistics for {len(valid_samples)} valid samples")
    
    # Collect statistics
    # field_attentions is now [layer][heads, field_length, seq_len], so get field_length from first layer
    field_lengths = [sample['field_attentions'][0].shape[1] if sample['field_attentions'] else 0 
                     for sample in valid_samples]
    completion_lengths = [len(sample['model_completion_ids']) for sample in valid_samples]
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Field length distribution
    axes[0, 0].hist(field_lengths, bins=10, alpha=0.7, color='blue')
    axes[0, 0].set_title('Target Field Length Distribution')
    axes[0, 0].set_xlabel('Number of Tokens in Target Field')
    axes[0, 0].set_ylabel('Frequency')
    
    # Completion length distribution
    axes[0, 1].hist(completion_lengths, bins=10, alpha=0.7, color='green')
    axes[0, 1].set_title('Model Completion Length Distribution')
    axes[0, 1].set_xlabel('Number of Generated Tokens')
    axes[0, 1].set_ylabel('Frequency')
    
    # Sample target fields (first 10 characters)
    sample_fields = [sample['target_field'][:50] + '...' if len(sample['target_field']) > 50 
                    else sample['target_field'] for sample in valid_samples[:10]]
    
    axes[1, 0].barh(range(len(sample_fields)), [len(field) for field in sample_fields])
    axes[1, 0].set_title('Target Field Lengths (First 10 Samples)')
    axes[1, 0].set_xlabel('Field Length (characters)')
    axes[1, 0].set_yticks(range(len(sample_fields)))
    axes[1, 0].set_yticklabels([f"Sample {i}" for i in range(len(sample_fields))], fontsize=8)
    
    # Success rate
    total_samples = len(data)
    valid_samples_count = len(valid_samples)
    success_rate = valid_samples_count / total_samples * 100
    
    axes[1, 1].pie([valid_samples_count, total_samples - valid_samples_count], 
                   labels=['Valid Samples', 'Invalid Samples'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%')
    axes[1, 1].set_title(f'Sample Processing Success Rate\n({success_rate:.1f}% successful)')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary_statistics.png")
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved summary statistics to {summary_path}")
    
    # Print text summary
    summary_text = f"""
Attention Visualization Summary
==============================
Total samples: {total_samples}
Valid samples with attention data: {valid_samples_count}
Success rate: {success_rate:.1f}%

Target Field Statistics:
- Average field length: {np.mean(field_lengths):.1f} tokens
- Min field length: {min(field_lengths)} tokens
- Max field length: {max(field_lengths)} tokens

Model Completion Statistics:
- Average completion length: {np.mean(completion_lengths):.1f} tokens
- Min completion length: {min(completion_lengths)} tokens  
- Max completion length: {max(completion_lengths)} tokens
"""
    
    summary_text_path = os.path.join(output_dir, "summary.txt")
    with open(summary_text_path, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)

def main():
    parser = argparse.ArgumentParser(description="Visualize attention weights from Qwen3 model")
    parser.add_argument("--data-path", type=str, default="results_extraction",
                       help="Path to directory containing sample_*.pt files")
    parser.add_argument("--output-dir", type=str, default="attention_visualizations",
                       help="Directory to save visualization outputs")
    parser.add_argument("--tokenizer-name", type=str, default="Qwen/Qwen3-14B",
                       help="Tokenizer name for decoding tokens")
    parser.add_argument("--max-samples", type=int, default=1,
                       help="Maximum number of samples to visualize (default: all)")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data directory {args.data_path} does not exist. Please run extract_attentions.py first.")
    if not os.path.isdir(args.data_path):
        raise ValueError(f"Data path {args.data_path} must be a directory containing sample_*.pt files.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load attention data from directory
    data = load_attention_data(args.data_path)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    
    # Limit samples if specified
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Limiting to {args.max_samples} samples")
    
    # Create summary statistics
    create_summary_statistics(data, args.output_dir)
    
    # Process each sample
    for idx, sample in enumerate(data):
        try:
            visualize_sample(sample, tokenizer, args.output_dir, idx)
        except Exception as e:
            print(f"Error visualizing sample {idx}: {str(e)}")
            continue
    
    print(f"\nVisualization complete! Check {args.output_dir} for results.")
    print("Generated files:")
    print("  - attention_heatmap_sample_*.png: Individual attention heatmaps")
    print("  - attention_heatmap_sample_*_max.png: Max pooled attention heatmaps")
    print("  - summary_statistics.png: Overview statistics")
    print("  - summary.txt: Text summary")

if __name__ == "__main__":
    main()