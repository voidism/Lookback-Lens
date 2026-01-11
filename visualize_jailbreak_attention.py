import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from pathlib import Path
import glob

FONT_SIZES = {
    "title": 23,
    "label": 21,
    "tick": 18,
    "legend": 16,
    "annotation": 18,
}

def auto_discover_jailbreak_files(jailbreak_results_dir="jailbreak_results"):
    """
    Automatically discover jailbreak attention result files in the specified directory.
    Returns list of .pt files (preferred) or .json files if .pt not available.
    """
    if not os.path.exists(jailbreak_results_dir):
        print(f"Warning: Directory {jailbreak_results_dir} does not exist")
        return []
    
    # Look for .pt files first (they contain full tensor data)
    pt_files = glob.glob(os.path.join(jailbreak_results_dir, "jailbreak_attention_*_n*.pt"))
    
    if pt_files:
        print(f"Found {len(pt_files)} .pt files in {jailbreak_results_dir}")
        # Sort by extracted template length
        pt_files.sort(key=lambda x: extract_template_size_from_filename(x))
        return pt_files
    
    # Fallback to .json files if no .pt files found
    json_files = glob.glob(os.path.join(jailbreak_results_dir, "jailbreak_attention_*_n*.json"))
    
    if json_files:
        print(f"Found {len(json_files)} .json files in {jailbreak_results_dir}")
        # Sort by extracted template length
        json_files.sort(key=lambda x: extract_template_size_from_filename(x))
        return json_files
    
    print(f"No jailbreak attention result files found in {jailbreak_results_dir}")
    return []

def extract_template_size_from_filename(filepath):
    """
    Extract template size from filename like 'jailbreak_attention_2k_n100.pt' -> 2
    """
    try:
        filename = os.path.basename(filepath)
        # Pattern: jailbreak_attention_{size}k_n{samples}.{ext}
        import re
        match = re.search(r'jailbreak_attention_(\d+)k_n\d+\.[^.]+$', filename)
        if match:
            return int(match.group(1))
    except:
        pass
    return 0

def load_results(file_paths):
    """Load results from multiple .pt or JSON files"""
    all_results = {}
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping")
            continue
            
        try:
            # Check file extension to determine loading method
            if file_path.endswith('.pt'):
                # Load PyTorch tensor data
                data = torch.load(file_path, map_location='cpu')
                template_name = data['template_name']
                results = data['attention_data']
                
                # Process tensor data format
                valid_results = []
                for r in results:
                    if 'attention_tensors' in r and r['attention_tensors'] is not None:
                        attention_tensors = r['attention_tensors']
                        
                        # Extract tensor data
                        part3_by_layer_head = attention_tensors['part3_attention_by_layer_head'].numpy()
                        other_by_layer_head = attention_tensors['other_attention_by_layer_head'].numpy()
                        
                        # Calculate ratios, handle division by zero
                        ratios_by_layer_head = np.divide(part3_by_layer_head, other_by_layer_head,
                                                       out=np.full_like(part3_by_layer_head, np.inf),
                                                       where=(other_by_layer_head != 0))
                        
                        # Calculate overall statistics
                        total_part3_attention = part3_by_layer_head.sum()
                        total_other_attention = other_by_layer_head.sum()
                        overall_ratio = total_part3_attention / total_other_attention if total_other_attention > 0 else np.inf
                        
                        processed_result = {
                            'instruction_idx': r['instruction_idx'],
                            'instruction': r['instruction'],
                            'category': r.get('category', 'Unknown'),
                            'num_layers': part3_by_layer_head.shape[0],
                            'num_heads': part3_by_layer_head.shape[1],
                            'part3_attention_by_layer_head': part3_by_layer_head,
                            'other_attention_by_layer_head': other_by_layer_head,
                            'ratios_by_layer_head': ratios_by_layer_head,
                            'overall_ratio': overall_ratio,
                            'total_part3_attention': total_part3_attention,
                            'total_other_attention': total_other_attention
                        }
                        valid_results.append(processed_result)
                
            else:
                # Load JSON format (legacy support)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                template_name = data['template_name']
                results = data['results']
                
                # Process JSON data format with attention_metadata
                valid_results = []
                for r in results:
                    if 'attention_metadata' in r and r['attention_metadata'] is not None:
                        attention_metadata = r['attention_metadata']
                        
                        # This is simplified - JSON format won't have the full tensor data
                        # We'll need the .pt file for full functionality
                        processed_result = {
                            'instruction_idx': r['instruction_idx'],
                            'instruction': r['instruction'],
                            'category': r.get('category', 'Unknown'),
                            'num_layers': attention_metadata['num_layers'],
                            'num_heads': attention_metadata['num_heads'],
                            'overall_ratio': attention_metadata['summary']['overall_ratio'],
                            'total_part3_attention': attention_metadata['summary']['total_part3_attention'],
                            'total_other_attention': attention_metadata['summary']['total_other_attention']
                        }
                        valid_results.append(processed_result)
            
            if valid_results:
                all_results[template_name] = {
                    'results': valid_results,
                    'success_count': len(valid_results),
                    'total_count': data.get('num_samples', len(results)),
                    'model_name': data.get('model_name', 'Unknown'),
                    'num_layers': valid_results[0]['num_layers'] if valid_results else 0,
                    'num_heads': valid_results[0]['num_heads'] if valid_results else 0
                }
                print(f"Loaded {template_name}: {len(valid_results)} valid samples")
                print(f"  - Model architecture: {valid_results[0]['num_layers']} layers, {valid_results[0]['num_heads']} heads")
            else:
                print(f"Warning: No valid attention data in {file_path}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return all_results

def extract_template_size(template_name):
    """Extract numerical size from template name (e.g., template_1k -> 1)"""
    try:
        # Extract number and convert k to thousands
        if 'k' in template_name.lower():
            size_str = template_name.lower().replace('template_', '').replace('k', '')
            return int(size_str)
        else:
            # Try to extract number directly
            import re
            numbers = re.findall(r'\d+', template_name)
            if numbers:
                return int(numbers[0])
    except:
        pass
    
    # Fallback: return based on template name order
    template_order = {
        'template_1k': 1,
        'template_3k': 3,
        'template_11k': 11,
        'template_21k': 21,
        'template_31k': 31,
        'template_47k': 47
    }
    
    return template_order.get(template_name, 0)

def create_box_plot(all_results, output_dir):
    """Create box plot showing attention ratio distribution across templates"""
    if not all_results:
        print("No data to plot")
        return
    
    # Sort templates by size
    sorted_templates = sorted(all_results.keys(), key=extract_template_size)
    
    # Prepare data for plotting
    data_for_plot = []
    labels = []
    
    for template_name in sorted_templates:
        data = all_results[template_name]
        results = data['results']
        
        # Extract overall ratios
        ratios = [r['overall_ratio'] for r in results if r['overall_ratio'] is not None and not np.isinf(r['overall_ratio'])]
        
        # Handle extreme values (cap at reasonable range for visualization)
        capped_ratios = [min(max(r, 0), 10) for r in ratios]  # Cap between 0 and 10
        
        data_for_plot.append(capped_ratios)
        
        # Create label with size and sample count
        size = extract_template_size(template_name)
        count = len(ratios)
        labels.append(f"{size}k\n(n={count})")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create box plot
    box_plot = plt.boxplot(data_for_plot, tick_labels=labels, patch_artist=True, 
                          showfliers=True, whis=1.5)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_for_plot)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Attention Ratio Distribution Across Template Lengths\n(Part 3 Attention / Other User Prompt Attention)', 
              fontsize=FONT_SIZES["title"], pad=20)
    plt.xlabel('Template Length (thousands of tokens)', fontsize=FONT_SIZES["label"])
    plt.ylabel('Attention Ratio', fontsize=FONT_SIZES["label"])
    plt.xticks(fontsize=FONT_SIZES["tick"])
    plt.yticks(fontsize=FONT_SIZES["tick"])
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = []
    for i, template_name in enumerate(sorted_templates):
        data = all_results[template_name]
        results = data['results']
        ratios = [r['overall_ratio'] for r in results if r['overall_ratio'] is not None and not np.isinf(r['overall_ratio'])]
        
        if ratios:  # Only calculate stats if we have valid ratios
            mean_ratio = np.mean(ratios)
            median_ratio = np.median(ratios)
            stats_text.append(f"{extract_template_size(template_name)}k: μ={mean_ratio:.3f}, m={median_ratio:.3f}")
    
    plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=FONT_SIZES["annotation"], 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_ratio_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {output_path}")
    
    plt.close()

def create_trend_plot(all_results, output_dir):
    """Create trend plot showing mean attention ratio vs template length"""
    if not all_results:
        print("No data to plot")
        return
    
    # Extract data for trend analysis
    template_sizes = []
    mean_ratios = []
    std_ratios = []
    ci_95_lower = []
    ci_95_upper = []
    template_names = []
    
    for template_name, data in all_results.items():
        results = data['results']
        ratios = [r['overall_ratio'] for r in results if r['overall_ratio'] is not None and not np.isinf(r['overall_ratio'])]
        if ratios:
            n = len(ratios)
            mean_val = np.mean(ratios)
            std_val = np.std(ratios, ddof=1)  # Sample standard deviation
            
            # 95% confidence interval using t-distribution
            from scipy import stats
            confidence = 0.95
            t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin_of_error = t_critical * std_val / np.sqrt(n)
            
            template_sizes.append(extract_template_size(template_name))
            mean_ratios.append(mean_val)
            std_ratios.append(std_val)
            ci_95_lower.append(mean_val - margin_of_error)
            ci_95_upper.append(mean_val + margin_of_error)
            template_names.append(template_name)
    
    # Sort by template size
    sorted_indices = np.argsort(template_sizes)
    template_sizes = [template_sizes[i] for i in sorted_indices]
    mean_ratios = [mean_ratios[i] for i in sorted_indices]
    std_ratios = [std_ratios[i] for i in sorted_indices]
    ci_95_lower = [ci_95_lower[i] for i in sorted_indices]
    ci_95_upper = [ci_95_upper[i] for i in sorted_indices]
    template_names = [template_names[i] for i in sorted_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot 95% CI as shaded region
    plt.fill_between(template_sizes, ci_95_lower, ci_95_upper, 
                     alpha=0.2, color='blue', label='95% Confidence Interval')
    
    # Plot mean with error bars (std)
    plt.errorbar(template_sizes, mean_ratios, yerr=std_ratios, 
                marker='o', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='Mean ± Std', color='blue')
    
    plt.xlabel('CoT Length (thousands of tokens)', fontsize=FONT_SIZES["label"])
    plt.ylabel('Attention Ratio', fontsize=FONT_SIZES["label"])
    plt.xticks(fontsize=FONT_SIZES["tick"])
    plt.yticks(fontsize=FONT_SIZES["tick"])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=FONT_SIZES["legend"])
    
    # Add value labels on points
    for i, (size, mean_val) in enumerate(zip(template_sizes, mean_ratios)):
        plt.annotate(f'{mean_val:.3f}', (size, mean_val), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=FONT_SIZES["annotation"])
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_ratio_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trend plot saved to: {output_path}")
    
    plt.close()
def create_distribution_plot(all_results, output_dir):
    """Create distribution plot showing histograms for each template"""
    if not all_results:
        print("No data to plot")
        return
    
    # Sort templates by size
    sorted_templates = sorted(all_results.keys(), key=extract_template_size)
    
    # Create subplots
    n_templates = len(sorted_templates)
    cols = min(3, n_templates)
    rows = (n_templates + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if n_templates == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        # When we have one row but multiple columns, axes is already an array
        pass
    elif rows > 1 and cols > 1:
        axes = axes.flatten()
    else:
        # Single subplot case is handled by n_templates == 1
        pass
    
    for i, template_name in enumerate(sorted_templates):
        data = all_results[template_name]
        results = data['results']
        ratios = [r['overall_ratio'] for r in results if r['overall_ratio'] is not None and not np.isinf(r['overall_ratio'])]
        
        # Cap extreme values for visualization
        capped_ratios = [min(max(r, 0), 10) for r in ratios]
        
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
        
        # Create histogram
        ax.hist(capped_ratios, bins=20, alpha=0.7, color=plt.cm.viridis(i/n_templates), edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(ratios)
        median_val = np.median(ratios)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_val:.3f}')
        
        ax.set_title(f'{template_name} ({extract_template_size(template_name)}k tokens)\nn={len(ratios)}',
                     fontsize=FONT_SIZES["title"])
        ax.set_xlabel('Attention Ratio', fontsize=FONT_SIZES["label"])
        ax.set_ylabel('Frequency', fontsize=FONT_SIZES["label"])
        ax.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_templates, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Attention Ratio Distributions by Template Length', fontsize=FONT_SIZES["title"])
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_ratio_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_path}")
    
    plt.close()

def create_summary_statistics(all_results, output_dir):
    """Create and save summary statistics"""
    if not all_results:
        print("No data for statistics")
        return
    
    # Prepare summary data
    summary_data = []
    
    for template_name, data in all_results.items():
        results = data['results']
        ratios = [r['overall_ratio'] for r in results if r['overall_ratio'] is not None and not np.isinf(r['overall_ratio'])]
        if ratios:
            stats = {
                'Template': template_name,
                'Size_k': extract_template_size(template_name),
                'Sample_Count': len(ratios),
                'Success_Rate': f"{len(ratios)}/{data['total_count']} ({len(ratios)/data['total_count']*100:.1f}%)",
                'Mean': float(np.mean(ratios)),
                'Median': float(np.median(ratios)),
                'Std': float(np.std(ratios)),
                'Min': float(np.min(ratios)),
                'Max': float(np.max(ratios)),
                'Q25': float(np.percentile(ratios, 25)),
                'Q75': float(np.percentile(ratios, 75)),
                'Model_Layers': data.get('num_layers', 0),
                'Model_Heads': data.get('num_heads', 0)
            }
            summary_data.append(stats)
    
    # Sort by template size
    summary_data.sort(key=lambda x: x['Size_k'])
    
    # Save to JSON
    output_path = os.path.join(output_dir, 'attention_ratio_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Summary statistics saved to: {output_path}")
    
    # Also create a readable text summary
    text_path = os.path.join(output_dir, 'attention_ratio_summary.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("Jailbreak Attention Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for stats in summary_data:
            f.write(f"Template: {stats['Template']} ({stats['Size_k']}k tokens)\n")
            f.write(f"  Samples: {stats['Success_Rate']}\n")
            f.write(f"  Mean ratio: {stats['Mean']:.4f} ± {stats['Std']:.4f}\n")
            f.write(f"  Median ratio: {stats['Median']:.4f}\n")
            f.write(f"  Range: [{stats['Min']:.4f}, {stats['Max']:.4f}]\n")
            f.write(f"  IQR: [{stats['Q25']:.4f}, {stats['Q75']:.4f}]\n")
            f.write("\n")
        
        # Add trend analysis
        if len(summary_data) > 1:
            sizes = [s['Size_k'] for s in summary_data]
            means = [s['Mean'] for s in summary_data]
            
            # Calculate correlation
            correlation = np.corrcoef(sizes, means)[0, 1]
            
            f.write("Trend Analysis:\n")
            f.write(f"  Correlation (size vs mean ratio): {correlation:.4f}\n")
            
            if correlation > 0.5:
                f.write("  Strong positive correlation - attention ratio increases with template length\n")
            elif correlation > 0.2:
                f.write("  Moderate positive correlation - attention ratio somewhat increases with template length\n")
            elif correlation > -0.2:
                f.write("  Weak correlation - no clear trend with template length\n")
            elif correlation > -0.5:
                f.write("  Moderate negative correlation - attention ratio decreases with template length\n")
            else:
                f.write("  Strong negative correlation - attention ratio decreases with template length\n")
    
    print(f"Summary text saved to: {text_path}")

def create_layer_analysis_plot(all_results, output_dir):
    """Create visualization showing attention ratio trends across different layers"""
    if not all_results:
        print("No data to plot")
        return
    
    # Only analyze if we have multiple templates to compare
    if len(all_results) < 2:
        print("Need at least 2 templates for layer analysis")
        return
    
    # Get layer count from first template
    first_template = next(iter(all_results.values()))
    num_layers = first_template.get('num_layers', 0)
    
    if num_layers == 0:
        print("No layer information available")
        return
    
    # Sort templates by size
    sorted_templates = sorted(all_results.keys(), key=extract_template_size)
    
    # Create simplified single plot layout
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot: Mean ratio by layer for each template (only line plots, no error bars)
    layer_indices = np.arange(num_layers)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_templates)))
    
    for idx, template_name in enumerate(sorted_templates):
        data = all_results[template_name]
        results = data['results']
        
        # Calculate mean ratio for each layer across all samples and heads
        layer_means = []
        
        for layer_idx in range(num_layers):
            layer_ratios = []
            for result in results:
                ratios_by_layer_head = result['ratios_by_layer_head']
                # Get all head ratios for this layer
                layer_head_ratios = ratios_by_layer_head[layer_idx]
                # Filter out infinite values
                valid_ratios = [r for r in layer_head_ratios if not np.isinf(r)]
                if valid_ratios:
                    layer_ratios.extend(valid_ratios)
            
            if layer_ratios:
                layer_means.append(np.mean(layer_ratios))
            else:
                layer_means.append(0)
        
        size = extract_template_size(template_name)
        ax.plot(layer_indices, layer_means, 
               label=f'{size}k tokens', color=colors[idx], 
               marker='o', linewidth=2, markersize=6)
    
    ax.set_xlabel('Layer Index', fontsize=FONT_SIZES["label"])
    ax.set_ylabel('Mean Attention Ratio', fontsize=FONT_SIZES["label"])
    ax.set_title('Attention Ratio by Layer Across CoT Lengths', fontsize=FONT_SIZES["title"])
    ax.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
    ax.legend(fontsize=FONT_SIZES["legend"])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_layer_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Layer analysis plot saved to: {output_path}")
    
    plt.close()

def create_head_analysis_plot(all_results, output_dir):
    """Create visualization showing attention ratio trends across attention heads for each layer"""
    if not all_results:
        print("No data to plot")
        return
    
    # Get model architecture info
    first_template = next(iter(all_results.values()))
    num_heads = first_template.get('num_heads', 0)
    num_layers = first_template.get('num_layers', 0)
    
    if num_heads == 0 or num_layers == 0:
        print("No head/layer information available")
        return
    
    # Create subdirectory for layer-wise head analysis plots
    head_analysis_dir = os.path.join(output_dir, 'head_analysis_by_layer')
    os.makedirs(head_analysis_dir, exist_ok=True)
    
    # Sort templates by size
    sorted_templates = sorted(all_results.keys(), key=extract_template_size)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_templates)))
    
    # Create separate plot for each layer
    for layer_idx in range(num_layers):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        head_indices = np.arange(num_heads)
        
        # Plot each template for this layer
        for template_idx, template_name in enumerate(sorted_templates):
            data = all_results[template_name]
            results = data['results']
            
            # Calculate mean ratio for each head in this specific layer
            head_means = []
            
            for head_idx in range(num_heads):
                head_ratios = []
                for result in results:
                    ratios_by_layer_head = result['ratios_by_layer_head']
                    # Get ratio for this specific head and layer
                    ratio = ratios_by_layer_head[layer_idx][head_idx]
                    if not np.isinf(ratio):  # Only filter out infinite values
                        head_ratios.append(ratio)
                
                if head_ratios:
                    head_means.append(np.mean(head_ratios))
                else:
                    head_means.append(0)
            
            size = extract_template_size(template_name)
            ax.plot(head_indices, head_means, 
                   label=f'{size}k tokens', color=colors[template_idx], 
                   marker='o', linewidth=2, markersize=6)
        
        ax.set_xlabel('Attention Head Index', fontsize=FONT_SIZES["label"])
        ax.set_ylabel('Mean Attention Ratio', fontsize=FONT_SIZES["label"])
        ax.set_title(f'Attention Ratio by Head - Layer {layer_idx}', fontsize=FONT_SIZES["title"])
        ax.tick_params(axis='both', labelsize=FONT_SIZES["tick"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot for this layer in the subdirectory
        output_path = os.path.join(head_analysis_dir, f'layer_{layer_idx:02d}_head_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Head analysis plots saved in {head_analysis_dir}/: {num_layers} plots for layers 0-{num_layers-1}")

def main():
    parser = argparse.ArgumentParser(description="Visualize jailbreak attention analysis results")
    parser.add_argument("--input-files", type=str, nargs='*', 
                       help="Paths to .pt or JSON result files from extract_jailbreak_attention.py. If not provided, will auto-discover files in --jailbreak-results-dir")
    parser.add_argument("--jailbreak-results-dir", type=str, default="jailbreak_results",
                       help="Directory to search for jailbreak attention result files when --input-files not provided (default: jailbreak_results)")
    parser.add_argument("--output-dir", type=str, default="jailbreak_attention_visualizations2",
                       help="Output directory for visualization files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Determine input files
    if args.input_files:
        input_files = args.input_files
        print(f"Using provided input files: {input_files}")
    else:
        print(f"Auto-discovering jailbreak result files in: {args.jailbreak_results_dir}")
        input_files = auto_discover_jailbreak_files(args.jailbreak_results_dir)
        if not input_files:
            print(f"No jailbreak attention result files found in {args.jailbreak_results_dir}")
            print("Please either:")
            print(f"  1. Place result files in the {args.jailbreak_results_dir} directory, or")
            print("  2. Specify files manually with --input-files")
            return
        print(f"Discovered files: {input_files}")
    
    # Load results
    print("Loading results from files...")
    all_results = load_results(input_files)
    
    if not all_results:
        print("No valid data loaded. Please check input files.")
        return
    
    print(f"Loaded data for {len(all_results)} templates")
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create visualizations
    print("Creating visualizations...")
    
    create_box_plot(all_results, args.output_dir)
    create_trend_plot(all_results, args.output_dir)
    create_distribution_plot(all_results, args.output_dir)
    create_layer_analysis_plot(all_results, args.output_dir)
    create_head_analysis_plot(all_results, args.output_dir)
    create_summary_statistics(all_results, args.output_dir)
    
    print(f"\nVisualization complete! Files saved to: {args.output_dir}")
    print("Generated files:")
    print("  - attention_ratio_boxplot.png: Box plots showing distribution across templates")
    print("  - attention_ratio_trend.png: Trend line showing mean/median vs template size")
    print("  - attention_ratio_distributions.png: Histograms for each template")
    print("  - attention_layer_analysis.png: Layer-wise attention analysis")
    print("  - head_analysis_by_layer/: Directory containing per-layer attention head analysis plots")
    print("    └── layer_XX_head_analysis.png: One plot per layer showing head ratios")
    print("  - attention_ratio_summary.json: Detailed statistics in JSON format")
    print("  - attention_ratio_summary.txt: Human-readable summary with trend analysis")
    print("\nNote: For full functionality (layer and head analysis), use .pt files generated by extract_jailbreak_attention.py")

if __name__ == "__main__":
    main()
