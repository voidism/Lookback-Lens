import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from pathlib import Path

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
              fontsize=14, pad=20)
    plt.xlabel('Template Length (thousands of tokens)', fontsize=12)
    plt.ylabel('Attention Ratio', fontsize=12)
    
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
    
    plt.figtext(0.02, 0.02, '\n'.join(stats_text), fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_ratio_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, 'attention_ratio_boxplot.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Box plot (PDF) saved to: {pdf_path}")
    
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
    median_ratios = []
    template_names = []
    
    for template_name, data in all_results.items():
        results = data['results']
        ratios = [r['overall_ratio'] for r in results if r['overall_ratio'] is not None and not np.isinf(r['overall_ratio'])]
        if ratios:
            template_sizes.append(extract_template_size(template_name))
            mean_ratios.append(np.mean(ratios))
            std_ratios.append(np.std(ratios))
            median_ratios.append(np.median(ratios))
            template_names.append(template_name)
    
    # Sort by template size
    sorted_indices = np.argsort(template_sizes)
    template_sizes = [template_sizes[i] for i in sorted_indices]
    mean_ratios = [mean_ratios[i] for i in sorted_indices]
    std_ratios = [std_ratios[i] for i in sorted_indices]
    median_ratios = [median_ratios[i] for i in sorted_indices]
    template_names = [template_names[i] for i in sorted_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot mean with error bars
    plt.errorbar(template_sizes, mean_ratios, yerr=std_ratios, 
                marker='o', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='Mean ± Std', color='blue')
    
    # Plot median
    plt.plot(template_sizes, median_ratios, 
            marker='s', linewidth=2, markersize=6, 
            label='Median', color='red', linestyle='--')
    
    plt.title('Attention Ratio Trend Across Template Lengths\n(Part 3 Attention / Other User Prompt Attention)', 
              fontsize=14, pad=20)
    plt.xlabel('Template Length (thousands of tokens)', fontsize=12)
    plt.ylabel('Attention Ratio', fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Add value labels on points
    for i, (size, mean_val, median_val) in enumerate(zip(template_sizes, mean_ratios, median_ratios)):
        plt.annotate(f'{mean_val:.3f}', (size, mean_val), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        plt.annotate(f'{median_val:.3f}', (size, median_val), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_ratio_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trend plot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, 'attention_ratio_trend.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Trend plot (PDF) saved to: {pdf_path}")
    
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
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
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
        
        ax.set_title(f'{template_name} ({extract_template_size(template_name)}k tokens)\nn={len(ratios)}')
        ax.set_xlabel('Attention Ratio')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_templates, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Attention Ratio Distributions by Template Length', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_ratio_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, 'attention_ratio_distributions.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Distribution plot (PDF) saved to: {pdf_path}")
    
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
    
    # Create layer-wise comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Mean ratio by layer for each template
    layer_indices = np.arange(num_layers)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_templates)))
    
    for idx, template_name in enumerate(sorted_templates):
        data = all_results[template_name]
        results = data['results']
        
        # Calculate mean ratio for each layer across all samples and heads
        layer_means = []
        layer_stds = []
        
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
                layer_stds.append(np.std(layer_ratios))
            else:
                layer_means.append(0)
                layer_stds.append(0)
        
        size = extract_template_size(template_name)
        ax1.errorbar(layer_indices, layer_means, yerr=layer_stds, 
                    label=f'{size}k tokens', color=colors[idx], 
                    marker='o', capsize=3, capthick=1)
    
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Mean Attention Ratio', fontsize=12)
    ax1.set_title('Attention Ratio by Layer Across Template Lengths', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Layer-wise ratio change (last layer / first layer) for each template
    template_sizes = []
    ratio_changes = []
    
    for template_name in sorted_templates:
        data = all_results[template_name]
        results = data['results']
        
        # Calculate ratio change from first to last layer
        first_layer_ratios = []
        last_layer_ratios = []
        
        for result in results:
            ratios_by_layer_head = result['ratios_by_layer_head']
            
            # First layer ratios
            first_layer = ratios_by_layer_head[0]
            valid_first = [r for r in first_layer if not np.isinf(r)]
            if valid_first:
                first_layer_ratios.extend(valid_first)
            
            # Last layer ratios
            last_layer = ratios_by_layer_head[-1]
            valid_last = [r for r in last_layer if not np.isinf(r)]
            if valid_last:
                last_layer_ratios.extend(valid_last)
        
        if first_layer_ratios and last_layer_ratios:
            mean_first = np.mean(first_layer_ratios)
            mean_last = np.mean(last_layer_ratios)
            change = mean_last / mean_first if mean_first > 0 else 1
            
            template_sizes.append(extract_template_size(template_name))
            ratio_changes.append(change)
    
    if template_sizes and ratio_changes:
        ax2.bar(range(len(template_sizes)), ratio_changes, 
               color=colors[:len(template_sizes)], alpha=0.7)
        ax2.set_xticks(range(len(template_sizes)))
        ax2.set_xticklabels([f'{size}k' for size in template_sizes])
        ax2.set_xlabel('Template Length', fontsize=12)
        ax2.set_ylabel('Ratio Change (Last/First Layer)', fontsize=12)
        ax2.set_title('Attention Ratio Change from First to Last Layer', fontsize=14)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No change')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_layer_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Layer analysis plot saved to: {output_path}")
    
    pdf_path = os.path.join(output_dir, 'attention_layer_analysis.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Layer analysis plot (PDF) saved to: {pdf_path}")
    
    plt.close()

def create_head_analysis_plot(all_results, output_dir):
    """Create visualization showing attention ratio distribution across attention heads"""
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
    
    # Sort templates by size
    sorted_templates = sorted(all_results.keys(), key=extract_template_size)
    
    # Create head analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Head-wise attention ratio distribution (box plot)
    ax1 = axes[0, 0]
    head_data = []
    head_labels = []
    
    # Aggregate data across all templates for each head
    for head_idx in range(min(num_heads, 16)):  # Limit to first 16 heads for clarity
        head_ratios = []
        for template_name in sorted_templates:
            data = all_results[template_name]
            results = data['results']
            
            for result in results:
                ratios_by_layer_head = result['ratios_by_layer_head']
                # Get ratios for this head across all layers
                for layer_idx in range(num_layers):
                    ratio = ratios_by_layer_head[layer_idx][head_idx]
                    if not np.isinf(ratio) and ratio <= 10:  # Cap extreme values
                        head_ratios.append(ratio)
        
        if head_ratios:
            head_data.append(head_ratios)
            head_labels.append(f'H{head_idx}')
    
    if head_data:
        ax1.boxplot(head_data[:8], tick_labels=head_labels[:8])  # Show first 8 heads
        ax1.set_title('Attention Ratio Distribution by Head (First 8 Heads)', fontsize=12)
        ax1.set_xlabel('Attention Head')
        ax1.set_ylabel('Attention Ratio')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Head variance across templates
    ax2 = axes[0, 1]
    template_sizes = []
    head_variances = []
    
    for template_name in sorted_templates:
        data = all_results[template_name]
        results = data['results']
        
        # Calculate variance in head attention ratios
        all_head_ratios = []
        for result in results:
            ratios_by_layer_head = result['ratios_by_layer_head']
            for layer_ratios in ratios_by_layer_head:
                valid_ratios = [r for r in layer_ratios if not np.isinf(r) and r <= 10]
                all_head_ratios.extend(valid_ratios)
        
        if all_head_ratios:
            variance = np.var(all_head_ratios)
            template_sizes.append(extract_template_size(template_name))
            head_variances.append(variance)
    
    if template_sizes and head_variances:
        ax2.bar(range(len(template_sizes)), head_variances, alpha=0.7)
        ax2.set_xticks(range(len(template_sizes)))
        ax2.set_xticklabels([f'{size}k' for size in template_sizes])
        ax2.set_title('Attention Head Variance by Template Length', fontsize=12)
        ax2.set_xlabel('Template Length')
        ax2.set_ylabel('Attention Ratio Variance')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Head attention heatmap for one template
    ax3 = axes[1, 0]
    if sorted_templates:
        # Use middle template for heatmap
        mid_template = sorted_templates[len(sorted_templates)//2]
        data = all_results[mid_template]
        results = data['results']
        
        # Create average attention matrix [layers x heads]
        avg_ratios = np.zeros((num_layers, num_heads))
        counts = np.zeros((num_layers, num_heads))
        
        for result in results:
            ratios_by_layer_head = result['ratios_by_layer_head']
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    ratio = ratios_by_layer_head[layer_idx][head_idx]
                    if not np.isinf(ratio):
                        avg_ratios[layer_idx, head_idx] += ratio
                        counts[layer_idx, head_idx] += 1
        
        # Calculate averages
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_ratios = np.divide(avg_ratios, counts, 
                                 out=np.zeros_like(avg_ratios), 
                                 where=(counts != 0))
        
        # Cap values for visualization
        avg_ratios = np.clip(avg_ratios, 0, 5)
        
        im = ax3.imshow(avg_ratios, cmap='viridis', aspect='auto')
        ax3.set_title(f'Attention Ratio Heatmap - {extract_template_size(mid_template)}k Template', fontsize=12)
        ax3.set_xlabel('Attention Head')
        ax3.set_ylabel('Layer')
        
        # Add colorbar
        plt.colorbar(im, ax=ax3, label='Attention Ratio')
    
    # Plot 4: Head consistency across templates
    ax4 = axes[1, 1]
    if len(sorted_templates) > 1:
        head_consistency = []
        
        for head_idx in range(min(num_heads, 8)):  # First 8 heads
            template_means = []
            for template_name in sorted_templates:
                data = all_results[template_name]
                results = data['results']
                
                head_ratios = []
                for result in results:
                    ratios_by_layer_head = result['ratios_by_layer_head']
                    for layer_ratios in ratios_by_layer_head:
                        ratio = layer_ratios[head_idx]
                        if not np.isinf(ratio):
                            head_ratios.append(ratio)
                
                if head_ratios:
                    template_means.append(np.mean(head_ratios))
            
            if len(template_means) > 1:
                consistency = np.std(template_means)
                head_consistency.append(consistency)
            else:
                head_consistency.append(0)
        
        if head_consistency:
            ax4.bar(range(len(head_consistency)), head_consistency, alpha=0.7)
            ax4.set_xticks(range(len(head_consistency)))
            ax4.set_xticklabels([f'H{i}' for i in range(len(head_consistency))])
            ax4.set_title('Head Consistency Across Templates (Lower = More Consistent)', fontsize=12)
            ax4.set_xlabel('Attention Head')
            ax4.set_ylabel('Standard Deviation of Means')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'attention_head_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Head analysis plot saved to: {output_path}")
    
    pdf_path = os.path.join(output_dir, 'attention_head_analysis.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Head analysis plot (PDF) saved to: {pdf_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize jailbreak attention analysis results")
    parser.add_argument("--input-files", type=str, nargs='+', required=True,
                       help="Paths to .pt or JSON result files from extract_jailbreak_attention.py")
    parser.add_argument("--output-dir", type=str, default="jailbreak_attention_visualizations",
                       help="Output directory for visualization files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load results
    print("Loading results from files...")
    all_results = load_results(args.input_files)
    
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
    print("  - attention_ratio_boxplot.png/pdf: Box plots showing distribution across templates")
    print("  - attention_ratio_trend.png/pdf: Trend line showing mean/median vs template size")
    print("  - attention_ratio_distributions.png/pdf: Histograms for each template")
    print("  - attention_layer_analysis.png/pdf: Layer-wise attention analysis")
    print("  - attention_head_analysis.png/pdf: Attention head analysis and heatmaps")
    print("  - attention_ratio_summary.json: Detailed statistics in JSON format")
    print("  - attention_ratio_summary.txt: Human-readable summary with trend analysis")
    print("\nNote: For full functionality (layer and head analysis), use .pt files generated by extract_jailbreak_attention.py")

if __name__ == "__main__":
    main()