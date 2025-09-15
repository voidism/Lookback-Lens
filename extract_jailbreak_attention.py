# Set environment variables BEFORE importing transformers
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp'

import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm
import argparse
import gc
import datetime
import importlib.util
import sys

transformers.logging.set_verbosity(40)

def set_mirror_environment():
    """Set environment variables for using Alibaba Cloud mirror"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp'
    print("Set HF_ENDPOINT to https://hf-mirror.com")
    print("Set cache directory to /root/autodl-tmp")

def load_jailbreakbench_data(file_path, num_samples):
    """Load jailbreakbench.json data"""
    print(f"Loading jailbreakbench data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total instructions in file: {len(data)}")
    selected_data = data[:num_samples]
    print(f"Selected {len(selected_data)} instructions for processing")
    
    return selected_data

def load_template(template_name):
    """Dynamically load template file and extract format_thinking_template"""
    template_path = f"template/{template_name}.py"
    
    if not os.path.exists(template_path):
        raise ValueError(f"Template file {template_path} does not exist")
    
    # Load the template module dynamically
    spec = importlib.util.spec_from_file_location("template_module", template_path)
    template_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template_module)
    
    if not hasattr(template_module, 'format_thinking_template'):
        raise ValueError(f"Template file {template_path} does not contain 'format_thinking_template'")
    
    print(f"Successfully loaded template: {template_name}")
    template_content = template_module.format_thinking_template
    print(f"Template content length: {len(template_content)} characters")
    
    return template_content

def create_full_text(template_content, instruction):
    """Create full text by formatting template with instruction"""
    try:
        full_text = template_content.format(instruction=instruction)
        return full_text
    except KeyError as e:
        raise ValueError(f"Template formatting failed: {e}")

def find_user_prompt_tokens(full_text, tokenizer):
    """Find token positions for user prompt range (<|im_start|>user to <|im_end|>)"""
    user_start_marker = "<|im_start|>user"
    user_end_marker = "<|im_end|>"
    
    # Find text positions
    user_start_pos = full_text.find(user_start_marker)
    if user_start_pos == -1:
        raise ValueError("Could not find <|im_start|>user marker")
    
    # Find the first <|im_end|> after user start
    user_end_pos = full_text.find(user_end_marker, user_start_pos)
    if user_end_pos == -1:
        raise ValueError("Could not find <|im_end|> marker after user start")
    
    user_end_pos += len(user_end_marker)  # Include the end marker
    
    # Extract user prompt text
    user_prompt_text = full_text[user_start_pos:user_end_pos]
    
    # Calculate token positions
    prefix_text = full_text[:user_start_pos]
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    user_prompt_tokens = tokenizer.encode(user_prompt_text, add_special_tokens=False)
    
    user_start_token = len(prefix_tokens)
    user_end_token = user_start_token + len(user_prompt_tokens)
    
    print(f"User prompt text length: {len(user_prompt_text)} characters")
    print(f"User prompt token range: [{user_start_token}, {user_end_token})")
    print(f"User prompt tokens: {len(user_prompt_tokens)}")
    
    return user_start_token, user_end_token, user_prompt_text

def find_part3_tokens(user_prompt_text, instruction, tokenizer, user_start_token):
    """Find token positions for 'part 3: {instruction}' within user prompt"""
    
    # First find the position of "part 3:" (with various possible formats)
    part3_prefixes = ["part 3:", "part3:", "part 3:", "part3:"]
    part3_pos = -1
    matched_prefix = None
    
    for prefix in part3_prefixes:
        part3_pos = user_prompt_text.find(prefix)
        if part3_pos != -1:
            matched_prefix = prefix
            break
    
    if part3_pos == -1:
        print(f"Warning: Could not find any 'part 3:' variant in user prompt")
        return None, None
    
    # Calculate where the instruction should start
    instruction_start_pos = part3_pos + len(matched_prefix)
    
    # Skip any whitespace or newlines after the prefix
    while instruction_start_pos < len(user_prompt_text) and user_prompt_text[instruction_start_pos] in [' ', '\n', '\r', '\t']:
        instruction_start_pos += 1
    
    # Verify that the instruction matches what follows
    remaining_text = user_prompt_text[instruction_start_pos:]
    
    # The instruction might end with <|im_end|>, so we need to check up to the instruction length
    if not remaining_text.startswith(instruction):
        print(f"Warning: Found '{matched_prefix}' but instruction doesn't match")
        print(f"Expected: {instruction[:100]}...")
        print(f"Found: {remaining_text[:100]}...")
        print(f"Full remaining text: {remaining_text[:200]}...")
        return None, None
    
    # Calculate full part 3 content boundaries
    part3_content_start = part3_pos
    part3_content_end = instruction_start_pos + len(instruction)
    
    # Calculate token positions within user prompt
    part3_prefix_text = user_prompt_text[:part3_content_start]
    part3_text = user_prompt_text[part3_content_start:part3_content_end]
    
    part3_prefix_tokens = tokenizer.encode(part3_prefix_text, add_special_tokens=False)
    part3_tokens = tokenizer.encode(part3_text, add_special_tokens=False)
    
    # Convert to absolute token positions in full sequence
    part3_start_token = user_start_token + len(part3_prefix_tokens)
    part3_end_token = part3_start_token + len(part3_tokens)
    
    print(f"Part 3 prefix: '{matched_prefix}'")
    print(f"Part 3 full content: '{part3_text[:100]}...'")
    print(f"Part 3 token range: [{part3_start_token}, {part3_end_token})")
    print(f"Part 3 tokens: {len(part3_tokens)}")
    
    return part3_start_token, part3_end_token

class QwenLLM:
    def __init__(self, model_name="Qwen/Qwen3-14B", device="cuda", num_gpus="auto", max_memory=40):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_memory = max_memory
        
        print(f"Loading model: {model_name}")
        self.model, self.tokenizer = self.load_model()
        
    def load_model(self):
        """Load Qwen3 model and tokenizer"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        if self.device == "cuda":
            kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto" if self.num_gpus == "auto" else None
            }
            
            if self.num_gpus != "auto":
                num_gpus = int(self.num_gpus)
                if num_gpus > 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_memory}GiB" for i in range(num_gpus)},
                    })
        else:
            kwargs = {}
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation="eager",  # Required for attention extraction
            **kwargs
        )
        
        if self.device == "cuda" and self.num_gpus == "1":
            model.cuda()
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Enable attention output
        model.config.output_attentions = True
        print(f"Attention extraction enabled. Model attention implementation: eager")
            
        return model, tokenizer
    
    def extract_last_token_attention(self, full_text):
        """
        Extract attention weights for the last token using optimized stage-2 approach
        Similar to extract_attentions.py - only compute attention for the last token position
        """
        print("Extracting attention for last token using optimized stage-2 approach")
        
        with torch.no_grad():
            # Tokenize full text
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            seq_len = inputs.input_ids.shape[-1]
            print(f"Full sequence length: {seq_len} tokens")
            
            # Define target range: only last token (memory optimization)
            last_token_pos = seq_len - 1
            target_range = (last_token_pos, seq_len)  # [start, end)
            
            print(f"Memory optimization: computing attention only for last token (position {last_token_pos})")
            print(f"Reduced from O(n²)={seq_len}² to O(1×n)=1×{seq_len} - saving ~{seq_len-1}x memory")
            
            # Stage 2: Forward pass to get attention with target range optimization
            outputs = self.model(
                **inputs,
                output_attentions=True,
                use_cache=True,
                target_field_range=target_range  # Key optimization: only compute last token attention
            )
            
            if outputs.attentions is None:
                print("Warning: No attentions returned from forward pass")
                return None
            
            # Process attention outputs with target_field_range optimization
            last_token_attentions = []
            num_layers = len(outputs.attentions)
            print(f"Processing {num_layers} attention layers")
            
            for layer_idx in range(num_layers):
                # With target_field_range, only target field attention weights are returned
                # Shape: [batch, heads, target_len, seq_len] where target_len=1 for last token
                layer_attn = outputs.attentions[layer_idx]  # [batch, heads, 1, seq_len]
                
                # Remove batch dimension and squeeze target token dimension
                layer_attn = layer_attn[0]  # Remove batch: [heads, 1, seq_len]
                last_token_attn = layer_attn[:, 0, :]  # [heads, seq_len] - squeeze the single token dim
                
                last_token_attentions.append(last_token_attn.cpu())
            
            print(f"Successfully extracted last token attention for {len(last_token_attentions)} layers")
            if last_token_attentions:
                print(f"Attention shape per layer: {last_token_attentions[0].shape} [heads, seq_len]")
            
            # Clean memory immediately
            torch.cuda.empty_cache()
            
            return last_token_attentions
                

def extract_attention_data(last_token_attentions, user_start_token, user_end_token, 
                          part3_start_token, part3_end_token):
    """Extract detailed attention data for analysis"""
    if last_token_attentions is None:
        return None, None
    
    # Stack all layer attentions: [layers, heads, seq_len]
    all_attentions = torch.stack(last_token_attentions)
    num_layers, num_heads, seq_len = all_attentions.shape
    
    print(f"Attention data shape: {num_layers} layers, {num_heads} heads, {seq_len} tokens")
    
    # Extract user prompt attention for all layers and heads
    user_prompt_attentions = all_attentions[:, :, user_start_token:user_end_token]  # [layers, heads, user_prompt_len]
    
    # Calculate relative positions within user prompt
    part3_rel_start = part3_start_token - user_start_token
    part3_rel_end = part3_end_token - user_start_token
    
    # Extract part3 and other attention weights for each layer and head
    part3_attentions = user_prompt_attentions[:, :, part3_rel_start:part3_rel_end]  # [layers, heads, part3_len]
    
    # Create mask for other parts
    user_prompt_len = user_prompt_attentions.shape[2]
    other_mask = torch.ones(user_prompt_len, dtype=torch.bool)
    other_mask[part3_rel_start:part3_rel_end] = False
    
    other_attentions = user_prompt_attentions[:, :, other_mask]  # [layers, heads, other_len]
    
    # Calculate attention sums for each layer and head
    part3_sums = part3_attentions.sum(dim=2)  # [layers, heads]
    other_sums = other_attentions.sum(dim=2)  # [layers, heads]
    
    # Prepare tensor data for .pt file
    attention_tensors = {
        'part3_attention_by_layer_head': part3_sums,  # [layers, heads] - keep as tensor
        'other_attention_by_layer_head': other_sums,  # [layers, heads] - keep as tensor
        'part3_raw_attentions': part3_attentions,     # [layers, heads, part3_len] - full detail
        'other_raw_attentions': other_attentions      # [layers, heads, other_len] - full detail
    }
    
    # Calculate overall statistics for summary
    total_part3 = part3_sums.sum().item()
    total_other = other_sums.sum().item()
    avg_ratio = total_part3 / total_other if total_other > 0 else float('inf')
    
    print(f"Part 3 total attention: {total_part3:.6f}")
    print(f"Other user prompt total attention: {total_other:.6f}")
    print(f"Overall attention ratio: {avg_ratio:.6f}")
    
    # Prepare metadata (JSON-serializable)
    attention_metadata = {
        'num_layers': num_layers,
        'num_heads': num_heads,
        'user_prompt_length': user_prompt_len,
        'part3_length': part3_rel_end - part3_rel_start,
        'part3_start_rel': part3_rel_start,
        'part3_end_rel': part3_rel_end,
        'summary': {
            'total_part3_attention': total_part3,
            'total_other_attention': total_other,
            'overall_ratio': avg_ratio
        }
    }
    
    return attention_metadata, attention_tensors

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

def save_results(results, output_path):
    """Save results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

def clean_memory():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Extract jailbreak attention analysis")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--num-gpus", type=str, default="auto")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--template-name", type=str, default='template_4k',
                       help="Template name (e.g., template_1k, template_3k)")
    parser.add_argument("--data-path", type=str, default="data/jailbreakbench.json")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of jailbreak instructions to process")
    parser.add_argument("--output-dir", type=str, default="jailbreak_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-memory", type=int, default=40)
    
    args = parser.parse_args()
    
    # Set environment for mirror
    set_mirror_environment()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Extract template size for filename
    template_size = extract_template_size(args.template_name)
    output_filename = f"jailbreak_attention_{template_size}k_n{args.num_samples}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Also prepare .pt file for attention tensors
    pt_filename = f"jailbreak_attention_{template_size}k_n{args.num_samples}.pt"
    pt_output_path = os.path.join(args.output_dir, pt_filename)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data file {args.data_path} does not exist.")
    
    # Load jailbreakbench data
    jailbreak_data = load_jailbreakbench_data(args.data_path, args.num_samples)
    
    # Load template
    template_content = load_template(args.template_name)
    
    # Initialize model
    llm = QwenLLM(
        model_name=args.model_name,
        device=args.device,
        num_gpus=args.num_gpus,
        max_memory=args.max_memory
    )
    
    # Process each instruction
    results = {
        'template_name': args.template_name,
        'model_name': args.model_name,
        'num_samples': args.num_samples,
        'processing_time': datetime.datetime.now().isoformat(),
        'results': []
    }
    
    # Prepare container for attention tensors
    all_attention_tensors = []
    
    success_count = 0
    failed_count = 0
    
    for idx in tqdm(range(len(jailbreak_data)), desc="Processing instructions"):
        instruction_data = jailbreak_data[idx]
        instruction = instruction_data['instruction']
        
        try:
            print(f"\n=== Processing instruction {idx+1}/{len(jailbreak_data)} ===")
            print(f"Instruction: {instruction[:100]}...")
            
            # Create full text
            full_text = create_full_text(template_content, instruction)
            print(f"Full text length: {len(full_text)} characters")
            
            # Find user prompt token positions
            user_start_token, user_end_token, user_prompt_text = find_user_prompt_tokens(
                full_text, llm.tokenizer
            )
            
            # Find part 3 token positions
            part3_start_token, part3_end_token = find_part3_tokens(
                user_prompt_text, instruction, llm.tokenizer, user_start_token
            )
            
            if part3_start_token is None:
                print(f"Skipping instruction {idx}: Could not locate part 3")
                failed_count += 1
                continue
            
            # Extract last token attention
            last_token_attentions = llm.extract_last_token_attention(full_text)
            
            if last_token_attentions is None:
                print(f"Skipping instruction {idx}: Failed to extract attention")
                failed_count += 1
                continue
            
            # Extract detailed attention data
            attention_metadata, attention_tensors = extract_attention_data(
                last_token_attentions, user_start_token, user_end_token,
                part3_start_token, part3_end_token
            )
            
            if attention_metadata is None or attention_tensors is None:
                print(f"Skipping instruction {idx}: Failed to extract attention data")
                failed_count += 1
                continue
            
            # Store metadata results (for JSON)
            result_entry = {
                'instruction_idx': idx,
                'instruction': instruction,
                'category': instruction_data.get('category', 'Unknown'),
                'user_prompt_tokens': user_end_token - user_start_token,
                'part3_tokens': part3_end_token - part3_start_token,
                'attention_metadata': attention_metadata
            }
            
            results['results'].append(result_entry)
            
            # Store tensor data (for .pt file)
            tensor_entry = {
                'instruction_idx': idx,
                'instruction': instruction,
                'category': instruction_data.get('category', 'Unknown'),
                'attention_tensors': attention_tensors
            }
            
            all_attention_tensors.append(tensor_entry)
            success_count += 1
            
            # Clean memory immediately
            clean_memory()
            
        except Exception as e:
            print(f"Error processing instruction {idx}: {str(e)}")
            failed_count += 1
            clean_memory()
            continue
    
    # Calculate average attention ratios across all successful samples
    if results['results']:
        all_ratios = []
        total_part3_attention = 0.0
        total_other_attention = 0.0
        
        for result in results['results']:
            metadata = result['attention_metadata']
            summary = metadata['summary']
            
            ratio = summary['overall_ratio']
            if ratio != float('inf') and not np.isnan(ratio):  # Filter out invalid ratios
                all_ratios.append(ratio)
                total_part3_attention += summary['total_part3_attention']
                total_other_attention += summary['total_other_attention']
        
        # Calculate statistics
        if all_ratios:
            avg_ratio = np.mean(all_ratios)
            median_ratio = np.median(all_ratios)
            std_ratio = np.std(all_ratios)
            
            # Overall ratio across all samples
            overall_avg_ratio = total_part3_attention / total_other_attention if total_other_attention > 0 else float('inf')
            
            # Add average statistics to results
            results['average_statistics'] = {
                'mean_ratio': avg_ratio,
                'median_ratio': median_ratio,
                'std_ratio': std_ratio,
                'overall_ratio_across_samples': overall_avg_ratio,
                'total_part3_attention_sum': total_part3_attention,
                'total_other_attention_sum': total_other_attention,
                'valid_samples_for_ratio': len(all_ratios)
            }
            
            print(f"\n=== Average Statistics ===")
            print(f"Mean attention ratio: {avg_ratio:.6f}")
            print(f"Median attention ratio: {median_ratio:.6f}")
            print(f"Std attention ratio: {std_ratio:.6f}")
            print(f"Overall ratio across samples: {overall_avg_ratio:.6f}")
        else:
            results['average_statistics'] = {
                'mean_ratio': None,
                'median_ratio': None, 
                'std_ratio': None,
                'overall_ratio_across_samples': None,
                'valid_samples_for_ratio': 0,
                'note': 'No valid ratios found'
            }
    else:
        results['average_statistics'] = {
            'note': 'No successful samples to calculate averages'
        }

    # Save final results
    results['success_count'] = success_count
    results['failed_count'] = failed_count
    save_results(results, output_path)
    
    # Save attention tensors to .pt file
    if all_attention_tensors:
        torch.save({
            'template_name': args.template_name,
            'model_name': args.model_name,
            'num_samples': args.num_samples,
            'processing_time': datetime.datetime.now().isoformat(),
            'success_count': success_count,
            'failed_count': failed_count,
            'attention_data': all_attention_tensors
        }, pt_output_path)
        print(f"Attention tensors saved to: {pt_output_path}")
    
    print(f"\n=== Processing Complete ===")
    print(f"Template: {args.template_name} ({template_size}k tokens)")
    print(f"Successfully processed: {success_count}/{args.num_samples} instructions")
    print(f"Failed: {failed_count}")
    print(f"Metadata saved to: {output_path}")
    print(f"Attention tensors saved to: {pt_output_path}")

if __name__ == "__main__":
    main()