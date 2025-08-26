# Set environment variables BEFORE importing transformers
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp'

import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import argparse
import gc
import datetime
import shutil

transformers.logging.set_verbosity(40)

def set_mirror_environment():
    """Set environment variables for using Alibaba Cloud mirror"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_CACHE'] = '/root/autodl-tmp'
    print("Set HF_ENDPOINT to https://hf-mirror.com")
    print("Set cache directory to /root/autodl-tmp")

def load_wandb_gemini(file_path="data/wandb_gemini.csv", num_samples=10):
    """
    Load wandb_gemini CSV data and filter for conv_num=2
    """
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Filter rows where conv_num == 2
    filtered_df = df[df['conv_num'] == 2]
    print(f"Rows with conv_num=2: {len(filtered_df)}")
    
    # Take first num_samples
    selected_df = filtered_df.head(num_samples)
    print(f"Selected {len(selected_df)} samples")
    
    list_data_dict = []
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        new_item = {
            'context': row['prompt'],
            'data_index': idx,
            'original_index': row.name  # Keep original CSV index
        }
        list_data_dict.append(new_item)
    
    return list_data_dict

def extract_target_field(model_completion, tokenizer):
    """
    Extract the target field with smart selection logic:
    - If last ### field has <= 3 English words: use second-to-last ### field
    - Otherwise: use last ### field
    Returns: (target_field_text, field_start_token_idx, field_end_token_idx)
    """
    import re
    
    # Find all occurrences of exactly ### (not #### or more)
    hash_positions = []
    pos = 0
    while pos < len(model_completion):
        pos = model_completion.find('###', pos)
        if pos == -1:
            break
        
        # Check if this is exactly ### (not #### or more)
        # Check character after ### (if exists)
        if pos + 3 < len(model_completion) and model_completion[pos + 3] == '#':
            # This is #### or more, skip it
            pos += 4
            continue
        
        # Check character before ### (if exists) to ensure it's not part of ####
        if pos > 0 and model_completion[pos - 1] == '#':
            # This ### is part of #### or more, skip it
            pos += 3
            continue
            
        # This is exactly ###
        hash_positions.append(pos)
        pos += 3
    
    if len(hash_positions) == 0:
        print("No exact ### markers found in model completion (#### or more are ignored)")
        return None, None, None
    elif len(hash_positions) == 1:
        # Only one ### found, use it regardless of word count
        target_hash_pos = hash_positions[0]
        print("Only one exact ### marker found, using it")
    else:
        # Multiple ### found, apply smart selection logic
        # First, extract the last ### field to check word count
        last_hash_pos = hash_positions[-1]
        last_start_pos = last_hash_pos + 3
        last_end_pos = model_completion.find('\n', last_start_pos)
        if last_end_pos == -1:
            last_end_pos = len(model_completion)
        last_field = model_completion[last_start_pos:last_end_pos].strip()
        
        # Count English words in the last field
        english_words = re.findall(r'[a-zA-Z]+', last_field)
        word_count = len(english_words)
        
        print(f"Last field: '{last_field}' contains {word_count} English words")
        
        if word_count <= 3:
            # Use second-to-last ### field
            target_hash_pos = hash_positions[-2]
            print(f"Last field has <= 3 English words, using second-to-last ### field")
        else:
            # Use last ### field
            target_hash_pos = hash_positions[-1]
            print(f"Last field has > 3 English words, using last ### field")
    
    # Extract field content (after ### until newline)
    start_pos = target_hash_pos + 3  # Skip ###
    end_pos = model_completion.find('\n', start_pos)
    if end_pos == -1:
        end_pos = len(model_completion)
    
    target_field = model_completion[start_pos:end_pos].strip()
    if not target_field:
        print("Warning: Empty target field extracted")
        return None, None, None
    
    # Calculate token positions
    # Tokenize the text before the field
    prefix_text = model_completion[:start_pos]
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    field_tokens = tokenizer.encode(target_field, add_special_tokens=False)
    
    field_start_token = len(prefix_tokens)
    field_end_token = field_start_token + len(field_tokens)
    
    print(f"Extracted field: '{target_field}'")
    print(f"Field token range: [{field_start_token}, {field_end_token})")
    
    return target_field, field_start_token, field_end_token


def dump_jsonl(data, output_path, append=False):
    """Write data to JSONL file"""
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')

def create_qwen_prompt(context):
    """
    Create prompt for Qwen3 with thinking mode enabled
    """
    messages = [
        {"role": "user", "content": context}
    ]
    return messages

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
    
    def generate(self, messages, max_new_tokens=512, temperature=0.6, top_p=0.95, 
                 top_k=20, return_attentions=True, **kwargs):
        """
        Generate text with Qwen3 thinking mode
        """
        # Apply chat template with thinking mode enabled
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode
        )
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[-1]
        
        print(f"Input length: {input_length} tokens")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                output_attentions=False,  # 第一阶段：关闭attention输出节省显存
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode generated tokens
        generated_tokens = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"Generated {len(generated_tokens)} tokens")
        
        # 第一阶段完成：返回生成的文本，attention稍后获取
        if return_attentions:
            return generated_text, None, generated_tokens.cpu().numpy()
        else:
            return generated_text, generated_tokens.cpu().numpy()
    
    def extract_field_attentions(self, full_text, field_start_token, field_end_token):
        """
        第二阶段：针对target field进行前向传播获取attention
        """
        if field_start_token is None or field_end_token is None:
            print("Warning: Invalid field positions, skipping attention extraction")
            return None
            
        print(f"第二阶段：提取field attention (tokens {field_start_token}-{field_end_token})")
        
        with torch.no_grad():
            # 对完整序列进行一次前向传播
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            print(f"Full sequence length: {inputs.input_ids.shape[-1]} tokens")
            
            # 确保field位置在序列范围内
            seq_len = inputs.input_ids.shape[-1]
            if field_end_token > seq_len:
                print(f"Warning: Field end ({field_end_token}) exceeds sequence length ({seq_len})")
                field_end_token = seq_len
                
            if field_start_token >= field_end_token:
                print("Warning: Invalid field range after adjustment")
                return None
            
            try:
                print(f"使用优化的attention计算，只计算target field tokens的attention")
                print(f"显存优化：从O(n²)={seq_len}²降低到O(k×n)={field_end_token-field_start_token}×{seq_len}")
                
                # 前向传播获取attention，传递target_field_range实现显存优化
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    use_cache=True,
                    target_field_range=(field_start_token, field_end_token)  # 关键优化参数
                )
                
                if outputs.attentions is None:
                    print("Warning: No attentions returned from forward pass")
                    return None
                
                # 现在attentions已经是优化后的切片，形状为[heads, field_length, seq_len]
                field_attentions = []
                num_layers = len(outputs.attentions)
                print(f"Processing {num_layers} attention layers")
                
                for layer_idx in range(num_layers):
                    # attention已经是field相关的切片，无需再次切片
                    layer_attn = outputs.attentions[layer_idx][0]  # [heads, field_length, seq_len]
                    
                    # 如果只需要对之前tokens的attention，可以进一步切片
                    # field_slice = layer_attn[:, :, :field_end_token]  # [heads, field_length, prev_tokens]
                    field_slice = layer_attn  # 保留完整的field attention
                    
                    # 保持在GPU上且使用float32精度以获得最佳质量
                    # field_slice = field_slice.cpu().half()  # 原始压缩版本
                    field_attentions.append(field_slice)
                
                print(f"成功提取 {len(field_attentions)} 层的field attention")
                if field_attentions:
                    print(f"每层attention形状: {field_attentions[0].shape}")
                    expected_shape = f"[{field_attentions[0].shape[0]}, {field_end_token-field_start_token}, {seq_len}]"
                    print(f"预期形状: [num_heads, field_length, seq_len] = {expected_shape}")
                
                # 立即清理显存
                torch.cuda.empty_cache()
                
                return field_attentions
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM during attention extraction: {str(e)}")
                    print("尝试使用更低精度...")
                    torch.cuda.empty_cache()
                    return None
                else:
                    raise e

def setup_output_directory(output_dir, clear_output=False):
    """Setup output directory and handle existing files"""
    if os.path.exists(output_dir):
        if clear_output:
            print(f"Clearing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            print(f"Output directory already exists: {output_dir}")
            print("Use --clear-output to remove existing files")
    else:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return output_dir

def save_sample_results(sample_data, sample_idx, output_dir):
    """Save individual sample results immediately"""
    # Generate filenames
    pt_filename = f"sample_{sample_idx:03d}.pt"
    log_filename = f"sample_{sample_idx:03d}.log"
    
    pt_path = os.path.join(output_dir, pt_filename)
    log_path = os.path.join(output_dir, log_filename)
    
    # Save .pt file
    torch.save([sample_data], pt_path)
    print(f"Saved attention data: {pt_path}")
    
    # Save .log file
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== SAMPLE {sample_idx} ===\n")
        f.write(f"Data Index: {sample_data['data_index']}\n")
        f.write(f"Original Index: {sample_data['original_index']}\n")
        f.write(f"Generated Tokens: {len(sample_data['model_completion_ids'])}\n")
        f.write(f"Target Field: {sample_data['target_field']}\n")
        f.write(f"Field Start Token: {sample_data['field_start_token']}\n")
        f.write(f"Field End Token: {sample_data['field_end_token']}\n")
        f.write(f"Has Attention Data: {sample_data['field_attentions'] is not None}\n")
        f.write(f"Processing Time: {datetime.datetime.now().isoformat()}\n")
        f.write("=== CONTEXT ===\n")
        f.write(sample_data['context'])
        f.write("\n=== MODEL COMPLETION ===\n")
        f.write(sample_data['model_completion'])
        f.write("\n=== END SAMPLE ===\n")
    
    print(f"Saved text record: {log_path}")
    
    return pt_path, log_path

def save_extraction_summary(output_dir, success_count, failed_count, total_samples, config_args):
    """Save processing summary and configuration"""
    # Save summary
    summary_path = os.path.join(output_dir, "extraction_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Attention Extraction Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Successfully processed: {success_count}\n")
        f.write(f"Failed samples: {failed_count}\n")
        f.write(f"Success rate: {success_count/total_samples*100:.1f}%\n")
        f.write(f"Processing completed: {datetime.datetime.now().isoformat()}\n")
        f.write("\nGenerated files:\n")
        for i in range(success_count):
            f.write(f"  - sample_{i:03d}.pt\n")
            f.write(f"  - sample_{i:03d}.log\n")
    
    # Save configuration
    config_path = os.path.join(output_dir, "extraction_config.json")
    config_data = {
        'model_name': config_args.model_name,
        'num_samples': config_args.num_samples,
        'max_new_tokens': config_args.max_new_tokens,
        'temperature': config_args.temperature,
        'top_p': config_args.top_p,
        'top_k': config_args.top_k,
        'seed': config_args.seed,
        'data_path': config_args.data_path,
        'processing_time': datetime.datetime.now().isoformat(),
        'success_count': success_count,
        'failed_count': failed_count
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved processing summary: {summary_path}")
    print(f"Saved configuration: {config_path}")

def check_existing_samples(output_dir):
    """Check for existing sample files and return list of completed indices"""
    if not os.path.exists(output_dir):
        return []
    
    completed_samples = []
    for filename in os.listdir(output_dir):
        if filename.startswith('sample_') and filename.endswith('.pt'):
            try:
                # Extract sample index from filename
                idx_str = filename[7:10]  # sample_XXX.pt -> XXX
                sample_idx = int(idx_str)
                completed_samples.append(sample_idx)
            except (ValueError, IndexError):
                continue
    
    completed_samples.sort()
    return completed_samples

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
    parser = argparse.ArgumentParser(description="Extract attentions from Qwen3 model")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--num-gpus", type=str, default="auto")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="data/wandb_gemini.csv")
    parser.add_argument("--output-dir", type=str, default="results_extraction",
                       help="Output directory for individual sample files")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=40000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-memory", type=int, default=40)
    parser.add_argument("--clear-output", action="store_true",
                       help="Clear existing output directory")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume processing from existing files")
    
    args = parser.parse_args()
    
    # Set environment for mirror
    set_mirror_environment()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir, args.clear_output)
    
    # Check for existing samples if resume is enabled
    completed_samples = []
    if args.resume:
        completed_samples = check_existing_samples(output_dir)
        if completed_samples:
            print(f"Found {len(completed_samples)} existing samples: {completed_samples}")
            print("Will skip already processed samples")
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data file {args.data_path} does not exist.")
    
    # Load data
    list_data_dict = load_wandb_gemini(args.data_path, args.num_samples)
    if not list_data_dict:
        raise ValueError("No data loaded from CSV file.")
    
    # Initialize model
    llm = QwenLLM(
        model_name=args.model_name,
        device=args.device,
        num_gpus=args.num_gpus,
        max_memory=args.max_memory
    )
    
    # Process each sample with immediate saving
    success_count = 0
    failed_count = 0
    
    for idx in tqdm(range(len(list_data_dict)), desc="Processing samples"):
        # Skip if already processed and resume is enabled
        if args.resume and idx in completed_samples:
            print(f"Skipping sample {idx} (already processed)")
            success_count += 1  # Count as success since it was processed before
            continue
            
        sample = list_data_dict[idx]
        
        # Create messages for Qwen
        messages = create_qwen_prompt(sample['context'])
        
        # 第一阶段：生成文本（无attention）
        try:
            model_completion, attentions, generated_tokens = llm.generate(
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                return_attentions=True  # 这里会被忽略，实际返回None
            )
            
            # Convert numpy array to list for serialization
            model_completion_ids = generated_tokens.tolist() if hasattr(generated_tokens, 'tolist') else list(generated_tokens)
            
            
            # Extract target field
            target_field, field_start_token, field_end_token = extract_target_field(
                model_completion, llm.tokenizer
            )
            
            # Show content around extracted field if found
            if target_field is not None:
                # Find the field position in the text
                field_pos = model_completion.rfind(f"### {target_field}")
                if field_pos == -1:
                    field_pos = model_completion.rfind(f"###{target_field}")
                if field_pos == -1:
                    field_pos = model_completion.rfind(target_field)
                    
                if field_pos != -1:
                    start = max(0, field_pos - 100)
                    end = min(len(model_completion), field_pos + 200)
                    print(f"Content around extracted field '{target_field}':")
                    print(repr(model_completion[start:end]))
            
            # 第二阶段：提取field attention
            field_attentions = None
            if target_field is not None and field_start_token is not None:
                # 构建完整文本序列
                original_prompt = llm.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                full_text = original_prompt + model_completion
                
                print("开始第二阶段attention提取...")
                field_attentions = llm.extract_field_attentions(
                    full_text, field_start_token, field_end_token
                )
            else:
                print("跳过attention处理：未找到有效的target field")
            
            # Prepare data to save immediately
            to_save = {
                'data_index': sample['data_index'],
                'original_index': sample['original_index'],
                'context': sample['context'],
                'model_completion': model_completion,
                'model_completion_ids': model_completion_ids,
                'target_field': target_field,
                'field_start_token': field_start_token,
                'field_end_token': field_end_token,
                'field_attentions': field_attentions,
                'tokenizer_info': llm.tokenizer.name_or_path
            }
            
            # Save sample immediately
            save_sample_results(to_save, idx, output_dir)
            success_count += 1
            
            # Clean up memory immediately
            del to_save, field_attentions, model_completion, generated_tokens
            clean_memory()
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            failed_count += 1
            clean_memory()  # Clean memory even on failure
            continue
    
    # Save processing summary and configuration
    save_extraction_summary(output_dir, success_count, failed_count, len(list_data_dict), args)
    
    print(f"\nExtraction complete!")
    print(f"Successfully processed: {success_count}/{len(list_data_dict)} samples")
    print(f"Failed samples: {failed_count}")
    print(f"Results saved to directory: {output_dir}")
    print(f"Generated files:")
    print(f"  - sample_XXX.pt: Individual attention data files")
    print(f"  - sample_XXX.log: Individual text record files")  
    print(f"  - extraction_summary.txt: Processing summary")
    print(f"  - extraction_config.json: Configuration record")

if __name__ == "__main__":
    main()