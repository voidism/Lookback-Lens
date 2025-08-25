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
    Extract the target field: content after the second-to-last ### until newline
    Returns: (target_field_text, field_start_token_idx, field_end_token_idx)
    """
    # Find all occurrences of ###
    hash_positions = []
    pos = 0
    while pos < len(model_completion):
        pos = model_completion.find('###', pos)
        if pos == -1:
            break
        hash_positions.append(pos)
        pos += 3
    
    if len(hash_positions) < 2:
        print(f"Warning: Found only {len(hash_positions)} ### markers, need at least 2 for second-to-last")
        if len(hash_positions) == 1:
            # Fall back to using the only ### found
            target_hash_pos = hash_positions[0]
            print("Falling back to using the only ### marker found")
        else:
            print("No ### markers found in model completion")
            return None, None, None
    else:
        # Use the second-to-last ### marker
        target_hash_pos = hash_positions[-2]
        print(f"Using second-to-last ### marker at position {target_hash_pos} (out of {len(hash_positions)} total markers)")
    
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
        
        # Debug: Check last few tokens to see why generation stopped
        print("Last 10 generated tokens:")
        for i, token_id in enumerate(generated_tokens[-10:]):
            token_text = self.tokenizer.decode([token_id])
            print(f"  Token {len(generated_tokens)-10+i}: {token_id} -> {repr(token_text)}")
        
        # Check if EOS token was generated
        if self.tokenizer.eos_token_id in generated_tokens:
            eos_positions = [i for i, t in enumerate(generated_tokens) if t == self.tokenizer.eos_token_id]
            print(f"EOS token ({self.tokenizer.eos_token_id}) found at positions: {eos_positions}")
        
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
                    use_cache=False,  # 减少显存使用
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
    parser.add_argument("--output-path", type=str, default="qwen3_attentions.pt")
    parser.add_argument("--jsonl-output", type=str, default="qwen3_records.log")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-memory", type=int, default=40)
    
    args = parser.parse_args()
    
    # Set environment for mirror
    set_mirror_environment()
    
    # Set random seed
    set_seed(args.seed)
    
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
    
    # Process each sample
    to_save_list = []
    for idx in tqdm(range(len(list_data_dict)), desc="Processing samples"):
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
            
            print(f"\nSample {idx} - Model completion preview:")
            print(model_completion[:500] + "..." if len(model_completion) > 500 else model_completion)
            
            # Debug: Check for ### markers
            hash_positions = []
            pos = 0
            while pos < len(model_completion):
                pos = model_completion.find('###', pos)
                if pos == -1:
                    break
                hash_positions.append(pos)
                pos += 3
            print(f"Found {len(hash_positions)} '###' markers at positions: {hash_positions}")
            
            # Show content around last ### marker
            if hash_positions:
                last_pos = hash_positions[-1]
                start = max(0, last_pos - 100)
                end = min(len(model_completion), last_pos + 200)
                print(f"Content around last '###' marker:")
                print(repr(model_completion[start:end]))
            
            # Extract target field
            target_field, field_start_token, field_end_token = extract_target_field(
                model_completion, llm.tokenizer
            )
            
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
            
            # Prepare data to save
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
            to_save_list.append(to_save)
            
            # Save to LOG file for inspection (no truncation)
            mode = 'a' if idx > 0 else 'w'
            with open(args.jsonl_output, mode, encoding='utf-8') as f:
                f.write(f"=== SAMPLE {idx} ===\n")
                f.write(f"Data Index: {sample['data_index']}\n")
                f.write(f"Original Index: {sample['original_index']}\n")
                f.write(f"Generated Tokens: {len(generated_tokens)}\n")
                f.write(f"Target Field: {target_field}\n")
                f.write("=== CONTEXT ===\n")
                f.write(sample['context'])
                f.write("\n=== MODEL COMPLETION ===\n")
                f.write(model_completion)
                f.write("\n=== END SAMPLE ===\n\n")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Save all data to PT file
    print(f"\nSaving {len(to_save_list)} samples to {args.output_path}")
    torch.save(to_save_list, args.output_path)
    
    print(f"Extraction complete! Results saved to:")
    print(f"  - Attention data: {args.output_path}")
    print(f"  - Text records: {args.jsonl_output}")

if __name__ == "__main__":
    main()