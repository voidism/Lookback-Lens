# Ref: https://github.com/kojima-takeshi188/zero_shot_cot

import re
import os
import json
import random
import torch
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse
import pickle

from generation import LLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.logging.set_verbosity(40)


def num_tokens_from_message(message, llama2_tokenizer):
    return len(llama2_tokenizer(message)['input_ids'])


def truncate_message(prompt1, prompt2, llama2_tokenizer):
    if num_tokens_from_message(prompt1 + prompt2) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2, llama2_tokenizer)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split(' ')[:-1])
    prompt = prompt1 + prompt2
    return prompt

data_context_names = {
    'cnndm': 'Document',
    'xsum': 'Article',
    'nq': 'Document',
}

data_response_names = {
    'cnndm': 'Summary',
    'xsum': 'Summary',
    'nq': 'Answer',
}

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

def load_nq_open(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='nq_open', subsample=None):
    list_data_dict = []
    is_train = 'nq_train' in file_path
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        data_indices = []
        data_index = 0
        for line in f:
            data.append(json.loads(line))
            data_indices.append(data_index)
            data_index += 1
        if debug:
            data = data[:10]
            data_indices = data_indices[:10]
        if subsample is not None:
            # select data if idx%subsample == 0
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
            data_indices = [data_indices[i] for i in range(len(data_indices)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size] if shard_id != total_shard - 1 else data[shard_id * chunk_size:]
            data_indices = data_indices[shard_id * chunk_size: (shard_id + 1) * chunk_size] if shard_id != total_shard - 1 else data_indices[shard_id * chunk_size:]

        for idx in range(len(data)):
            data_index = data_indices[idx]
            question = data[idx]['question']
            # capitalize the first letter of the question, add the question mark if not present at the end
            question = question[0].upper() + question[1:]
            if question[-1] != '?':
                question += '?'
            answers = data[idx]['answers']
            if is_train:
                pos_ctxs = data[idx]['positive_ctxs']
                neg_ctxs = data[idx]['negative_ctxs']
            else:
                ctxs = data[idx]['ctxs']
                pos_ctxs = [ctx for ctx in ctxs if ctx['hasanswer']]
                neg_ctxs = [ctx for ctx in ctxs if not ctx['hasanswer']]
            assert len(pos_ctxs) > 0, "No positive context found."
            assert len(neg_ctxs) >= 2, "At least two negative contexts are required."
            context = f"#Document#: " + neg_ctxs[0]['text'] + '\n' + pos_ctxs[0]['text'] + '\n' + neg_ctxs[1]['text']
            context += f"\n#Question#: {question}"
            response = f"\n#Answer#:"
            new_item = dict(
                context=context,
                response=response,
                answer=answers[0],
                data_index=data_index
            )
            list_data_dict.append(new_item)
    return list_data_dict


def load_jsonl(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='cnndm', subsample=None):
    list_data_dict = []
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        data_indices = []
        data_index = 0
        for line in f:
            data.append(json.loads(line))
            data_indices.append(data_index)
            data_index += 1
        if debug:
            data = data[:10]
            data_indices = data_indices[:10]
        if subsample is not None:
            # select data if idx%subsample == 0
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
            data_indices = [data_indices[i] for i in range(len(data_indices)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size] if shard_id != total_shard - 1 else data[shard_id * chunk_size:]
            data_indices = data_indices[shard_id * chunk_size: (shard_id + 1) * chunk_size] if shard_id != total_shard - 1 else data_indices[shard_id * chunk_size:]

        for idx in range(len(data)):
            data_index = data_indices[idx]
            if data_type == 'mt_bench':
                context = data[idx]['document']
                category = data[idx]['category']
            else:
                context = f"#{data_context_names[data_type]}#: " + data[idx]["document"]
            new_item = dict(
                context=context,
                data_index=data_index,
                category=category if data_type == 'mt_bench' else None
            )
            list_data_dict.append(new_item)

    return list_data_dict

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def create_demo_text(data_type='cnndm'):
    if data_type == 'cnndm':
        return "Generate a summary based on the information in the document.\n\n"
    elif data_type == 'nq':
        return "Answer the question based on the information in the document. Explain your reasoning in the document step-by-step before providing the final answer.\n\n"
    elif data_type == 'xsum':
        return "Generate a summary comprising of 1 sentence for the given article.\n\n"
    else:
        return None


def build_prompt(context, response, data_type='cnndm', llama2_tokenizer=None):
    demo = create_demo_text(data_type)
    prompt = demo + context
    if data_type == 'cnndm':
        input_text_prompt = truncate_message(prompt, response, llama2_tokenizer)
    else:
        input_text_prompt = prompt + response
    return input_text_prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-7b-chat-hf")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-memory", type=int, default=45)
    parser.add_argument("--auth-token", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="./output.jsonl")
    # data
    parser.add_argument("--data-type", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./data/nq-open-10_total_documents_gold_at_4.jsonl")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--subsample", type=int, default=None)
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    # generation
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95) # only used when do_sample is True
    parser.add_argument("--top_k", type=int, default=0) # only used when do_sample is True
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # classifier model path
    parser.add_argument("--guiding-classifier", type=str, default=None)
    # chunk size
    parser.add_argument("--chunk-size", type=int, default=5)
    # num candidates
    parser.add_argument("--num-candidates", type=int, default=8)
    # conversion matrix
    parser.add_argument("--conversion-matrix", type=str, default=None)
    # feat_layer
    parser.add_argument("--feat-layer", type=int, default=None)
    

    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    set_seed(args.seed)

    forced_truncate = ('gpt2' in args.model_name)
    if args.data_type is None:
        if 'cnndm' in args.data_path or 'summ' in args.data_path:
            args.data_type = 'cnndm'
        elif 'nq-open' in args.data_path:
            args.data_type = 'nq'
        elif 'xsum' in args.data_path:
            args.data_type = 'xsum'
        elif 'mt_bench' in args.data_path:
            args.data_type = 'mt_bench'
        else:
            raise ValueError("Please specify the data type.")

    fp = args.data_path
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")

    if "nq-open" in fp:
        list_data_dict = load_nq_open(fp, parallel=args.parallel, total_shard=args.total_shard, shard_id=args.shard_id, debug=args.debug, subsample=args.subsample)
    else:
        list_data_dict = load_jsonl(fp, parallel=args.parallel, total_shard=args.total_shard, shard_id=args.shard_id, debug=args.debug, data_type=args.data_type, subsample=args.subsample)
    
    llm = LLM(
        model_name, device, num_gpus, 
        auth_token=args.auth_token, 
        max_memory=args.max_memory)
    stop_word_list = ["### User:", "Q:", "\end{code}", "#Document#:", "#Pondering#:", "#Question#:", "#Dialogue History#:"]
    llm.set_stop_words(stop_word_list)
    guiding_classifier = None
    if args.guiding_classifier is not None:
        if args.guiding_classifier == 'vectara/hallucination_evaluation_model':
            from sentence_transformers import CrossEncoder
            guiding_classifier = {}
            model = CrossEncoder(args.guiding_classifier)
            tokenizer = llm.tokenizer
            guiding_classifier['model'] = model
            guiding_classifier['tokenizer'] = tokenizer
            guiding_classifier['is_cross_encoder'] = True
            guiding_classifier['is_deberta'] = False
        elif '.pkl' in args.guiding_classifier:
            guiding_classifier = pickle.load(open(args.guiding_classifier, 'rb'))
            guiding_classifier['is_cross_encoder'] = False
            guiding_classifier['is_deberta'] = False
        else: # is deberta nli model
            nli_model = AutoModelForSequenceClassification.from_pretrained(args.guiding_classifier)
            nli_tokenizer = AutoTokenizer.from_pretrained(args.guiding_classifier)
            nli_model.to(device)
            guiding_classifier = {'model': nli_model, 'tokenizer': nli_tokenizer, 'is_deberta': True, 'is_cross_encoder': False}
        mode = "classifier_guided"
        print("MODE: classifier guided decoding", flush=True)
    else:
        mode = "vanilla"
        print("MODE: vanilla decoding", flush=True)

    conversion_matrix = None
    if args.conversion_matrix is not None:
        conversion_matrix = pickle.load(open(args.conversion_matrix, 'rb'))

    output_path = args.output_path+"_"+str(args.shard_id)+".jsonl"
    done_indices = {}
    if os.path.exists(output_path):
        print("Try to resume from the existing output file.")
        with open(output_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                for key, value in data.items():
                    done_indices[int(key)] = value
        fw = open(output_path, 'a')
    else:
        fw = open(output_path, 'w')
    if args.data_type == 'mt_bench':
        extra_prompt_length = len(llm.tokenizer(f"\n\n### Assistant:")['input_ids'])
    else:
        extra_prompt_length = len(llm.tokenizer(f"\n#{data_response_names[args.data_type]}#:")['input_ids']) - 1
    time_decoding = 0.0
    for idx in tqdm(range(len(list_data_dict))):
        sample = list_data_dict[idx]
        if sample['data_index'] in done_indices:
            continue
        
        if args.data_type != 'mt_bench':
            input_text = build_prompt(sample['context'], f"\n#{data_response_names[args.data_type]}#:", data_type=args.data_type, llama2_tokenizer=llm.tokenizer)
        else:
            input_text = sample['context']
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, 
                               do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k,
                               temperature=args.temperature, 
                               mode=mode)
        if args.data_type == 'mt_bench':
            if sample["category"] in temperature_config:
                temperature = temperature_config[sample["category"]]
            else:
                temperature = 0.7
            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True
            generate_kwargs['temperature'] = temperature
            generate_kwargs['do_sample'] = do_sample

        model_completion, gen_seq = llm.generate(
            input_text, guiding_classifier=guiding_classifier, conversion_matrix=conversion_matrix, 
            extra_prompt_length=extra_prompt_length,
            feat_layer=args.feat_layer,
            chunk_size=args.chunk_size, num_candidates=args.num_candidates, **generate_kwargs)
        cropped_model_completion = model_completion
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                cropped_model_completion = model_completion[:-length_to_remove]
        cropped_gen_seq = llm.tokenizer(model_completion)['input_ids'][1:]
        return_dict = {
            sample['data_index']: cropped_model_completion.strip()
        }
        fw.write(json.dumps(return_dict, ensure_ascii=False) + '\n')
        fw.flush()
    fw.close()