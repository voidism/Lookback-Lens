import argparse
import json
import os
import openai
import tiktoken
import torch

# Define the prompt components

data_response_names = {
    'summarization': 'Summary',
    'nq_open': 'Answer'
}

data_response_names_gt = {
    'summarization': 'Summary',
    'nq_open': 'Answers (a list of valid answers)'
}

eval_prompt_before = {
    'summarization': "You will be provided with a document and a proposed summary. Your task is to determine if the proposed summary can be directly inferred from the document. If the summary contains any information not found in the document, it is considered false. Even if the summary is different from a ground truth summary, it might still be true, as long as it doesn't contain false information.\nFor each proposed summary, explain why it is true or false based on the information from the document. Focus only on the original document's content, disregarding any external context.\nAfter your explanation, give your final conclusion as **Conclusion: True** if the proposed summary is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. If your conclusion is 'False', identify the exact phrases or name entities from the summary that is incorrect by stating **Problematic Spans: [the inaccurate text spans from the summary, in Python list of strings format]**.",
    'nq_open': "You will be provided with a document and a proposed answer to a question. Your task is to determine if the proposed answer can be directly inferred from the document. If the answer contains any information not found in the document, it is considered false. Even if the answer is different from a ground truth answer, it might still be true, as long as it doesn't contain false information.\nFor each proposed answer, explain why it is true or false based on the information from the document. Focus only on the original document's content, disregarding any external context.\nAfter your explanation, give your final conclusion as **Conclusion: True** if the proposed answer is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. If your conclusion is 'False', identify the exact phrases or name entities from the answer that is incorrect by stating **Problematic Spans: [the inaccurate text spans from the answer, in Python list of strings format]**."
}

eval_prompt_after = {
    'summarization': "Write your explanation first, and then give your final conclusion as **Conclusion: True** if the proposed summary is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. Add **Problematic Spans: [the exact inaccurate text spans from the summary, in a list of strings]** if your conclusion is 'False'.",
    'nq_open': "Write your explanation first, and then give your final conclusion as **Conclusion: True** if the proposed answer is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. Add **Problematic Spans: [the exact inaccurate text spans from the answer, in a list of strings]** if your conclusion is 'False'.",
}

# Function to load jsonl files
def load_jsonl(file_path):
    list_data_dict = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

def load_summarization(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='cnndm', subsample=None):
    list_data_dict = {}
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
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size]
            data_indices = data_indices[shard_id * chunk_size: (shard_id + 1) * chunk_size]

        for idx in range(len(data)):
            data_index = data_indices[idx]
            context = "#Document#: " if data_type == 'cnndm' else "#Article#: "
            context += data[idx]['document']
            new_item = dict(
                context=context,
                data_index=data_index,
                net_response=data[idx]['summary'],
            )
            list_data_dict[data_index] = new_item

    return list_data_dict

def load_nq_open(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='nq_open', subsample=None):
    list_data_dict = {}
    is_train = 'nq_train' in file_path
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        if debug:
            data = data[:10]
        if subsample is not None:
            # select data if idx%subsample == 0
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size]

        for idx in range(len(data)):
            data_index = idx
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
                net_response=str(answers),
                answer=answers[0],
                data_index=data_index
            )
            list_data_dict[data_index] = new_item
    return list_data_dict

# Function to evaluate responses using GPT-4o
def evaluate_response(document, gt_response, response, tokenizer, data_type='summarization', debug=False):
    prompt = f"{eval_prompt_before[data_type]}\n\n#Document#: {document}\n\n#Ground Truth {data_response_names_gt[data_type]}#: {gt_response}\n\n#Proposed {data_response_names[data_type]}#: {response}\n\n{eval_prompt_after[data_type]}"
    
    # Calculate input token usage
    input_token_usage = len(tokenizer.encode(prompt))

    response = openai.chat.completions.create(
        model='gpt-4o-2024-05-13',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    text = response.choices[0].message.content
    
    # Calculate output token usage
    output_token_usage = len(tokenizer.encode(text))

    if debug:
        print('-------------------')
        print(prompt)
        print('\n'+text+'\n')
        print('-------------------', flush=True)

    problematic_spans = []
    if "Problematic Spans: " in text:
        problematic_spans = text.split('Problematic Spans: ')[1]
        if '**' in problematic_spans:
            problematic_spans = problematic_spans.split('**')[0].strip()
        # problematic_spans is in python list of string format, extract the list
        try:
            problematic_spans = eval(problematic_spans)
        except:
            print("Error in parsing problematic spans:", problematic_spans)
            problematic_spans = problematic_spans[1:-1].split(', ')

        if debug:
            print(problematic_spans)

    if "Conclusion: " in text:
        dec = text.split('Conclusion: ')[1]
        if '**' in dec:
            dec = dec.split('**')[0]
        if debug:
            print(dec)
        if "True" in dec:
            decision = True
        elif "False" in dec:
            decision = False
        else:
            decision = None
    else:
        decision = None
    
    # Calculate cost
    cost = (input_token_usage / 1_000_000 * 5) + (output_token_usage / 1_000_000 * 15)
    
    return decision, text, problematic_spans, cost

def main(hyp_path, ref_path, output_path, limit=None):
    # Load jsonl files
    if not 'nq' in ref_path:
        data_type = 'summarization'
        gold_data = load_summarization(ref_path, data_type=data_type)
    else:
        data_type = 'nq_open'
        gold_data = load_nq_open(ref_path, data_type='nq_open')
    
    if not '.pt' in hyp_path:
        response_data = load_jsonl(hyp_path)
        if limit is not None:
            response_data = response_data[:limit]
        # Extract summaries
        responses = [value for item in response_data for value in item.values()]
    else:
        response_data = torch.load(hyp_path)
        if limit is not None:
            response_data = response_data[:limit]
        responses = [item['model_completion'] for item in response_data]

    # Initialize OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("o200k_base")

    done_dict = {}
    if os.path.exists(output_path):
        print("Try to resume from existing output file.")
        with open(output_path, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                data = json.loads(line)
                done_dict[data['index']] = data
            

    # Open output file
    with open(output_path, 'w') as fw:
        results = []
        total_cost = 0
        corr = 0
        total = 0

        # Evaluate each pair of summaries
        for idx in range(len(responses)):
            response = responses[idx]
            assert idx in gold_data, f"Index {idx} not found in data_dict"
            document = gold_data[idx]['context']
            gt_response = gold_data[idx]['net_response']

            if idx in done_dict:
                fw.write(json.dumps(done_dict[idx]) + '\n')
                continue
            decision, gpt4_explanation, problematic_spans, cost = evaluate_response(document, gt_response, response, tokenizer, data_type=data_type, debug=True)
            results.append({'index': idx, 'document': document.strip(), 'ground_truth': gt_response.strip(), 'response': response, 'decision': decision, 'gpt4_explanation': gpt4_explanation, 'problematic_spans': problematic_spans, 'cost': cost})
            fw.write(json.dumps({'index': idx, 'document': document.strip(), 'ground_truth': gt_response.strip(), 'response': response, 'decision': decision, 'gpt4_explanation': gpt4_explanation, 'problematic_spans': problematic_spans, 'cost': cost}) + '\n')
            fw.flush()

            # Accumulate total cost
            total_cost += cost

            # Accuracy
            if decision:
                corr += 1
            total += 1

        # Print total cost
        print(f"Total cost: ${total_cost:.9f}")
        # Print accuracy
        print(f"Accuracy: {corr / total:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate faithfulness of summaries using GPT-4o.")
    parser.add_argument('--hyp', type=str, required=True, help='Path to the hypothesis jsonl file')
    parser.add_argument('--ref', type=str, required=True, help='Path to the reference jsonl file')
    parser.add_argument('--out', type=str, required=True, help='Path to the output jsonl file')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')

    args = parser.parse_args()
    main(args.hyp, args.ref, args.out, args.limit)
    # Usage: OPENAI_API_KEY=[your_key] python step02_eval_gpt4o.py --hyp data/hypothesis_from_step01.pt--ref data/gold_data.jsonl --out data/anno_output.jsonl    
