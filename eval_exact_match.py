import json
import sys
import string
from typing import List

import regex

def load_nq_open(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='nq_open', subsample=None):
    '''Format of NQ Open'''
    '''{"question": "who got the first nobel prize in physics", "answers": ["Wilhelm Conrad R\u00f6ntgen"], "ctxs": [{"id": "628725", "title": "Nobel Prize in Phys
ics", "text": "receive a diploma, a medal and a document confirming the prize amount. Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award
 given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of
 the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in
 Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was", "score": "1.6234328", "hasanswer": false,
"original_retrieval_index": 0, "isgold": false},'''
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
            data = data[:100]
            data_indices = data_indices[:100]
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
                net_response=None,
                answer=answers,
                data_index=data_index
            )
            list_data_dict.append(new_item)
    return list_data_dict

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--hyp', type=str, required=True)
    args = parser.parse_args()

    if '*' in args.hyp: # wildcard expansion
        import glob
        files = list(glob.glob(args.hyp))
    else:
        files = [args.hyp]

    data = load_nq_open(args.ref)
    for file in files:
        print(file)
        predictions = [json.loads(x) for x in open(file).readlines()]

        total = 0
        total_best_span_em = 0

        for item in predictions:
            data_index = list(item.keys())[0]
            response = item[data_index]
            score = best_subspan_em(response, data[int(data_index)]['answer'])
            total_best_span_em += score
            total += 1

        print(f"Best span EM: {total_best_span_em / total}")

