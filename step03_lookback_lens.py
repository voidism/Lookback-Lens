import editdistance as ed
import numpy as np
import json
import torch
import transformers
import pickle
from tqdm import tqdm
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

def min_edit_distance_substring(s1, s2):
    len_s1 = len(s1)
    min_edit_dist = float('inf')
    best_substring = None

    assert len(
        s2) >= len_s1, "s2 must be longer than s1\ns1: {}\ns2: {}".format(s1, s2)

    # Slide over s2 to find all substrings of length s1
    for i in range(len(s2) - len_s1 + 1):
        sub_s2 = s2[i:i + len_s1]
        # Calculate edit distance between s1 and this substring
        dist = ed.eval(s1, sub_s2)

        if dist < min_edit_dist:
            min_edit_dist = dist
            best_substring = sub_s2

    return best_substring, min_edit_dist


def load_files(anno_file, attn_file, predefined_span=True, verbose=False, is_feat=False, feat_layer=32, tokenizer_name=None):
    anno_data = []
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    with open(anno_file, 'r') as f:
        for line in f:
            anno_data.append(json.loads(line))

    attn_data = torch.load(attn_file)
    lookback_ratio_key = 'lookback_ratio' if 'lookback_ratio' in attn_data[0] else 'attn_scores'

    # Assuming `lookback_tensor` is in the shape (num_examples, num_layers, num_heads, num_new_tokens)
    # Assuming `labels` is a tensor with shape (num_examples,) indicating hallucination (1) or non-hallucination (0)

    num_examples = len(anno_data)

    lookback_tensor = []
    labels = []
    skipped_examples = 0
    for idx in range(len(anno_data)):
        hallu_label = [0] * len(attn_data[idx]['model_completion_ids'])
        is_hallu = (
            not anno_data[idx]['decision']) if anno_data[idx]['decision'] is not None else True
        if is_hallu:
            tokenized_hallucination = tokenizer(
                anno_data[idx]['response'], return_offsets_mapping=True)
            hallucination_text2ids = tokenized_hallucination['input_ids'][1:]
            hallucination_token_offsets = tokenized_hallucination['offset_mapping'][1:]
            hallucination_attn_ids = attn_data[idx]['model_completion_ids'].tolist(
            )
            # drop the final token if == 2
            if hallucination_attn_ids[-1] == 2:
                hallucination_attn_ids = hallucination_attn_ids[:-1]
            mismatch = False
            if not hallucination_text2ids == hallucination_attn_ids:
                # compute the maximum common substring
                best_substring, min_edit_dist = min_edit_distance_substring(hallucination_text2ids, hallucination_attn_ids) if len(
                    hallucination_text2ids) < len(hallucination_attn_ids) else min_edit_distance_substring(hallucination_attn_ids, hallucination_text2ids)
                if min_edit_dist < 5:
                    if verbose:
                        print(
                            "Usable example with min edit distance:", min_edit_dist)
                    # it means tokenizer.decode and tokenizer.encode are not consistent
                    mismatch = True
                    # best_substring, min_edit_dist = min_edit_distance_substring(hallucination_text2ids, hallucination_attn_ids)
                else:
                    if verbose:
                        print(
                            "Skip example:", f"\n{hallucination_text2ids}\n != \n{hallucination_attn_ids}\n")
                    skipped_examples += 1
                    continue
            # get hallucinated spans from anno_data[idx]['problematic_spans']
            hallucinated_spans = anno_data[idx]['problematic_spans']
            # use the offset of tokenizer to get the span ids positions in the tokenizer(anno_data[idx]['response'])['input_ids']
            hallucinated_spans_token_offsets = []
            for span_text in hallucinated_spans:
                if not span_text in anno_data[idx]['response']:
                    if verbose:
                        print(
                            "Warning:", f"\n{span_text}\n not in \n{anno_data[idx]['response']}\n")
                    if len(span_text) > len(anno_data[idx]['response']):
                        span_text = anno_data[idx]['response']
                    else:
                        best_substring, min_edit_dist = min_edit_distance_substring(
                            span_text, anno_data[idx]['response'])
                        if verbose:
                            print(
                                f"Best substring: {best_substring}, min_edit_dist: {min_edit_dist}")
                        span_text = best_substring
                span_start_char_pos = anno_data[idx]['response'].index(
                    span_text)
                span_end_char_pos = span_start_char_pos + len(span_text)
                # use hallucination_token_offsets to get the span ids positions in the tokenizer(anno_data[idx]['response'])['input_ids']
                # format of the offset_mapping: [(token 1 start_char_pos, token 1 end_char_pos), (token 2 start_char_pos, token 2 end_char_pos), ...]
                span_start_token_pos = -1
                span_end_token_pos = -1

                for i, (start_char_pos, end_char_pos) in enumerate(hallucination_token_offsets):
                    if end_char_pos >= span_start_char_pos and span_start_token_pos == -1:
                        span_start_token_pos = i
                    if end_char_pos >= span_end_char_pos and span_end_token_pos == -1:
                        span_end_token_pos = i
                        break

                assert span_start_token_pos != -1 and span_end_token_pos != -1
                hallucinated_spans_token_offsets.append(
                    (span_start_token_pos, span_end_token_pos))
                min_edit_dist_value = float('inf')
                min_edit_dist_span_start_token_pos = -1
                min_edit_dist_span_end_token_pos = -1
                if mismatch:  # check
                    decoded_span = tokenizer.decode(
                        hallucination_attn_ids[span_start_token_pos:span_end_token_pos+1])
                    edit_dist = ed.eval(span_text, decoded_span)
                    move_total_steps = edit_dist
                    if not span_text == decoded_span:
                        min_edit_dist = abs(
                            len(span_text) - len(decoded_span))
                        # best_substring, min_edit_dist = min_edit_distance_substring(span_text, decoded_span) if len(span_text) < len(decoded_span) else min_edit_distance_substring(decoded_span, span_text)
                        if verbose:
                            print("Mismatched check:",
                                    f"\n{span_text}\n != \n{decoded_span}\n")
                        # try to move the span_start_token_pos and span_end_token_pos within the min_edit_dist
                        exact_match_found = False
                        for move_dist in range(-move_total_steps, move_total_steps+1):
                            if span_start_token_pos + move_dist < len(hallucination_attn_ids) and span_end_token_pos + move_dist < len(hallucination_attn_ids):
                                decoded_span = tokenizer.decode(
                                    hallucination_attn_ids[span_start_token_pos+move_dist:span_end_token_pos+1+move_dist])
                                if span_text == decoded_span:
                                    if verbose:
                                        print(
                                            "Matched check after moving:", f"\n{span_text}\n == \n{decoded_span}\n")
                                    span_start_token_pos += move_dist
                                    span_end_token_pos += move_dist
                                    exact_match_found = True
                                    break
                                else:
                                    edit_dist = ed.eval(
                                        span_text, decoded_span)
                                    if edit_dist < min_edit_dist_value:
                                        min_edit_dist_value = edit_dist
                                        min_edit_dist_span_start_token_pos = span_start_token_pos + move_dist
                                        min_edit_dist_span_end_token_pos = span_end_token_pos + move_dist
                        # if still not break, perform grid search with double for loop
                        for move_dist in range(-move_total_steps, move_total_steps+1):
                            for move_dist2 in range(-move_total_steps, move_total_steps+1):
                                if span_start_token_pos + move_dist < len(hallucination_attn_ids) and span_end_token_pos + move_dist2 < len(hallucination_attn_ids):
                                    decoded_span = tokenizer.decode(
                                        hallucination_attn_ids[span_start_token_pos+move_dist:span_end_token_pos+1+move_dist2])
                                    if span_text == decoded_span:
                                        if verbose:
                                            print(
                                                "Matched check after moving:", f"\n{span_text}\n == \n{decoded_span}\n")
                                        span_start_token_pos += move_dist
                                        span_end_token_pos += move_dist2
                                        exact_match_found = True
                                        break
                                    else:
                                        edit_dist = ed.eval(
                                            span_text, decoded_span)
                                        if edit_dist < min_edit_dist_value:
                                            min_edit_dist_value = edit_dist
                                            min_edit_dist_span_start_token_pos = span_start_token_pos + move_dist
                                            min_edit_dist_span_end_token_pos = span_end_token_pos + move_dist
                            if exact_match_found:
                                break

                        if not exact_match_found:
                            if verbose:
                                print(
                                    f"No exact match found after moving the {span_start_token_pos} and {span_end_token_pos} in the range of {-min_edit_dist} to {min_edit_dist}")
                        if min_edit_dist_span_start_token_pos != -1 and min_edit_dist_value < 5:
                            span_start_token_pos = min_edit_dist_span_start_token_pos
                            span_end_token_pos = min_edit_dist_span_end_token_pos
                            if verbose:
                                print(
                                    f"Adopt the best match with min edit distance: {min_edit_dist_value}")
                            decoded_span = tokenizer.decode(
                                hallucination_attn_ids[span_start_token_pos:span_end_token_pos+1])
                            if verbose:
                                print("Matched check after moving:",
                                        f"\n{span_text}\n ~= \n{decoded_span}\n")
                    else:
                        if verbose:
                            print("Matched check:",
                                    f"\n{span_text}\n == \n{decoded_span}\n")

            if len(hallucinated_spans_token_offsets) == 0:
                if verbose:
                    print("Skip example:", "No hallucinated spans found")
                skipped_examples += 1
                continue
            if predefined_span:
                tmp_lookback_tensor = []
                for i, (s, e) in enumerate(hallucinated_spans_token_offsets):
                    # attn_data[idx]['attn_scores'] shape: (num_layers, num_heads, num_new_tokens)
                    # only extract the attention scores for the tokens in the span
                    # it can have multi spans for one example, so need to concatenate them
                    if i == 0 and s > 0:
                        # extract a non-hallucination span from the beginning of the response
                        if not is_feat:
                            lookback_tensor.append(
                                attn_data[idx][lookback_ratio_key][:, :, :s])
                        else:
                            lookback_tensor.append(
                                attn_data[idx]['extracted_hiddens'][feat_layer].transpose(1, 0).unsqueeze(0)[:, :, :s])
                        labels.append(1)
                    if not is_feat:
                        tmp_lookback_tensor.append(
                            attn_data[idx][lookback_ratio_key][:, :, s:e+1])
                    else:
                        tmp_lookback_tensor.append(
                            attn_data[idx]['extracted_hiddens'][feat_layer].transpose(1, 0).unsqueeze(0)[:, :, s:e+1])
                lookback_tensor.append(torch.cat(tmp_lookback_tensor, dim=-1))
                labels.append(0)
                if e < len(hallucination_token_offsets) - 1:
                    # extract a non-hallucination span from the end of the response
                    if not is_feat:
                        lookback_tensor.append(
                            attn_data[idx][lookback_ratio_key][:, :, e+1:])
                    else:
                        lookback_tensor.append(
                            attn_data[idx]['extracted_hiddens'][feat_layer].transpose(1, 0).unsqueeze(0)[:, :, e+1:])
                    labels.append(1)
            else:
                if not is_feat:
                    sequential_labels = [1] * \
                        attn_data[idx][lookback_ratio_key].shape[-1]
                    for i, (s, e) in enumerate(hallucinated_spans_token_offsets):
                        sequential_labels[s:e+1] = [0] * (e-s+1)
                    lookback_tensor.append(attn_data[idx][lookback_ratio_key][:, :, :])
                else:
                    sequential_labels = [1] * \
                        attn_data[idx]['extracted_hiddens'][feat_layer].shape[0]
                    for i, (s, e) in enumerate(hallucinated_spans_token_offsets):
                        sequential_labels[s:e+1] = [0] * (e-s+1)
                    lookback_tensor.append(attn_data[idx]['extracted_hiddens'][feat_layer].transpose(1, 0).unsqueeze(0))
                labels.append(sequential_labels)
        else:
            if not is_feat:
                lookback_tensor.append(attn_data[idx][lookback_ratio_key])
                if not predefined_span:
                    labels.append([1] * attn_data[idx][lookback_ratio_key].shape[-1])
                else:
                    labels.append(1)
            else:
                lookback_tensor.append(attn_data[idx]['extracted_hiddens'][feat_layer].transpose(1, 0).unsqueeze(0))
                if not predefined_span:
                    labels.append([1] * attn_data[idx]['extracted_hiddens'][feat_layer].shape[0])
                else:
                    labels.append(1)
    if predefined_span:
        labels = np.array(labels)
    if verbose:
        print("Skipped examples:", skipped_examples)

    return lookback_tensor, labels


def convert_to_token_level(lookback_tensor, labels, sliding_window=8, sequential=False, min_pool_target=False):
    # convert to token level
    lookback_tensor_token_level = []
    labels_token_level = []
    for i in range(len(lookback_tensor)):
        num_layers, num_heads, num_new_tokens = lookback_tensor[i].shape
        if sliding_window == 1:
            for j in range(num_new_tokens):
                lookback_tensor_token_level.append(
                    lookback_tensor[i][:, :, j].unsqueeze(-1))
                if sequential:
                    labels_token_level.append(labels[i][j])
                else:
                    labels_token_level.append(labels[i])
        else:
            for j in range(sliding_window-1, num_new_tokens):
                lookback_tensor_token_level.append(
                    lookback_tensor[i][:, :, j-sliding_window+1:j+1])
                if sequential:
                    labels_token_level.append(
                        min(labels[i][j-sliding_window+1:j+1]) if min_pool_target else labels[i][j])
                else:
                    labels_token_level.append(labels[i])
    return lookback_tensor_token_level, labels_token_level


def extract_time_series_features(lookback_tensor):
    features = []
    num_examples = len(lookback_tensor)
    num_layers, num_heads = lookback_tensor[0].shape[:2]
    # Loop over each example to extract features
    baseline_predictions = []
    detailed_feature_names = []
    for i in tqdm(range(num_examples)):
        example_org = lookback_tensor[i]
        example = example_org.clone()
        example = example.view(-1, example.shape[2])
        example = example.transpose(0, 1)
        # shape: (num_new_tokens, num_layers * num_heads)
        # Baseline: Assume higher lookback ratio means less hallucination
        baseline_predictions.append(example.mean(dim=1).mean(0).item())

        # Feature names are: means-L1-H1, means-L1-H2, ..., means-L2-H1, ...
        # L means layers, H means heads, they are flattened in the feature vector (32*32=1024) for each token position
        if i == 0:
            h_index = 0
            for l in range(num_layers):
                for h in range(num_heads):
                    detailed_feature_names.append(
                        f"lookback-mean-L{l}-H{h}")
                    h_index += 1

        # Concatenate the features into a vector
        feature_vector = example.mean(dim=0).numpy()
        if np.isnan(feature_vector).any():
            raise ValueError("NaN detected in the feature vector")
        features.append(feature_vector)

    return np.array(features), detailed_feature_names, baseline_predictions


def main(anno_file_1, attn_file_1, anno_file_2, attn_file_2, 
         sliding_window=8, 
         predefined_span=True,
         is_feat=False,
         feat_layer=32,
         two_fold=False,
         conversion=None,
         tokenizer_name=None,
         output_path=None
        ):
    comb1 = (anno_file_1, attn_file_1, anno_file_2, attn_file_2)
    comb2 = (anno_file_2, attn_file_2, anno_file_1, attn_file_1)
    if conversion is None:
        all_combs = [comb1, comb2]
    else:
        all_combs = [comb1]
    output_small_table = []
    output_small_table.append(
        ['Train AUROC (on A)', 'Test AUROC (on A)', 'Transfer AUROC (on B)']
    )
    for anno_file, attn_file, transfer_anno_file, transfer_attn_file in all_combs:
        print(f"======== Loading data from {anno_file} and {attn_file}...")
        # load data
        lookback_tensor, labels = load_files(
            anno_file, attn_file, predefined_span=predefined_span,
            is_feat=is_feat, feat_layer=feat_layer, tokenizer_name=tokenizer_name)
        if not predefined_span:
            lookback_tensor, labels = convert_to_token_level(
                lookback_tensor, labels, sliding_window=sliding_window, sequential=True, min_pool_target=True)

        # Extract features from the time series
        time_series_features, feature_names, baseline_predictions = extract_time_series_features(lookback_tensor)

        # Baseline prediction AUROC
        baseline_auroc = roc_auc_score(labels, baseline_predictions)
        print("A trivial baseline: if higher lookback ratio means less hallucination.")
        print(f"Baseline AUROC: {baseline_auroc:.9f}")

        total_train_auroc = 0
        total_test_auroc = 0
        if conversion is None:
            # Train-test split
            if two_fold:
                X_train, X_test, y_train, y_test = train_test_split(
                    time_series_features, labels, test_size=0.5, random_state=42)
                datasets = [(X_train, y_train, X_test, y_test), (X_test, y_test, X_train, y_train)]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    time_series_features, labels, test_size=0.2, random_state=42)
                datasets = [(X_train, y_train, X_test, y_test)]

            for X_train, y_train, X_test, y_test in datasets:
                classifier = LogisticRegression(max_iter=1000)
                classifier.fit(X_train, y_train)

                # Train AUROC
                y_pred_proba = classifier.predict_proba(X_train)[:, 1]
                train_auroc = roc_auc_score(y_train, y_pred_proba)
                total_train_auroc += train_auroc

                print(
                    f"Train AUROC of the classifier: {train_auroc:.9f}")

                # Evaluate AUROC
                y_pred_proba = classifier.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y_test, y_pred_proba)
                total_test_auroc += auroc

                print(
                    f"Test AUROC of the classifier: {auroc:.9f}")

                # Feature importance
                if not hasattr(classifier, 'coef_'):
                    feature_importance = classifier.feature_importances_
                    important_features = sorted(
                        zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
                else:
                    feature_importance = classifier.coef_[0]
                    important_features = sorted(
                        zip(feature_names, feature_importance), key=lambda x: abs(x[1]), reverse=True)

                print("Top-10 important features:")
                for feature, importance in important_features[:10]:
                    print(f"{feature}: {importance:.9f}")

            total_train_auroc /= len(datasets)
            total_test_auroc /= len(datasets)

        # Train a classifier on 100% of the data
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(time_series_features, labels)

        y_pred_proba = classifier.predict_proba(time_series_features)[:, 1]

        # save classifier
        prediction_level = (f'sliding_window_{sliding_window}' if not predefined_span else 'predefined_span')
        if is_feat:
            prediction_level += f'_feat_{feat_layer}'

        basename = anno_file.split('/')[-1].replace('.jsonl', '')
        output_file = f"classifier_{basename}_{prediction_level}.pkl"
        if output_path is not None:
            output_file = os.path.join(output_path, output_file)
        with open(output_file, 'wb') as f:
            pickle.dump({'clf': classifier}, f)

        # Transfer the classifier to the other dataset
        print(
            f"======== Transfer to data from {transfer_anno_file} and {transfer_attn_file}...")
        transfer_lookback_tensor, transfer_labels = load_files(
            transfer_anno_file, transfer_attn_file, predefined_span=predefined_span,
            is_feat=is_feat, feat_layer=feat_layer, tokenizer_name=tokenizer_name)
        if not predefined_span:
            transfer_lookback_tensor, transfer_labels = convert_to_token_level(
                transfer_lookback_tensor, transfer_labels, sliding_window=sliding_window, sequential=True, min_pool_target=True)
        transfer_time_series_features, transfer_feature_names, transfer_baseline_predictions = extract_time_series_features(transfer_lookback_tensor)
        # Baseline prediction AUROC
        transfer_auroc = roc_auc_score(
            transfer_labels, transfer_baseline_predictions)
        print("A trivial baseline: if higher lookback ratio means less hallucination.")
        print(f"Transfer Baseline AUROC: {transfer_auroc:.9f}")
        if conversion is not None:
            weight = conversion['weights_matrix']
            bias = conversion['intercepts']
            transfer_time_series_features = (torch.tensor(transfer_time_series_features) @ weight.T + bias).numpy()
        y_pred = classifier.predict(transfer_time_series_features)
        y_pred_proba = classifier.predict_proba(
            transfer_time_series_features)[:, 1]
        transfer_auroc = roc_auc_score(transfer_labels, y_pred_proba)
        print(
            f"Transfer AUROC of the classifier: {transfer_auroc:.9f}")
        # make a output table in csv format for all the scores recorded
        output_small_table.append(
            [total_train_auroc, total_test_auroc, transfer_auroc]
        )
    print("======== Results:")
    file_1 = anno_file_1.split('/')[-1].replace('.jsonl', '').replace('anno-', '')
    file_2 = anno_file_2.split('/')[-1].replace('.jsonl', '').replace('anno-', '')
    width = len(f'A={file_1};B={file_2}')
    names = [' '*width, f'A={file_1};B={file_2}', f'A={file_2};B={file_1}']
    for i, row in enumerate(output_small_table):
        print(', '.join([names[i]]+[str(x) for x in row]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Process some features and tensors.")

    parser.add_argument('--anno_1', type=str, default=None)
    parser.add_argument('--lookback_ratio_1', type=str, default=None)
    parser.add_argument('--anno_2', type=str, default=None)
    parser.add_argument('--lookback_ratio_2', type=str, default=None)

    parser.add_argument('--sliding_window', type=int, default=None,
                        help='Sliding window size')

    parser.add_argument('--feat', action='store_true',
                        help='Flag to use hidden state features.')
    parser.add_argument('--feat_layer', type=int, default=32,
                        help='Layer index to use the features from the teacher-forcing model')
    
    # model: [7b or 13b]
    parser.add_argument('--model', type=str, default='7b')
    # tokenizer name
    parser.add_argument('--tokenizer_name', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    # conversion_matrix
    parser.add_argument('--conversion_matrix', type=str, default=None)
    # output path
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()
    conversion = None
    if args.conversion_matrix is not None:
        conversion = pickle.load(open(args.conversion_matrix, 'rb'))
    
    predefined_span = (args.sliding_window is None)
    sliding_window = args.sliding_window
    is_feat = args.feat
    feat_layer = args.feat_layer
    two_fold = True

    if args.model == '7b':
        num_heads = 32
        num_layers = 32
    elif args.model == '13b':
        num_heads = 40
        num_layers = 40

    main(
        args.anno_1, 
        args.lookback_ratio_1,
        args.anno_2, 
        args.lookback_ratio_2,
        predefined_span=predefined_span,
        sliding_window=sliding_window,
        is_feat=is_feat,
        feat_layer=feat_layer,
        two_fold=two_fold,
        conversion=conversion,
        tokenizer_name=args.tokenizer_name,
        output_path=args.output_path
    )
