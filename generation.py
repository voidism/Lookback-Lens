# Reference: https://github.com/lm-sys/FastChat

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import numpy as np

class LLM:
    def __init__(self, model_name, device, num_gpus, auth_token=None, max_memory=40, **kwargs):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_memory = max_memory
        self.tuned_lens = None

        self.model, self.tokenizer = self.load_model(model_name=model_name, max_memory=max_memory, auth_token=auth_token)

        
    def load_model(self, model_name, max_memory, auth_token=None):
        if 'gpt2' in model_name:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.cuda()
            return model, tokenizer
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"offload/{model_name}"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{max_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        # low_cpu_mem_usage = True if not '70b' in model_name else False
        if auth_token is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
            model = AutoModelForCausalLM.from_pretrained(model_name,
                # low_cpu_mem_usage=True, 
                token=auth_token, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,
                # low_cpu_mem_usage=True, 
                **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            if 'llama' in self.model_name.lower():
                stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            else:
                stop_word_ids = self.tokenizer.encode('\n' + stop_word)
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mode='vanilla', verbose=True, remove_stop_words=False, return_attentions=False, guiding_classifier=None, chunk_size=None, num_candidates=None, conversion_matrix=None, extra_prompt_length=None, teacher_forcing_seq=None, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            if verbose:
                print('MODEL INPUT LENGTH: {0}'.format(input_ids.shape[-1]))
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'vanilla':
                outputs = self.model.generate(inputs=input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, 
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, 
                                    output_attentions=return_attentions, teacher_forcing_seq=teacher_forcing_seq, **kwargs)

            elif mode == 'classifier_guided':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, 
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, 
                                        extra_prompt_length=extra_prompt_length,
                                        guiding_classifier=guiding_classifier, chunk_size=chunk_size, 
                                        num_candidates=num_candidates, conversion_matrix=conversion_matrix, **kwargs,)
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()
        if not return_attentions:
            return output_str, gen_arr
        else:
            return output_str, outputs.attentions, gen_arr
