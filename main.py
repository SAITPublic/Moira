import argparse
from importlib.metadata import version
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils import export_base_distribution, export_results
from moira import Moira
import os
import json

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, cache_dir, seed, seqlen, tokenizer):
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=cache_dir)
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=cache_dir)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, cache_dir, nsamples=128, seed=0, seqlen=4096, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, cache_dir, seed, seqlen, tokenizer)

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    dataset = args.data
    print(f"evaluating on {dataset}")
    _, testloader = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()
    return ppl.item()

def eval_zero_shot(save_model_path, task_list, cache_dir, num_fewshot=0, parallelize=False):
    import lm_eval
    model_args = f"cache_dir={cache_dir},pretrained={save_model_path}"
    if parallelize:
        model_args += ",parallelize=True"
    return lm_eval.simple_evaluate(model="hf", 
                                   model_args=model_args,
                                   tasks=task_list,
                                   num_fewshot=num_fewshot,
                                   batch_size=8,
                                   device=None)

def extract_acc_none(source_dict, keys=None):  
    if keys is None:
        keys = source_dict.keys()

    result = {}  
    for k in keys:  
        sub = source_dict.get(k, {})  
        value = sub.get('acc,none')  
        result[k] = {"acc,none": value}  
    return result  


def get_llm(model_name, cache_dir):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 dtype=torch.float16, 
                                                 cache_dir=cache_dir, 
                                                 low_cpu_mem_usage=True,
                                                 device_map="auto")
    if model.config.max_position_embeddings>=4096:
        model.seqlen = 4096
    else:
        model.seqlen = model.config.max_position_embeddings
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--data', type=str, help='Dataset')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--target_wires', type=float, default=1.0, help='Expected target wires.')
    parser.add_argument('--eval_zero_shot', action="store_true", help="test for zero shot")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--cache_dir", default="./assets/cache", type=str )
    parser.add_argument('--base_dist_dir', type=str, default="./base_dist", help='Path to save logs for base distribution.')
    parser.add_argument('--converge_dir', type=str, default="./converge", help='Path to save layerwise wiring logs.')
    parser.add_argument('--results_dir', type=str, default="./results", help='Path to save experiment results.')
    args = parser.parse_args()
    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    os.makedirs(args.base_dist_dir, exist_ok=True)
    os.makedirs(args.converge_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    moira = Moira(target_wires=args.target_wires)

    device_map = getattr(model, "hf_device_map", None) or {}
    multi_gpu = len(set(device_map.values())) > 1
    device = device_map["lm_head"] if multi_gpu else model.device
    vanilla_ppl = eval_ppl(args, model, tokenizer, device)
    print(f"[{args.data}] Vanilla perplexity: {vanilla_ppl}")

    vanilla_csv = os.path.join(args.base_dist_dir, f"base_dist_{model_name}_vanilla.csv")
    if not os.path.exists(vanilla_csv):
        print(f"Start the measurement of candidate wires of the baseline LLM [{model_name}]")
        export_base_distribution(model, filename=vanilla_csv)
        print(f"Complete vanilla profiling")
    else:
        print(f"Already exists base profile of vanilla")
    vanilla_dist = pd.read_csv(vanilla_csv).set_index("Metric / Basis")
    vanilla_wires = int(vanilla_dist.loc["Total_Wires_Used", "Global_Total"])
    
    _, testloaders = get_loaders("c4", seqlen=model.seqlen, tokenizer=tokenizer)
    total_wires_used, final_estimated_averged_wires, log_steps = moira.run(model, tokenizer, testloaders)
    
    df = pd.DataFrame(log_steps)
    conv_logs = os.path.join(args.converge_dir, f"layerwise_wiring_{model_name}_{args.target_wires}.csv")
    df.to_csv(conv_logs, index=False)  
    print(f"[INFO] Step‑wise log saved to {conv_logs}")
    print(f"Start the measurement of candidate wires of the optimized LLM")
    export_base_distribution(model, filename=os.path.join(args.base_dist_dir, f"base_dist_{model_name}_moira-{args.target_wires}.csv"))
    print(f"Complete profiling")

    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"[{args.data}] Moira-{args.target_wires} perplexity: {ppl_test}")

    results_dict = {
        "Model": model_name,
        "Vanilla_PPL": vanilla_ppl,
        "Vanilla_wires": vanilla_wires,
        "n_wires_used": total_wires_used,
        "N_bar": final_estimated_averged_wires,
        "N_target": args.target_wires,
        "Moira_PPL": ppl_test,
    }
    export_results(os.path.join(args.results_dir, f"ppl_and_wires_{model_name}.csv"), results_dict)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.eval_zero_shot:
        if not args.save_model:
            parser.error("--eval_zero_shot requires --save_model")
        task_list = ["boolq","rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.save_model, task_list, args.cache_dir, num_shot, parallelize=multi_gpu)
        print("********************************")
        print("zero_shot evaluation results")
        print(results["results"])

        all_extracted = extract_acc_none(results["results"])
        os.makedirs(args.results_dir, exist_ok=True)
        lm_eval_logs = os.path.join(args.results_dir, f"lm_eval_{model_name}_moira-{args.target_wires}.json")
        with open(lm_eval_logs, "w", encoding="utf-8") as f:  
            json.dump(all_extracted, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {lm_eval_logs}")

if __name__ == '__main__':
    main()
