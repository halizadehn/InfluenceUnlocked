import os
import sys
import csv
import json
import torch
import argparse
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import model_configs
from defenses import SmoothLLM
import language_models
from attacks import Prompt

CSV_PATH = '/home/julien/CIKMJournal/persuasive_prompts.csv'
OUT_PATH = 'results_rand_insert.json'
PERT_TYPE = 'RandomInsertPerturbation'
PERT_PCT = 20  # 0.2 as percent
NUM_COPIES = 10

def read_prompts(csv_path):
    prompts = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts.append(row['persuasive_prompt'])
    return prompts

def main(args):
    prompts = read_prompts(CSV_PATH)
    config = model_configs.MODELS[args.target_model]
    target_model = language_models.LLM(
        model_path=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        device='cuda:0'
    )
    defense = SmoothLLM(target_model, PERT_TYPE, PERT_PCT, NUM_COPIES)
    results = []
    for prompt_str in tqdm(prompts):
        prompt_obj = Prompt(prompt_str, prompt_str, 100)
        response = defense(prompt_obj)
        results.append({'prompt': prompt_str, 'response': response})
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', type=str, default='vicuna', choices=['vicuna', 'vicuna7b', 'llama2', 'llama3', 'gemma', 'deepseek', 'phi4'])
    args = parser.parse_args()
    # Per-model output file naming
    base, ext = os.path.splitext(OUT_PATH)
    model_suffix = args.target_model
    OUT_PATH = f"{base}_{model_suffix}{ext}"
    main(args)
