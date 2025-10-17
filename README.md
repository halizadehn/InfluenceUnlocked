# Influence Unlocked: Persuasion-Aware Jailbreaking and the Hidden Persuasive Fingerprints of Large Language Models

This repository contains the implementation and evaluation framework for studying persuasion-based jailbreak attacks on Large Language Models (LLMs). The project operationalizes Cialdini’s persuasion principles to generate adversarial prompts and analyze model-specific susceptibility profiles.

## Pipeline overview
![overview](./Figures/framework.jpg)



## Dataset

- **AdvBench**: This dataset contains 520 queries, covering various types of harmful behavior ([`Datasts/AdvBench.csv`](https://github.com/halizadehn/InfluenceUnlocked/blob/main/Datasets/AdvBench.csv)). 
- **StrongREJECT**: This dataset encompasses six major categories of behaviors that are explicitly prohibited under all major LLM usage policies: (1) illegal goods and services, (2) non-violent crimes, (3) hate, harassment, and discrimination, (4) disinformation and deception, (5) violence, and (6) sexual content ([`Datasts/StrongREJECT.csv`](https://github.com/halizadehn/InfluenceUnlocked/blob/main/Datasets/StrongREJECT.csv)). 

## Target Models

- Vicuna-7b
- Llama2-7b-chat
- Llama3
- DeepSeek-R1
- Gemma3
- Phi4

> **Note:** Target models are available via [Ollama](https://ollama.com/).

## Baselines:
- **[GCG](https://github.com/llm-attacks/llm-attacks)**: finds suffixes via gradient synthesis
- **[PAIR](https://arxiv.org/pdf/2310.08419)**: an algorithm that creates semantic jailbreaks using only black-box access to an LLM
- **[PAP](https://github.com/CHATS-lab/persuasive_jailbreaker)**: generates persuasive adversarial prompts using different strategies
- **['Sure, here’s'](https://github.com/llm-attacks/llm-attacks)**: appends harmful prompts with the suffix “Sure, here’s”
  
> **Note:** We leveraged baseline implementations provided by the [StrongReject benchmark](https://github.com/dsbowen/strong_reject).

## Code Structure

- [persuasive_prompt_generation.py](./Code/persuasive_prompt_generation.py): generates persuasive variants of harmful queries by leveraging Cialdini's persuasion principles to evaluate their effectiveness in eliciting harmful responses from LLMs.

- [get_model_responses.py](./Code/get_model_responses.py): implements the process of querying LLMs with original and persuasive prompt variants, collecting their responses to evaluate refusal rates and the effectiveness of persuasion-based jailbreaking attempts.

- [evaluation.py](./Code/evaluation.py): assesses model responses to both original and persuasive prompts using Attack Success Rate (ASR) and informativeness scores, and computes Influential Power (IP) to quantify the impact of persuasive strategies across different LLMs.

- [Code/Baselines/](./Code/Baselines/): contains baseline methods for generating adversarial prompts, serving as comparison points to assess the added effectiveness of persuasion-based jailbreak attempts.


