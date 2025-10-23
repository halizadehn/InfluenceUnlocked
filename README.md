# Influence Unlocked: Persuasion-Aware Jailbreaking and the Hidden Persuasive Fingerprints of Large Language Models

This repository contains the implementation and evaluation framework for studying persuasion-based jailbreak attacks on Large Language Models (LLMs). The project operationalizes Cialdini’s persuasion principles to generate adversarial prompts and analyze model-specific susceptibility profiles.

## Pipeline overview
<img src="./Figures/framework.jpg" alt="overview" style="width:400px;"/>



## Datasets

- **AdvBench**: This dataset contains 520 queries, covering various types of harmful behavior. 
- **StrongREJECT**: This dataset encompasses six major categories of behaviors that are explicitly prohibited under all major LLM usage policies:
  (1) illegal goods and services, (2) non-violent crimes, (3) hate, harassment, and discrimination, (4) disinformation and deception, (5) violence, and (6) sexual content. 

## Victim Models

- Vicuna-7b
- Llama2-7b-chat
- Llama3
- DeepSeek-R1
- Gemma3
- Phi4

> **Note:** Target models are available via [Ollama](https://ollama.com/).

## Attack Baselines
- **[GCG](https://github.com/llm-attacks/llm-attacks)**: finds suffixes via gradient synthesis
- **[PAIR](https://arxiv.org/pdf/2310.08419)**: an algorithm that creates semantic jailbreaks using only black-box access to an LLM
- **[PAP](https://github.com/CHATS-lab/persuasive_jailbreaker)**: generates persuasive adversarial prompts using different strategies

  > **Note:** We leveraged baseline implementations provided by the [StrongReject benchmark](https://github.com/dsbowen/strong_reject).

## Defense Baselines
- **[Mutation-Based](https://arxiv.org/pdf/2309.00614)**: These methods mitigate harmful behavior by perturbing the input while preserving its meaning. We implement Rephrase and Retokenize strategies, which alter sentence structure or tokenization to disrupt adversarial intent.
- **[Detection-Based](https://aclanthology.org/2024.acl-long.568.pdf)**: These approaches identify adversarial prompts before inference through random perturbations such as Rand-Drop, Rand-Insert, Rand-Swap, and Rand-Patch, which test model stability by modifying input tokens or sequences.

## Code Structure

- [persuasive_prompt_generation.py](./Code/persuasive_prompt_generation.py): Generates persuasive variants of harmful queries based on Cialdini’s persuasion principles. These variants are used to analyze how different persuasion strategies influence LLM compliance with harmful instructions.

- [get_model_responses.py](./Code/get_model_responses.py): Handles the process of querying LLMs with both original and persuasion-based prompts, collecting model outputs to measure refusal behavior and susceptibility to persuasion-driven jailbreaks.

- [evaluation.py](./Code/evaluation.py): Evaluates collected responses using metrics such as Attack Success Rate (ASR), Informativeness Score (IS), and Influence Power (IP) to quantify the effectiveness and impact of various persuasion strategies across models.

- [Code/Attack baselines/](./Code/Baselines/): Contains baseline implementations of traditional adversarial prompt generation techniques, serving as benchmarks to compare against persuasion-aware jailbreak methods.
  
- [Code/Defense baselines/](./Code/Baselines/): Includes implementations of mutation-based and detection-based defense strategies used to evaluate the robustness of LLMs against persuasion-based attacks.


