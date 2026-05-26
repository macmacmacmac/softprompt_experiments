# Soft Prompt Interpretability Experiments
## Abstract

Soft prompting or prompt tuning refers to a parameter efficient tuning technique where the base LLM is frozen and a set of trainable embeddings are prepended to the input sequence. In this work, we demonstrate two findings. First (1): LLMs natively have a basic internal comprehension of soft prompts and are capable of verbalizing explanations of certain soft prompts in natural language, revealing activated concepts. Second (2): these verbalized explanations reveal how soft prompts can implicitly extract and utilize an LLM’s latent scientific knowledge. This indicates that soft prompting can act as a loosely "science informed" learning algorithm, capable of drawing upon the vast quantity of scientific literature ingested by LLMs during its pre-training to inform its predictions. 

## Command for setting up Locally
Setup Virtual Environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

## Command for reproducing experiments
Example Usage 
```bash
python -m run_experiment --scripts \
  soft_prompt_mapper.supernat_instruct_DoD.train_softprompts \
  soft_prompt_mapper.supernat_instruct_DoD.compile_mapper_dataset \
  soft_prompt_mapper.supernat_instruct_DoD.train_mapper \
  soft_prompt_mapper.supernat_instruct_DoD.test_mapper
```
This will:
1. Train softprompts
2. Compile trained softprompts into a dataset for translator
3. Train the translator
4. Test the translator

You can also call these scripts individually like
```bash
python -m run_experiment --scripts soft_prompt_mapper.supernat_instruct_DoD.train_softprompts
```

## Misc

All our scripts use the word 'mapper' interchangeably with 'translator'

## Scripts for Classification DoD


## Scripts for General DoD
