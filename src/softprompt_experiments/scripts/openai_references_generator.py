import torch
import copy
import numpy as np
import argparse
import os
from transformers import (
    AutoTokenizer,
    # AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.utils import tokenize_and_save, log_json
from softprompt_experiments.models.openaimodel import OpenAIModel

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_directory", type=str, default="./datasets/math_datasetv2")
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_DIR = args.save_directory

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    # Get dataset sub directories
    dataset_dirs = []
    for entry in os.scandir(SAVE_DIR):
        if entry.is_dir():  # Check if the entry is a directory
            if "dataset_" in entry.name:
                dataset_dirs.append(entry.path)

    num_datasets = len(dataset_dirs)
    if num_datasets > 0:
        print(f"\nFound ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    openai_model = OpenAIModel(
        "gpt-4.1-mini",
        "You are an expert reference sentences generator for math problems."
    )

    prompt_prefix = (
        "# TASK:\n"
        "I need you to look at a math problem and generate three (3) concise reference "
        "sentences that describe chain-of-thought (CoT) like steps for how you would solve "
        "the given math problem. For example, given '1x + 2y' you might say: 'First, I should "
        "multiply x by 1, multiply y by 2, and then add everything together', etc. "
        "Try to make the reference sentences varied but semantically accurate to the problem."
        "You will now be shown the math expression below.\n"
        "\""
    )
    prompt_suffix = (
        "\""
        "This concludes the other LLM's response.\n"
        "Give your answer as a dictionary like so with nothing else:\n"
        "{'Reference1': <reference sentence here>, "
        "'Reference2': <reference sentence here>, "
        "'Reference3': <reference sentence here>"
        "}\n"
        "# ANSWER:\n"
    )

    for dataset_dir in tqdm(dataset_dirs):
        # load hardprompt
        loaded = torch.load(os.path.join(dataset_dir, "dataset.pt"), weights_only=False)
        hard_prompt = loaded['hardprompt']

        # generate reference sentences
        prompt = prompt_prefix + hard_prompt + prompt_suffix
        references = openai_model.pred(prompt)


    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









