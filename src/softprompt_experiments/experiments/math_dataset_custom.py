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

from softprompt_experiments.utils import tokenize_and_save

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples_per_dataset", type=int, default=500)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset_custom")
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    NUM_SAMPLES_PER = args.num_samples_per_dataset
    # NUM_DATASETS = args.num_datasets
    SAVE_DIR = args.save_directory

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # generate dataset
    exprs = [
        "(6.673e-11) * (x*y)/(z**2)",
        "(8.99e9) * (x*y)/(z**2)",
        # "(4/3) * (3.14) * x * y * z",
        # "(x**2 + y**2 + z**2)**(1/2)",
        # "(x%2) + (y%2) + (z%2)"
    ]
    def expr_to_func(expr):        
        func = np.vectorize(lambda x, y, z: eval(expr))
        return func, expr

    def get_sentences_from_func(func):
        x = np.random.randint(low=1, high=10, size=NUM_SAMPLES_PER)
        y = np.random.randint(low=1, high=10, size=NUM_SAMPLES_PER)
        z = np.random.randint(low=1, high=10, size=NUM_SAMPLES_PER)

        outputs = func(x, y, z)

        input_sentences = [
            # f"Input: ({x},{y},{z})\nOutput: " 
            f"Input: m1={x},m2={y},r={z}\n Answer: "
            for x,y,z in zip(x,y,z)
        ]

        # target_sentences = [f"{out:.2f}" for out in outputs]
        target_sentences = [f"{out:.3e} N" for out in outputs]

        return input_sentences, target_sentences

    # pipeline
    for i, expr in enumerate(exprs):
        save_dir = os.path.join(SAVE_DIR, f"dataset_{i}")
        os.makedirs(save_dir, exist_ok=True)

        func, expr = expr_to_func(expr)
        input_sentences, target_sentences = get_sentences_from_func(func)
                
        tokenized = tokenize_and_save(input_sentences, target_sentences, save_dir, expr, tokenizer)

        

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









