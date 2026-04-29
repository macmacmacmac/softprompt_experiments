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
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8b-Instruct")
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = args.model_name
    NUM_SAMPLES_PER = args.num_samples_per_dataset
    # NUM_DATASETS = args.num_datasets
    SAVE_DIR = args.save_directory

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # generate dataset
    formulas = [
        { 
            'expr': "(max(x,y,z))", #max
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "(max(x,y,z))", #max
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "(max(x,y,z))", #max
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "(max(x,y,z))", #max
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "(max(x,y,z))", #max
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "(min(x,y,z))", #min
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },

        { 
            'expr': "(min(x,y,z))", #min
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },

        { 
            'expr': "(min(x,y,z))", #min
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },

        { 
            'expr': "(min(x,y,z))", #min
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },

        { 
            'expr': "(min(x,y,z))", #min
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "x+y+z", #sum
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "x+y+z", #sum
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "x+y+z", #sum
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "x+y+z", #sum
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },
        { 
            'expr': "x+y+z", #sum
            'high':10,
            'input_template': "Task: Find the correct number from this list. Input: [{x}, {y}, {z}] Output:",
            'output_template': "{out}"
        },


    ]
    def expr_to_func(expr):        
        func = np.vectorize(lambda x, y, z: eval(expr))
        return func, expr

    def get_sentences_from_func(func, input_template, output_template, high, num_samples):
        # x = np.random.randint(low=1, high=1000, size=num_samples)
        # y = np.random.randint(low=1, high=1000, size=num_samples)
        # z = np.random.randint(low=1, high=1000, size=num_samples)

        triples = set()

        while len(triples) < num_samples:
            triple = (
                np.random.randint(1, high),
                np.random.randint(1, high),
                np.random.randint(1, high),
            )
            triples.add(triple)

        x, y, z = map(np.array, zip(*triples))

        outputs = func(x, y, z)

        input_sentences = [
            input_template.format(x=x,y=y,z=z)
            for x,y,z in zip(x,y,z)
        ]

        target_sentences = [
            output_template.format(out=out) 
            for out in outputs
        ]

        return input_sentences, target_sentences

    # pipeline
    for i, formula in enumerate(formulas):
        save_dir = os.path.join(SAVE_DIR, f"dataset_{i}")
        os.makedirs(save_dir, exist_ok=True)

        expr = formula['expr']
        input_template = formula['input_template']
        output_template = formula['output_template']
        high = formula["high"]
        func, expr = expr_to_func(expr)
        num_vars = ("x" in expr) + ("y" in expr) + ("z" in expr)
        input_sentences, target_sentences = get_sentences_from_func(
            func, input_template, output_template, high, NUM_SAMPLES_PER
        )
        
        print(input_sentences[0], target_sentences[0])

        tokenized = tokenize_and_save(input_sentences, target_sentences, save_dir, expr, tokenizer)

        for idx in range(3):
            labels = tokenized['labels'][idx]
            full_ids = tokenized['input_ids'][idx]
            mask = (labels==-100)
            antimask = (labels!=-100)

            tokenized_text = full_ids[mask]
            tokenized_label = full_ids[antimask]

            print(f"Input tokens len: {len(tokenized_text)}")
            print(f"Label tokens len: {len(tokenized_label)}")
            input_text = tokenizer.decode(tokenized_text)
            label_text = tokenizer.decode(tokenized_label)

            print(f"<input>{input_text}</input>\n<label>{label_text}</label>")


    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









