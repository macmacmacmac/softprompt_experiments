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
    formulas = [
        {   #gravitational force between two objects
            'expr': "(6.673e-11) * (x*y)/(z**2)",
            'input_template': "Input: m1={x},m2={y},r={z}\n Answer: ",
            'output_template': "{out:.3e} N"
        },
        {
            #circular orbital velocity
            'expr': "((6.673e-11) * x / y)**0.5",
            'input_template': "Input: M={x}, r={y}\n Answer: ",
            'output_template': "{out:.3e} m/s"
        },
        {
            #e = mc^2
            'expr': "x * (2.998e8**2)",
            'input_template': "Input: m={x} kg\n Answer: ",
            'output_template': "{out:.3e} J"
        },
        {
            #photon energy
            'expr': "(6.626e-34 * 2.998e8) / x",
            'input_template': "Input: λ={x} m\n Answer: ",
            'output_template': "{out:.3e} J"
        },
        {
            #attraction between two charges
            'expr': "(8.99e9) * (x*y)/(z**2)",
            'input_template': "Input: q1={x},q2={y},r={z}\n Answer: ",
            'output_template': "{out:.3e} N"
        },
        {
            #electric potential energy
            'expr': "(8.99e9) * (x*y)/z",
            'input_template': "Input: q1={x},q2={y},r={z}\n Answer: ",
            'output_template': "{out:.3e} J"
        },
        {
            #number of particles from mass
            'expr': "(6.022e23) * x/y",
            'input_template': "Input: m={x} g, M={y} g/mol\n Answer: ",
            'output_template': "{out:.3e} particles"
        },
        {
            #faraday constant definition
            'expr': "6.022e23 * x",
            'input_template': "Input: e={x} C\n Answer: ",
            'output_template': "{out:.3e} C/mol"
        },
        {
            #kinetic energy
            'expr': "0.5*x*(y**2)",
            'input_template': "Input: m={x},v={y}\n Answer: ",
            'output_template': "{out:.3e} J"
        },
        {
            # Wave speed
            'expr': "x*y",
            'input_template': "Input: f={x},λ={y}\n Answer: ",
            'output_template': "{out:.3e} m/s"
        },
        {
            # Ideal gas density
            'expr': "(8.3145) * y * z / x",
            'input_template': "Input: P={x},n={y},T={z}\n Answer: ",
            'output_template': "{out:.3e} kg/m³"
        },
        {
            #volume of an ellipsoid ish
            'expr': "3.1415*x*y*z",
            'input_template': "Input: a={x},b={y},c={z}\n Answer: ",
            'output_template': "{out:.3f}"
        },
    ]
    def expr_to_func(expr):        
        func = np.vectorize(lambda x, y, z: eval(expr))
        return func, expr

    def get_sentences_from_func(func, input_template, output_template):
        x = np.random.randint(low=1, high=10, size=NUM_SAMPLES_PER)
        y = np.random.randint(low=1, high=10, size=NUM_SAMPLES_PER)
        z = np.random.randint(low=1, high=10, size=NUM_SAMPLES_PER)

        outputs = func(x, y, z)

        input_sentences = [
            input_template.format(x=x,y=y,z=z)
            for x,y,z in zip(x,y,z)
        ]

        target_sentences = [output_template.format(out=out) for out in outputs]

        return input_sentences, target_sentences

    # pipeline
    for i, formula in enumerate(formulas):
        save_dir = os.path.join(SAVE_DIR, f"dataset_{i}")
        os.makedirs(save_dir, exist_ok=True)

        expr = formula['expr']
        input_template = formula['input_template']
        output_template = formula['output_template']
        func, expr = expr_to_func(expr)
        input_sentences, target_sentences = get_sentences_from_func(func, input_template, output_template)
        
        print(input_sentences[0], target_sentences[0])

        tokenized = tokenize_and_save(input_sentences, target_sentences, save_dir, expr, tokenizer)

        

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









