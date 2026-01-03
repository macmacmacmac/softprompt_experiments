import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm
import json 
import numpy as np

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    train_softprompt_from_tokenized,
    eval_softprompt,
    eval_softprompt_regression,
    log_json
)

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset")
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_DIR = args.save_directory
    BATCH_SIZE = args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype
    ).to(device)
    model.eval()
    word_embeddings = model.get_input_embeddings()

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

    for dataset_dir in tqdm(dataset_dirs):
        # load dataset
        train_dataset, test_dataset, train_loader, test_loader = get_train_test_from_tokenized(
            dataset_dir,
            BATCH_SIZE,
            train_portion = 0.8
        )
        
        # print("Initial tokens: ", init)
        softprompt = SoftPrompt(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=os.path.join(dataset_dir, "softprompt.pt")
        )
        
        hardprompt = torch.load(
            os.path.join(dataset_dir,'dataset.pt'),
            weights_only=False
        )['hardprompt']

        outputs = eval_softprompt_regression(softprompt, test_dataset, dataset_dir)
        performance = {
            'hardprompt':hardprompt,
            'outputs': outputs
        }
        print(dataset_dir, performance)
        # log_json(os.path.join('softprompt_performance.json'), performance)

        with open(os.path.join(dataset_dir,'explanations.json')) as f:
            data = json.load(f)
        data['performance'] = outputs
        log_json(os.path.join(dataset_dir, "explanations.json"), data)

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









