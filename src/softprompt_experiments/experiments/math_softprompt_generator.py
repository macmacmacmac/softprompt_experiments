import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments import SoftPrompt
from softprompt_experiments.utils import get_train_test_loaders, train_softprompt

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--num_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset")
    args = parser.parse_args(args_list)

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_DIR = args.save_directory
    LR = args.lr
    EPOCHS = args.epochs
    NUM_TOKENS = args.num_tokens
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
        print(f"found ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    for dataset_dir in dataset_dirs:
        train_dataset, test_dataset, train_loader, test_loader = get_train_test_loaders(
            dataset_dir,
            BATCH_SIZE,
            train_portion = 0.8
        )

        # subclasses nn.Module, forward call will get the prompt embeddings
        softprompt = SoftPrompt(
            model, 
            tokenizer, 
            word_embeddings, 
            path_to_model=dataset_dir, 
            num_tokens=NUM_TOKENS
        )
        
        train_softprompt(softprompt, LR, EPOCHS, train_loader, test_loader, verbose=True)

        softprompt.save_softprompt(dataset_dir)

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









