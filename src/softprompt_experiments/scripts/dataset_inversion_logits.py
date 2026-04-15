import torch
import copy
import numpy as np
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.utils import tokenize_and_save, batched_tokenize_and_save
import pandas as pd

import pickle
from datasets import load_dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

import glob
import torch

import torch.nn as nn
import torch.nn.functional as F

class LogitsDataset(torch.utils.data.Dataset):
    def __init__(self, save_dir):
        self.files = sorted(glob.glob(f"{save_dir}/shard_*.pt"))
        self.shard_sizes = []
        for f in self.files:
            tensor = torch.load(f, map_location='cpu')
            self.shard_sizes.append(tensor.size(0))
        self.cumsum = [0] + list(torch.cumsum(torch.tensor(self.shard_sizes), dim=0).numpy())
        self.total_size = self.cumsum[-1]

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Find which shard contains idx
        shard_idx = max(i for i in range(len(self.cumsum)) if self.cumsum[i] <= idx) 
        local_idx = idx - self.cumsum[shard_idx]
        tensor = torch.load(self.files[shard_idx], map_location='cpu')
        return tensor[local_idx]
    
def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_directory", type=str, default="./datasets/inversion_dataset/instructions_logits_dataset")
    parser.add_argument("--num_tokens", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    args, _ = parser.parse_known_args(args_list)
    
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    # NUM_DATASETS = args.num_datasets
    SAVE_DIR = args.save_directory
    NUM_TOKENS = args.num_tokens
    EPOCHS = args.epochs
    BATCHSIZE = args.batchsize
    LR = args.lr

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    word_embeddings = model.get_input_embeddings()
    vocab_size = word_embeddings.num_embeddings
    dataset = load_dataset("wentingzhao/one-million-instructions", split="train", streaming=True)
    def collate_fn(batch):
        texts = [x["user"] for x in batch]
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # adjust
            return_tensors="pt"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        collate_fn=collate_fn
    )

    def get_last_token_logits(logits, attention_mask):
        # logits: [B, T, V]
        # attention_mask: [B, T]

        # lengths = number of non-pad tokens
        lengths = attention_mask.sum(dim=1) - 1  # [B]

        # gather indices
        B = logits.size(0)
        V = logits.size(-1)

        # shape → [B, 1, V]
        last_logits = logits[torch.arange(B), lengths]

        return last_logits  # [B, V]
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    shard = []
    shard_size = 31
    shard_idx = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits  # [B, T, V]

            target_logits = get_last_token_logits(
                logits,
                batch["attention_mask"]
            ).to(torch.bfloat16)

            shard.append(target_logits.cpu())

            if len(shard) >= shard_size:
                print(f"saving shard to {SAVE_DIR}")
                shard_tensor = torch.cat(shard, dim=0)

                torch.save(
                    shard_tensor,
                    f"{SAVE_DIR}/shard_{shard_idx:05d}.pt"
                )

                shard = []
                shard_idx += 1
    
    
    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









