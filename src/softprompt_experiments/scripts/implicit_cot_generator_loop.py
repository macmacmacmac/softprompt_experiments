import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
import argparse
import transformers
from typing import Any, Literal, Optional, Union
import os
from tqdm.auto import tqdm

from copy import deepcopy

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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--num_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_directory", type=str, default="./datasets/implicit_cot_science")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.set_defaults(auto_split=True)

    args, _ = parser.parse_known_args(args_list)

    SAVE_DIR = args.save_directory
    LR = args.lr
    EPOCHS = args.epochs
    NUM_TOKENS = args.num_tokens
    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens

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

    # Load the implicit CoT model
    MODEL_NAME = 'yuntian-deng/implicit-cot-math-mistral7b'
    # MODEL_NAME = 'mistralai/Mistral-7B-v0.1'
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model.config.pad_token_id = tokenizer.pad_token_id

    base_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    base_model.eval()

    for dataset_dir in tqdm(dataset_dirs):
        #TODO: Initialize peft model
        PROMPT_TUNING_CONFIG = PromptTuningConfig(
            peft_type="PROMPT_TUNING",
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=NUM_TOKENS,
            prompt_tuning_init="SAMPLE_VOCAB",
            tokenizer_name_or_path=MODEL_NAME,
        )

        model = get_peft_model(base_model, PROMPT_TUNING_CONFIG)

        #TODO: Load dataset
        train_dataset, test_dataset, train_loader, test_loader = get_train_test_from_tokenized(
            dataset_dir,
            BATCH_SIZE,
            train_portion = 0.8,
        )
        device = model.device

        #TODO: Set up training loop
        # Only train the softprompt parameters
        optimizer = torch.optim.AdamW(model.prompt_encoder.parameters(), lr=LR)

        final_train_loss = 0.0
        final_test_loss = 0.0
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for i, batch in enumerate(train_loader):
                input_ids, labels = [b.to(device) for b in batch]
                batchsize = input_ids.size(0)

                # HF autoregressive LM loss
                loss = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=(input_ids != tokenizer.pad_token_id)
                ).loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                
            # ---- evaluation ----
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                entropy = []
                for batch in test_loader:
                    input_ids, labels = [b.to(device) for b in batch]
                    batchsize = input_ids.size(0)

                    # HF autoregressive LM loss
                    loss = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=(input_ids != tokenizer.pad_token_id)
                    ).loss
                    test_loss += loss.item()
                final_train_loss = train_loss/len(train_loader)
                final_test_loss = test_loss/len(test_loader)        
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Train Loss: {final_train_loss:.4f} | "
                    f"Test Loss: {final_test_loss:.4f} | "
                )
        model.save_pretrained(os.path.join(dataset_dir, "peft_softprompt"))

        #TODO: Do evaluation
        outputs = []
        with torch.no_grad():
            model.generation_config.pad_token_id = tokenizer.eos_token_id
            for full_ids, labels in tqdm(test_dataset, desc="generating sample outputs..."):
                # full_ids contains a sequence of [inputs;targets;padding]
                # labels masks out the inputs [mask;targets;padding]
                # we need to index the input_ids so we're only using the inputs
                # for generations without the target so we're not snooping ahead
                full_ids = full_ids.to(device)
                labels = labels.to(device)        

                target_idxs = (labels != -100).to(device)
                input_idxs = (labels == -100).to(device)
                only_input_ids = full_ids[input_idxs].unsqueeze(0) #[1, seq_len-target_len]
                only_target_ids = full_ids[target_idxs]
                target = tokenizer.decode(only_target_ids, skip_special_tokens=True)
                full_sequence = tokenizer.decode(full_ids, skip_special_tokens=True)

                generation = model.generate(
                    input_ids=only_input_ids,
                    attention_mask=torch.ones((1,input_ids.shape[1]),device=device),
                    max_new_tokens=MAX_NEW_TOKENS
                )[0]

                gneration_text = tokenizer.decode(generation, skip_special_tokens=True)

                explanation_prompt = tokenizer.decode(only_input_ids[0], skip_special_tokens=True) + "First, I should"
                explanation_ids = tokenizer(explanation_prompt, return_tensors='pt',add_special_tokens=False)['input_ids'].to(device)

                explanation = model.generate(
                    input_ids=explanation_ids,
                    attention_mask=torch.ones((1,input_ids.shape[1]),device=device),
                    max_new_tokens=MAX_NEW_TOKENS,
                )[0]

                explanation_text = tokenizer.decode(explanation, skip_special_tokens=True)

                output = f"Full sequence: {full_sequence}\nActual Target: {target}, Generation: {gneration_text}\n Explanation: {explanation_text}"
                print(output)
                outputs.append(output)
        
        #TODO: Log shit
        hardprompt = torch.load(
            os.path.join(dataset_dir,'dataset.pt'),
            weights_only=False
        )['hardprompt']
        performance = {
            'hardprompt':hardprompt,
            'train loss':final_train_loss,
            'test_loss':final_test_loss,
            'outputs':outputs,
        }

        dump_dir = os.path.join(dataset_dir,'softprompt_performance.json')
        log_json(dump_dir, performance)
        print(f"Logged performance to {dump_dir}")

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )


