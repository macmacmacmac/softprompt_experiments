import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    train_softprompt_from_tokenized,
    eval_softprompt,
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
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset_mini")
    parser.add_argument("--max_new_tokens", type=int, default=50)
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
        train_dataset, test_dataset, train_loader, test_loader = get_train_test_from_tokenized(
            dataset_dir,
            BATCH_SIZE,
            train_portion = 0.8
        )
        hardprompt = torch.load(
            os.path.join(dataset_dir,'dataset.pt'),
            weights_only=False
        )['hardprompt']
        softprompt = SoftPrompt(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=dataset_dir
        )

        print(f"Actual hardprompt: {hardprompt}\n\n\n")
        random_idxs = torch.randint(0, len(test_dataset), (args.num_samples,))

        #just softprompt
        for idx in random_idxs:
            soft_gen = softprompt.generate_from_embeds(embeds=None, max_new_tokens=50)[0]
            print(f"Actual hardprompt: {hardprompt}\nSoftprompt by itself: {soft_gen}\n")

        #standard
        gen_prompt = ""
        for idx in random_idxs:
            labels = test_dataset[idx][1].to(model.device)
            full_ids = test_dataset[idx][0].to(model.device)
            mask = (labels==-100).to(model.device)
            
            tokenized_text = full_ids[mask].to(model.device)
            input_text = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            input_embed = word_embeddings(tokenized_text).unsqueeze(0)

            soft_gen = softprompt.generate_from_embeds(embeds=input_embed, max_new_tokens=50, suffix_str=gen_prompt)[0]
            print(f"Actual hardprompt: {hardprompt}\nSoftprompt explanation: {input_text}{gen_prompt}{soft_gen}")

        #unconditioned on output
        gen_prompt = "First, I should"
        for idx in random_idxs:
            labels = test_dataset[idx][1].to(model.device)
            full_ids = test_dataset[idx][0].to(model.device)
            mask = (labels==-100).to(model.device)
            
            tokenized_text = full_ids[mask].to(model.device)
            input_text = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            input_embed = word_embeddings(tokenized_text).unsqueeze(0)

            soft_gen = softprompt.generate_from_embeds(embeds=input_embed, max_new_tokens=50, suffix_str=gen_prompt)[0]
            print(f"Actual hardprompt: {hardprompt}\nSoftprompt explanation{input_text}{gen_prompt}{soft_gen}")

        #conditioned on output
        gen_prompt = "\nExplanation: "
        for idx in random_idxs:
            tokenized_text = test_dataset[idx][0].to(model.device)
            input_text = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            input_embed = word_embeddings(tokenized_text).unsqueeze(0)

            soft_gen = softprompt.generate_from_embeds(embeds=input_embed, max_new_tokens=50, suffix_str=gen_prompt)[0]
            print(f"Actual hardprompt: {hardprompt}\nSoftprompt explanation (conditioned on answer): {input_text}{gen_prompt}{soft_gen}")

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









