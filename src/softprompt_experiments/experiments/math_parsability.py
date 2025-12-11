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
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset_mini")
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
        squishyprompt = SquishyPrompt(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=dataset_dir
        )

        gen_prompt = "| Explain what the text before '|' say: "
        # gen_prompt = ". This is because "

        tokenized_text = tokenizer(gen_prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(model.device)
        input_embed = word_embeddings(tokenized_text)
        soft_gen = softprompt.generate_from_embeds(input_embed, max_new_tokens=50)
        squishy_gen = squishyprompt.generate_from_embeds(input_embed, max_new_tokens=50)

        # input_text = tokenizer.decode(test_dataset[0][0], skip_special_tokens=True)
        # tokenized_text = test_dataset[0][0].to(model.device)
        # input_embed = word_embeddings(tokenized_text).unsqueeze(0)
        soft_gen = softprompt.generate_from_embeds(input_embed, max_new_tokens=50, suffix_str=gen_prompt)[0]
        squishy_gen = squishyprompt.generate_from_embeds(input_embed, max_new_tokens=50, suffix_str=gen_prompt)[0]

        soft_logits, _ = softprompt.get_nearest_to_logits(3)
        squishy_logits, _ = squishyprompt.get_nearest_to_logits(3)

        _, soft_nearest = softprompt.get_nearest_to_embeds()
        _, squishy_nearest = squishyprompt.get_nearest_to_embeds()
        soft_initial = tokenizer.decode(softprompt.initial_tokens, skip_special_tokens=True)
        squishy_initial = tokenizer.decode(softprompt.initial_tokens, skip_special_tokens=True)

        print(
            f"Actual hardprompt: {hardprompt}\n"
            f"Softprompt gen: {gen_prompt}{soft_gen}\n"
            f"Squishyprompt gen: {gen_prompt}{squishy_gen}\n\n"
            f"Soft top k logits:  {soft_logits}\n"
            f"Squishy top k logits:  {squishy_logits}\n\n"
            f"Soft initial: {soft_initial}\n"
            f"Squishy initial: {squishy_initial}\n\n"
            f"Soft k nearest neighbors: {soft_nearest}\n"
            f"Squishy k nearest neighbors: {squishy_nearest}\n\n"
        )

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









