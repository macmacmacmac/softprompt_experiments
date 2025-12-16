import torch
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm

from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.utils import (
    get_train_test_from_softprompt_logits, 
    train_softprompt_from_embeds,
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset")
    parser.add_argument("--use_parsability", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    args, _ = parser.parse_known_args(args_list)

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
        print(f"\nFound ({num_datasets}) datasets in directory")
    else:
        raise ValueError("path to directory has no datasets")

    _, test_dataset, train_loader, test_loader = get_train_test_from_softprompt_logits(
        model,
        word_embeddings,
        tokenizer,
        dataset_dirs,
        BATCH_SIZE,
        0.8,
        use_parsability=args.use_parsability
    )

    softprompt = SoftPrompt(
        model=model, 
        tokenizer=tokenizer, 
        word_embeddings=word_embeddings, 
        num_tokens=NUM_TOKENS
    )

    # Suffix to mark end of input
    suffix = "\nOutput: "
    suffix_ids = tokenizer(
        suffix,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids'].to(model.device)
    SUFFIX_LEN = suffix_ids.shape[1]
    suffix_emb = model.get_input_embeddings()(suffix_ids).to(model.dtype).detach()

    train_loss, test_loss = train_softprompt_from_embeds(softprompt, suffix_emb, LR, EPOCHS, train_loader, test_loader, verbose=args.verbose)

    performance = {
        'train loss':train_loss,
        'test_loss':test_loss,
    }
    log_json(os.path.join(SAVE_DIR,'softprompt_performance.json'), performance)

    softprompt.save_softprompt(SAVE_DIR)

    dtype = model.dtype
    device = model.device
    outputs = []
    for full_embeds, labels in test_dataset:
        # full_ids contains a sequence of [inputs;targets;padding]
        # labels masks out the inputs [mask;targets;padding]
        # we need to index the input_ids so we're only using the inputs
        # for generations without the target so we're not snooping ahead
        full_embeds = full_embeds.to(device)
        labels = labels.to(device)        

        target_idxs = (labels != -100).to(device)
        input_idxs = (labels == -100).to(device)

        hardprompt_ids = labels[target_idxs]
        input_embeds = full_embeds[input_idxs].unsqueeze(0).to(dtype=dtype) #[1, seq_len-target_len, seq_dim]
        input_embeds = torch.cat([input_embeds, suffix_emb], dim=1)
        max_new_tokens = len(full_embeds) - len(full_embeds[input_idxs])
        generation = softprompt.generate_from_embeds(
            input_embeds,
            max_new_tokens=max_new_tokens
        )[0]
        full_sequence = tokenizer.decode(hardprompt_ids, skip_special_tokens=True)

        output = f"Hard prompt: {full_sequence}\Prediction: {generation}"
        print(output)
        outputs.append(output)


    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









