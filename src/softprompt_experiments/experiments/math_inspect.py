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
    parser.add_argument("--save_directory", type=str, default="./datasets/math_physics2")
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
    
    def extract_hidden(emb, tokenized_text, layer_idx=16):
        soft_prompt_len = 8   # number of soft prompt tokens

        input_embs = word_embeddings(tokenized_text).unsqueeze(0)
        full_embs = torch.cat([emb, input_embs], dim=1)

        with torch.no_grad():
            out = model(inputs_embeds=full_embs,attention_mask=None)

            # hidden_states is a tuple: (embeddings, layer1, layer2, ...)
            # Each tensor: [batch, seq_len, hidden_dim]
            hidden_states = out.hidden_states

            # Extract soft prompt hidden states at layer ℓ
            # Assume soft prompt tokens are at positions [0 : K]
            soft_h = hidden_states[layer_idx][:, :soft_prompt_len, :].clone()
        
        return soft_h
    
    def patch(soft_h, layer_idx=16):
        target_prompt = (
            "Task: sort numbers | arrange numbers in ascending order, "
            "Task: translate English to French | convert English sentences into French, "
            "Task: 1 2 3 4 | "
        )

        inputs_tgt = tokenizer(target_prompt, return_tensors="pt")

        # Identify token indices corresponding to xxxx
        xxxx_token_ids = tokenizer("1 2 3 4 ", add_special_tokens=False)["input_ids"]
        xxxx_positions = []

        for i in range(len(inputs_tgt["input_ids"][0]) - len(xxxx_token_ids) + 1):
            if inputs_tgt["input_ids"][0][i:i+len(xxxx_token_ids)].tolist() == xxxx_token_ids:
                xxxx_positions = list(range(i, i + len(xxxx_token_ids)))
                break

        assert len(xxxx_positions) == 8

        def make_patch_hook(soft_h, positions):
            def hook(module, input, output):
                # output: [batch, seq_len, hidden_dim]
                output = output.clone()
                output[:, positions, :] = soft_h
                return output
            return hook

        handle = model.transformer.h[layer_idx].register_forward_hook(
            make_patch_hook(soft_h, xxxx_positions)
        )

        with torch.no_grad():
            generated = model.generate(
                **inputs_tgt,
                max_new_tokens=50,
                do_sample=False
            )

            handle.remove()

        print(tokenizer.decode(generated[0], skip_special_tokens=True))



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
        softprompt_emb = SoftPrompt(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=os.path.join(dataset_dir,'softprompt.pt')
        ).forward()

        results = {}
        results['hardprompt'] = hardprompt
        print(f"\n--------------------------Actual hardprompt: {hardprompt}--------------------------\n")
        random_idxs = torch.randint(0, len(test_dataset), (args.num_samples,))
        for idx in random_idxs:
            labels = test_dataset[idx][1].to(model.device)
            full_ids = test_dataset[idx][0].to(model.device)
            mask = (labels==-100).to(model.device)
            
            tokenized_text = full_ids[mask].to(model.device)

            soft_h = extract_hidden(softprompt_emb, tokenized_text)
            patch(soft_h)

        

    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









