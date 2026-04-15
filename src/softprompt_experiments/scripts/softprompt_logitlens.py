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
    log_json
)

import json
import matplotlib.pyplot as plt

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
    parser.add_argument("--show_target", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--no_auto_split",dest="auto_split",action="store_false")
    parser.set_defaults(auto_split=True)

    args, _ = parser.parse_known_args(args_list)

    MODEL_NAME = args.model_name
    SAVE_DIR = args.save_directory
    BATCH_SIZE = args.batch_size
    AUTO_SPLIT = args.auto_split

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
    unembedding_matrix = model.get_output_embeddings().weight

    # Get all the tokens that correspond to natural numbers which are only 1 token long
    natural_number_tokens = []
    still_only_one_token_long = True
    num = 0
    while still_only_one_token_long:
        tokenized_num = tokenizer(str(num), add_special_tokens=False).input_ids
        print("THIS ISM TOKENIZED NUM", tokenized_num)
        still_only_one_token_long = len(tokenized_num) == 1

        if still_only_one_token_long:
            natural_number_tokens.append(tokenized_num[0])
            num += 1
    print(natural_number_tokens)


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
            train_portion = 0.8,
            auto_split=AUTO_SPLIT
        )

        has_perf_file = False
        try:
            with open(os.path.join(dataset_dir,'softprompt_performance.json')) as f:
                soft_perf = json.load(f)
            has_perf_file = True

            entropy = soft_perf['entropy']

            pearson_r = None
            accuracy = None
            if "pearson_r" in soft_perf:
                pearson_r = soft_perf['outputs']['pearson_r']
            elif "accuracy" in soft_perf:
                accuracy = soft_perf['outputs']['accuracy']
        except FileNotFoundError:
            print("Directory doesn't have a softprompt_performance.json file inside"
                  "likely because evalautions werent enabled.\n"
                  "Skipping logging...")

        if AUTO_SPLIT:
            hardprompt = torch.load(
                os.path.join(dataset_dir,'dataset.pt'),
                weights_only=False
            )['hardprompt']
        else:
            hardprompt = torch.load(
                os.path.join(dataset_dir,'train_dataset.pt'),
                weights_only=False
            )['hardprompt']
        softprompt = SoftPrompt(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=os.path.join(dataset_dir,'softprompt.pt')
        )

        results = {}
        results['hardprompt'] = hardprompt
        print(f"\n--------------------------Actual hardprompt: {hardprompt}--------------------------\n")
        if has_perf_file:
            print(f"|=== Entropy: {entropy}")
            if pearson_r is not None:
                print(f"|=== Pearson R: {pearson_r}")
            if accuracy is not None:
                print(f"|=== Accuracy: {accuracy}")

        random_idxs = torch.randint(0, len(test_dataset), (args.num_samples,))

        for i, idx in enumerate(random_idxs):
            #TODO: run input through model
            labels = test_dataset[idx][1].to(model.device)
            full_ids = test_dataset[idx][0].to(model.device)
            mask = (labels==-100).to(model.device)

            tokenized_text = full_ids[mask].to(model.device)
            # tokenized_text = torch.cat(
            #     [tokenized_text, tokenizer(" ", return_tensors='pt',add_special_tokens=False).input_ids.to(device).squeeze(0)], 
            #     dim=-1
            # )
            input_text = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            # print(f"full \"{tokenizer.decode(full_ids)}\"")
            # print(f"inp \"{tokenizer.decode(tokenized_text)}\"")
            input_embed = word_embeddings(tokenized_text).unsqueeze(0)
            full_embs = torch.cat([softprompt.forward(), input_embed], dim=1)

            antimask = (labels!=-100).to(model.device)
            tokenized_target = full_ids[antimask].to(model.device)
            target_number_str = tokenizer.decode(tokenized_target, skip_special_tokens=True)

            attention_mask = torch.ones(full_embs.size()[:-1], device=input_embed.device, dtype=torch.long)
            outputs = model(
                inputs_embeds=full_embs,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Get logits for last token in batch 0
            logits_last = outputs.logits[0, -1, :]  # shape: (vocab_size,)

            # Get top 1
            pred_id = torch.argmax(logits_last).item()
            print(f"<logit last>{tokenizer.decode([pred_id])}")
             
            #TODO: collect logits at each layer
            #      first element is just the embedding layer, so skip
            hidden_states = outputs.hidden_states

            #TODO: pass them through the logit lens projector
            #      just multiply them by the word embedding matrix, but only over the numbers token
            #      softmax normalize

            layer_probabilities = []
            for hidden_state in enumerate(hidden_states):
                print(f"original unembedding matrix {unembedding_matrix.shape}")
                only_number_unembedding_matrix = unembedding_matrix[natural_number_tokens].squeeze(1)
                print(f"original only_number_unembedding_matrix matrix {only_number_unembedding_matrix.shape}")
                last_token_hidden = hidden_state[1][:,-1,:]
                print(f"lsat token hidden shape: {last_token_hidden.shape}")
                unembeddings = last_token_hidden @ only_number_unembedding_matrix.T

                layer_prob = torch.nn.functional.softmax(unembeddings, dim=-1)
                print(f"this is layer shape: {layer_prob.shape}")
                layer_probabilities.append(layer_prob)

                k_most_likely = 10

                # Take batch index 0 for simplicity
                probs = layer_prob[0]           # (K,)
                top_probs, top_indices = torch.topk(probs, k=k_most_likely)
                top_numbers = [natural_number_tokens[i] for i in top_indices]
                for num, prob in zip(top_numbers, top_probs):
                    print(f"Number: {num}, Probability: {prob.item():.4f}")

            #TODO: visualize the probabilities as a heat map.
            #      on the x axis is the layers
            #      on the y axis is the number tokens
            #      (let's say the target_number + 10 and the target_number - 10 for our min max range)
            k = 20
            target_number = int(target_number_str)

            probs_per_layer = torch.concatenate(layer_probabilities, dim=0)  # (num_layers, K)
            # print(f"This is probs per layer shape {probs_per_layer.shape}")

            numbers_to_show = list(range(max(0,target_number - k), target_number + k + 1))
            tokenized_numbers_to_show = [
                tokenizer(str(num),add_special_tokens=False).input_ids[0]
                for num in numbers_to_show
            ]

            numbers_to_show_filtered = [
                n for n, tok in zip(numbers_to_show, tokenized_numbers_to_show)
            ]

            # Create mapping from token ID → index in your subset
            token_to_index = {tok: i for i, tok in enumerate(natural_number_tokens)}
            # Filter numbers that exist in your subset
            token_indices = [
                token_to_index[tok] 
                for tok in tokenized_numbers_to_show 
                if tok in token_to_index
            ]
            probs_to_show = probs_per_layer[:, token_indices].detach().cpu().float().numpy()            
            # print(f"\n\nThis is probs to show shape: {probs_to_show.shape}")
            # print(f"\n\nThis is probs to show: {probs_to_show}")

            plt.figure(figsize=(12,6))
            plt.imshow(probs_to_show.T, aspect='auto', cmap='viridis')
            plt.colorbar(label='Probability')
            plt.xlabel('Layer number')
            plt.ylabel('Probability of decoding to this number token')
            # plt.yticks(ticks=range(len(numbers_to_show_filtered)), labels=numbers_to_show_filtered)            
            plt.title(f"Probability distribution over numbers \n(target = {target_number}, hardprompt = {hardprompt}")
            save_path = os.path.join(dataset_dir, f"visual_probs{i}.jpg")
            plt.savefig(save_path, dpi=300)
            plt.close()


    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









