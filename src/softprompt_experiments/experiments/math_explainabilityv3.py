import torch
import torch.nn.functional as F
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm
import json
from softprompt_experiments.models.softprompt import SoftPrompt
from softprompt_experiments.models.squishyprompt import SquishyPrompt
from softprompt_experiments.utils import (
    get_train_test_from_tokenized, 
    eval_softprompt_regression,
    train_softprompt_from_tokenized,
    eval_softprompt,
    log_json
)

import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pandas as pd

"""
run me
sbatch -t 90 --output explainabilityv2-%j.out job.sh --experiment math_dataset_generatorv2 math_softprompt_generator math_explainabilityv2 --num_datasets 100 --save_directory ./datasets/math_datasetv2_same --epochs 8 --init "________"
"""

def run(args_list):
    exp_name = os.path.basename(__file__)
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_directory", type=str, default="./datasets/math_dataset_custom")
    parser.add_argument("--visualizations_dir", type=str, default="./visualizations")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    args, _ = parser.parse_known_args(args_list)

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    SAVE_DIR = args.save_directory
    VIS_DIR = args.visualizations_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype
    ).to(device)
    model.eval()
    word_embeddings = model.get_input_embeddings()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # grabs the softprompts
    def get_softprompts_set(group):
        dataset_dirs = []
        for entry in os.scandir(f"{SAVE_DIR}{group}"):
            if entry.is_dir():  # Check if the entry is a directory
                if "dataset_" in entry.name:
                    dataset_dirs.append(entry.path)

        num_datasets = len(dataset_dirs)
        if num_datasets > 0:
            print(f"\nFound ({num_datasets}) datasets in directory")
        else:
            raise ValueError("path to directory has no datasets")

        soft_embeds = []
        out = {}
        for dataset_dir in tqdm(dataset_dirs):
            hardprompt = torch.load(
                os.path.join(dataset_dir,'dataset.pt'),
                weights_only=False
            )['hardprompt']

            softprompt = SoftPrompt(
                model=model,
                word_embeddings=word_embeddings,
                tokenizer=tokenizer,
                path_to_model=os.path.join(dataset_dir, "softprompt.pt")
            )
            with torch.no_grad():
                prompt_embed = softprompt.forward()

            soft_embeds.append(prompt_embed)
            out[hardprompt] = {'embed':prompt_embed}

            # load dataset
            _, test_dataset, _, _ = get_train_test_from_tokenized(dataset_dir,16)
            out[hardprompt]['testset'] = test_dataset

            # # performance
            # perf = eval_softprompt_regression(softprompt, test_dataset, VIS_DIR)
            # print(f"softprompt performance1: {perf}")

            # # performance
            # softprompt = SoftPrompt(model=model,tokenizer=tokenizer,word_embeddings=word_embeddings)
            # softprompt.set_prompt_embeddings(prompt_embed.squeeze(0))
            # perf = eval_softprompt_regression(softprompt, test_dataset, VIS_DIR)
            # print(f"softprompt performance2: {perf}")

            # # performance
            # softprompt = SoftPrompt(model=model,tokenizer=tokenizer,word_embeddings=word_embeddings)
            # softprompt.set_prompt_embeddings(out[hardprompt]['embed'].squeeze(0))
            # perf = eval_softprompt_regression(softprompt, test_dataset, VIS_DIR)
            # print(f"softprompt performance3: {perf}")


        centroid = torch.mean(torch.cat(soft_embeds, dim=0), dim=0)

        for hardprompt in out:
            out[hardprompt]['normed_embed'] = out[hardprompt]['embed'] - centroid

        return out, centroid
        
    outA = get_softprompts_set("A")
    outB = get_softprompts_set("B")
    # outC = get_softprompts_set("C")
    # outD = get_softprompts_set("D")
    # outE = get_softprompts_set("E")


    # def plot_softprompt_pca(
    #     group_outputs,         # dict: group_name -> (out_dict, centroid)
    #     VISUALIZATION_DIR,
    #     embed_key="embed",     # "embed" or "normed_embed"
    #     filename="softprompt_pca.png"
    # ):
    #     os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    #     X = []
    #     task_labels = []
    #     group_labels = []

    #     # Collect pooled embeddings
    #     for group_name, (out, _) in group_outputs.items():
    #         for task, d in out.items():
    #             embed = d[embed_key]  # [1, seq_len, dim] or [seq_len, dim]

    #             if embed.dim() == 3:
    #                 embed = embed.squeeze(0)

    #             pooled = (
    #                 embed
    #                 .mean(dim=0)      # [dim]
    #                 .detach()
    #                 .float()
    #                 .cpu()
    #                 .numpy()
    #             )

    #             X.append(pooled)
    #             task_labels.append(task)
    #             group_labels.append(group_name)

    #     X = np.stack(X)

    #     # PCA
    #     pca = PCA(n_components=2)
    #     X_pca = pca.fit_transform(X)

    #     # Color by task
    #     unique_tasks = sorted(set(task_labels))
    #     cmap = plt.get_cmap("tab10")
    #     task_to_color = {
    #         task: cmap(i % 10) for i, task in enumerate(unique_tasks)
    #     }

    #     plt.figure(figsize=(8, 6))

    #     for i in range(len(X_pca)):
    #         plt.scatter(
    #             X_pca[i, 0],
    #             X_pca[i, 1],
    #             color=task_to_color[task_labels[i]],
    #             s=70,
    #             alpha=0.85
    #         )
    #         # Label by group
    #         plt.text(
    #             X_pca[i, 0],
    #             X_pca[i, 1],
    #             group_labels[i],
    #             fontsize=9,
    #             ha="center",
    #             va="center"
    #         )

    #     # Legend for tasks
    #     legend_handles = [
    #         plt.Line2D(
    #             [0], [0],
    #             marker="o",
    #             linestyle="",
    #             label=str(task),
    #             markerfacecolor=task_to_color[task],
    #             markeredgecolor="black",
    #             markersize=8
    #         )
    #         for task in unique_tasks
    #     ]

    #     plt.legend(
    #         handles=legend_handles,
    #         title="Task (Hardprompt)",
    #         frameon=False,
    #         loc="best"
    #     )

    #     plt.xlabel("PC 1")
    #     plt.ylabel("PC 2")
    #     plt.title(f"PCA of Softprompts ({embed_key})\nColor = Task, Label = Group")
    #     plt.tight_layout()

    #     save_path = os.path.join(VISUALIZATION_DIR, filename)
    #     plt.savefig(save_path, dpi=300)
    #     plt.close()

    #     print(f"Saved PCA plot to {save_path}")

    # plot_softprompt_pca(
    #     group_outputs={
    #         "A": outA,
    #         "B": outB,
    #         "C": outC,
    #         "D": outD,
    #         "E": outE,

    #     },
    #     VISUALIZATION_DIR=VIS_DIR,
    #     embed_key="embed",          # or "normed_embed"
    #     filename="softprompt_pca.png"
    # )
    # plot_softprompt_pca(
    #     group_outputs={
    #         "A": outA,
    #         "B": outB,
    #         "C": outC,
    #         "D": outD,
    #         "E": outE,
    #     },
    #     VISUALIZATION_DIR=VIS_DIR,
    #     embed_key="normed_embed",          # or "normed_embed"
    #     filename="normed_softprompt_pca.png"
    # )

    # TODO:
    # We want to see if given the centroid of A
    # and a normed softprompt from B
    # if centroid of A + softprompt from B produces a valid softprompt

    # Also check if the centroid by themselves do anything

    embsA, centroidA = outA
    embsB, centroidB = outB
    for (hardpromptA,hardpromptB) in zip(embsA.keys(), embsB.keys()):
        print(hardpromptA, hardpromptB)

        soft_embedsA = embsA[hardpromptA]['embed']
        soft_embedsB = embsB[hardpromptB]['embed']

        flat_embA = torch.mean(soft_embedsA, dim=1)
        flat_embB = torch.mean(soft_embedsB, dim=1)

        print(F.cosine_similarity(flat_embA, flat_embB, dim=1))

        normed_embedsA = embsA[hardpromptA]['normed_embed']
        normed_embedsB = embsB[hardpromptB]['normed_embed']

        flat_normedA = torch.mean(normed_embedsA, dim=1)
        flat_normedB = torch.mean(normed_embedsB, dim=1)

        print(F.cosine_similarity(flat_normedA, flat_normedB, dim=1))

        converted = normed_embedsA + centroidB
        control = normed_embedsB + centroidB

        test_dataset = embsB[hardpromptB]['testset']

        # load target softprompt
        targ_softprompt = SoftPrompt(model=model,tokenizer=tokenizer,word_embeddings=word_embeddings)
        targ_softprompt.set_prompt_embeddings(control.squeeze(0))
        target_performance = eval_softprompt_regression(targ_softprompt, test_dataset, VIS_DIR)
        print(f"target performance: {target_performance['pearson_r']:.3f}")

        # load converted softprompt
        converted_softprompt = SoftPrompt(model=model,tokenizer=tokenizer,word_embeddings=word_embeddings)
        converted_softprompt.set_prompt_embeddings(converted.squeeze(0))
        converted_performance = eval_softprompt_regression(converted_softprompt, test_dataset, VIS_DIR)
        print(f"converted performance: {converted_performance['pearson_r']:.3f}")

        # load centroid softprompt
        cont_softprompt = SoftPrompt(model=model,tokenizer=tokenizer,word_embeddings=word_embeddings)
        cont_softprompt.set_prompt_embeddings(centroidB.squeeze(0))
        cont_performance = eval_softprompt_regression(cont_softprompt, test_dataset, VIS_DIR)
        print(f"centroid performance: {cont_performance['pearson_r']:.3f}")

        # load baseline softprompt
        baseline_softprompt = SoftPrompt(model=model,tokenizer=tokenizer,word_embeddings=word_embeddings)
        baseline_performance = eval_softprompt_regression(baseline_softprompt, test_dataset, VIS_DIR)
        print(f"baseline performance: {baseline_performance['pearson_r']:.3f}")



    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









