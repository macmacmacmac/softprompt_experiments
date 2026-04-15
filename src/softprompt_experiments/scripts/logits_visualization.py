import torch
import argparse
import os

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset

from tqdm import tqdm

import glob
import torch

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from softprompt_experiments.models.logit_priors import GMM_prior
import joblib
import logging
import numpy as np

class LogitsDataset(torch.utils.data.Dataset):
    def __init__(self, save_dir):
        self.files = sorted(glob.glob(f"{save_dir}/shard_*.pt"))
        self.shard_sizes = []
        for f in tqdm(self.files):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualizations_dir", type=str, default="./visualizations") 
    parser.add_argument("--save_directory", type=str, default="./datasets/inversion_dataset/instructions_logits_dataset")
    args, _ = parser.parse_known_args(args_list)
    
    SAVE_DIR = args.save_directory
    VISUALIZATIONS_DIR = args.visualizations_dir

    logging.getLogger().setLevel(logging.WARNING)

    logger = logging.getLogger(f"{exp_name}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            # logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            logging.Formatter("%(message)s")
        )

        # File handler
        file_handler = logging.FileHandler(os.path.join(SAVE_DIR,f"{exp_name}.log"), mode="w")
        file_handler.setFormatter(
            # logging.Formatter("%(levelname)s - %(message)s")
            logging.Formatter("%(message)s")
        )
        file_handler.flush = file_handler.stream.flush

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    logger.propagate = False

    # logging.getLogger("transformers").setLevel(logging.INFO)
    # logging.getLogger("torch").setLevel(logging.INFO)

    logger.info(
        f"{'='*100}\n\t\t\t\tRunning script: {exp_name}\n{'='*100}"
    )

    logger.info("Args: %s", vars(args))    


    # dataset = LogitsDataset(SAVE_DIR) #directory to save visualization
    # all_samples = []
    # for f in tqdm(dataset.files):
    #     tensor = torch.load(f, map_location='cpu')
    #     idx = torch.randperm(tensor.size(0))[:1000]  # sample 500 rows
    #     all_samples.append(tensor[idx])
    # sample_tensor = torch.cat(all_samples, dim=0)

    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    dataset = load_dataset("wentingzhao/one-million-instructions", split="train")
    total = len(dataset)
    MAX_SAMPLES = 20000
    indices = np.random.choice(total, MAX_SAMPLES, replace=False)
    subset = dataset.select(indices)  # memory-efficient subset
    def get_last_token_logits(logits, attention_mask):
        lengths = attention_mask.sum(dim=1) - 1
        return logits[torch.arange(logits.size(0)), lengths]

    from torch.utils.data import DataLoader

    def collate_fn(batch):
        texts = [x["user"] for x in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return tokenized, texts

    BATCHSIZE = 64
    loader = DataLoader(subset, batch_size=BATCHSIZE, collate_fn=collate_fn)

    # -----------------------------
    # Collect logits
    # -----------------------------
    all_logits = []
    # all_texts = []

    with torch.no_grad():
        for batch, texts in tqdm(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits  # [B, T, V]

            last_logits = get_last_token_logits(
                logits,
                batch["attention_mask"]
            )

            all_logits.append(last_logits.cpu())
            # all_texts.append(texts)


    # -----------------------------
    # Concatenate + trim
    # -----------------------------
    all_logits = torch.cat(all_logits, dim=0) 
    sample_tensor = all_logits

    logger.info(f"Collected logits shape: {all_logits.shape}")
     
    def visualize_logits(sample_tensor, save_dir, title="logits_analysis"):
        os.makedirs(save_dir, exist_ok=True)

        # Convert dataset to a single tensor (N, D)
        probs = torch.log_softmax(sample_tensor, dim=-1)
        X = probs.data.float().numpy()  # (num_samples, num_logits)
        N, D = X.shape
        logger.info(f"Dataset shape: {X.shape}")

        # -----------------------------
        # 1. PCA (2D) Scatter Plot
        # -----------------------------
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(6,6))
        plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.3, s=5)
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title("PCA 2D Scatter of Logits")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{title}_pca_scatter.jpg"), dpi=300)
        plt.close()

    # visualize_logits(sample_tensor, VISUALIZATIONS_DIR)

    # sample_tensor = sample_tensor - sample_tensor[:,0].unsqueeze(1)
    
    # visualize_logits(sample_tensor, VISUALIZATIONS_DIR, "standardized_logits_analysis")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="bfloat16")
    # word_embeddings = model.get_input_embeddings().weight

    # del model
    # torch.cuda.empty_cache()  # if using GPU
    def gmm_elbow_plot(
        sample_tensor, 
        save_dir, 
        title="gmm_elbow",
        max_components=5,
        pca_dim=20
    ):
        os.makedirs(save_dir, exist_ok=True)

        # -----------------------------
        # 1. Load / prepare data
        # -----------------------------
        probs = torch.log_softmax(sample_tensor, dim=-1)
        X_sample = probs.data.float().numpy()
        logger.info(f"Data shape: {X_sample.shape}")

        # -----------------------------
        # 2. PCA
        # -----------------------------
        pca = PCA(n_components=pca_dim, random_state=42)
        X_reduced = pca.fit_transform(X_sample)
        # probs = torch.softmax(sample_tensor, dim=-1)
        # X_reduced = torch.matmul(probs, word_embeddings).data.float().cpu().numpy()   # (batch, seq, hidden_dim)
        logger.info(f"PCA reduced shape: {X_reduced.shape}")

        # -----------------------------
        # 3. Sweep over n_components
        # -----------------------------
        n_components_list = list(range(1, max_components + 1))
        log_likelihoods = []

        for k in n_components_list:
            logger.info(f"Fitting GMM with {k} components...")
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=42,
                reg_covar=1e-3,
            )
            gmm.fit(X_reduced)

            ll = gmm.score(X_reduced)  # avg log-likelihood per sample
            log_likelihoods.append(ll)

        # -----------------------------
        # 4. Plot elbow curve
        # -----------------------------
        plt.figure(figsize=(6,4))
        plt.plot(n_components_list, log_likelihoods, marker='o')
        plt.xlabel("Number of Components")
        plt.ylabel("Avg Log-Likelihood")
        plt.title("GMM Elbow Plot (Log-Likelihood)")
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{title}_elbow.jpg")
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"Elbow plot saved to {save_path}")

        return n_components_list, log_likelihoods
    
    # gmm_elbow_plot(sample_tensor, VISUALIZATIONS_DIR)

    def save_gmm_pca(gmm, pca, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(gmm, os.path.join(save_dir, "gmm.joblib"))
        joblib.dump(pca, os.path.join(save_dir, "pca.joblib"))

    def gmm_cluster_logits(sample_tensor, save_dir, title="logits_gmm", 
                        n_components=3, pca_dim=20, n_samples_per_shard=500):
        """
        Subsamples logits from each shard, reduces dimensionality with PCA, 
        fits a Gaussian Mixture Model, and visualizes cluster assignments.
        """
        os.makedirs(save_dir, exist_ok=True)

        # X_sample = torch.cat(logits_dataset, dim=0).numpy()
        num_samples = len(sample_tensor)
        indices = torch.randperm(num_samples)
        sample_tensor = sample_tensor[indices]
        probs = torch.log_softmax(sample_tensor, dim=-1)
        probs_train = probs.data.float().numpy()[:int(num_samples*0.8)]
        probs_test = probs.data.float().numpy()[int(num_samples*0.8):]
        logger.info(f"Subsampled data shape: {probs_train.shape}")

        # -----------------------------
        # 2. PCA dimensionality reduction
        # -----------------------------
        pca = PCA(n_components=pca_dim, random_state=42)
        X_reduced_train = pca.fit_transform(probs_train)
        logger.info(f"PCA reduced shape: {X_reduced_train.shape}")

        # -----------------------------
        # 3. Fit Gaussian Mixture Model
        # -----------------------------
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', reg_covar=1e-3, random_state=42)
        gmm.fit(X_reduced_train)
        labels = gmm.predict(X_reduced_train)

        logger.info(f"GMM Weights: {gmm.weights_}")
        logger.info(f"GMM Means shape: {gmm.means_.shape}")
        logger.info(f"GMM Covariances shape: {gmm.covariances_.shape}")

        X_reduced_test = pca.transform(probs_test)

        logger.info(f"Average log-likelihood per sample (train): {gmm.score(X_reduced_train)}")
        logger.info(f"Average log-likelihood per sample (test): {gmm.score(X_reduced_test)}")

        save_gmm_pca(gmm, pca, os.path.join(SAVE_DIR, "models/gmm"))
        gmm_prior = GMM_prior(os.path.join(SAVE_DIR, "models/gmm"))

        sanity_log_likelihood = gmm_prior.log_prob(probs[:int(num_samples*0.8)]).mean().item()
        logger.info(f"SAnity check: average log-likelihood per sample (train): {sanity_log_likelihood}")
        sanity_log_likelihood = gmm_prior.log_prob(probs[int(num_samples*0.8):]).mean().item()
        logger.info(f"SAnity check: average log-likelihood per sample (test): {sanity_log_likelihood}")

        # -----------------------------
        # 4. Visualization (2D PCA scatter)
        # -----------------------------
        plt.figure(figsize=(6,6))
        plt.scatter(X_reduced_train[:,0], X_reduced_train[:,1], c=labels, cmap='tab10', s=5, alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"GMM Cluster Assignment: {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{title}_gmm_clusters.jpg"), dpi=300)
        plt.close()

        logger.info(f"GMM cluster visualization saved to {save_dir}")
        return gmm, X_reduced_train, labels

    gmm_cluster_logits(sample_tensor, VISUALIZATIONS_DIR)

    #:
    """
        TODO:
        1. When I took a subsample of the logits, and visualized it with PCA,
           I saw two distinct clusters.
           I want to check qualitatively what the two clusters I saw during PCA
           correspond to. I want to know if the original prompts that produced
           the logits belonging to each of the clusters are noticeably different. 
           Our logits dataset doesn't include the original hardprompts, so
           we should just load the original dataset, recompute logits, label which cluster
           they belong to in the PCA, and repeat until we get enough examples for both clusters.
           That probably means training a KMeans off the PCA first so we can get a cluster labeler.
        2. I trained a GMM on a sub sample of the logits dataset and got decent 
           performance (NLL score ~56 for 20 dim), but using this as a regularizer
           during training of soft prompts, my log prob never went below 86. 
           As a sanity check, I want to check the logit's logprobs under the GMM model for
           (A) samples from the original hard prompt dataset, (B) random natural language sentences, 
           (C) soft prompts I already have, and compamre how their averages.
    """
    # 1. Check qualitatively what the two clusters we saw
    # pca = PCA(n_components=2)
    # logits_2d = pca.fit_transform(torch.log_softmax(sample_tensor, dim=-1).data.float().numpy())
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # kmeans.fit(pca.transform(torch.log_softmax(sample_tensor, dim=-1).float().cpu().numpy()))

    # examples_per_cluster = {0: [], 1: []}
    # NUM_EXAMPLES = 10  # per cluster

    # with torch.no_grad():
    #     for batch, texts in loader:
    #         batch = {k: v.to(model.device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         logits = outputs.logits  # [B, T, V]

    #         last_logits = get_last_token_logits(logits, batch["attention_mask"])
    #         last_logits = torch.log_softmax(last_logits, dim=-1).float().cpu().numpy()
            
    #         # project to 2D using your PCA
    #         projected = pca.transform(last_logits)
            
    #         # assign cluster
    #         labels = kmeans.predict(projected)

    #         for text, label in zip(texts, labels):
    #             if len(examples_per_cluster[label]) < NUM_EXAMPLES:
    #                 examples_per_cluster[label].append(text)

    #         if all(len(lst) == NUM_EXAMPLES for lst in examples_per_cluster.values()):
    #             break

    # # logger.info examples
    # for cluster_id, texts in examples_per_cluster.items():
    #     logger.info(f"\nCluster {cluster_id} examples:")
    #     for t in texts:
    #         logger.info("<EXAMPLE START>", t, "<EXAMPLE END>")

    # 2. gmm prior sanity check
    # sample a few text examples from the loader, find their logits
    # then logger.info out their log prob under the gmm prior
    # then come up with a few examples of original prompts
    # pass them into the LLM, get the logits, then logger.info out their log prob under the GMM prior
    # to compare
    def compute_log_probs_for_texts(texts, model, tokenizer, gmm_prior, device):
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
            logits = outputs.logits

            last_logits = get_last_token_logits(
                logits,
                tokenized["attention_mask"]
            )

        # IMPORTANT: match training dtype / format
        logits_np = last_logits#.float().cpu().numpy()

        log_probs = gmm_prior.log_prob(torch.log_softmax(logits_np, dim=-1))  # shape: [B]
        return log_probs
    
    gmm_prior = GMM_prior(os.path.join(SAVE_DIR, "models/gmm")).to(model.device)

    logger.info("\n=== Dataset samples ===")

    dataset_texts = []

    for i, batch in enumerate(loader):
        texts = [x["user"] for x in batch] if isinstance(batch, list) else None

        # if using your collate_fn version:
        if isinstance(batch, tuple):
            _, texts = batch

        dataset_texts.extend(texts)

        if len(dataset_texts) >= 32:
            break

    dataset_texts = dataset_texts[:32]

    dataset_log_probs = compute_log_probs_for_texts(
        dataset_texts,
        model,
        tokenizer,
        gmm_prior,
        model.device
    )

    logger.info("Dataset log probs:")
    for t, lp in zip(dataset_texts[:10], dataset_log_probs[:10]):
        logger.info(f"{lp:.2f} | {t[:100]}")
        
    logger.info(f"Mean dataset log prob: {dataset_log_probs.mean()}")

    logger.info("\n=== Manual prompts ===")

    manual_texts = [
        "Hello",
        "What is 2+2?",
        "Write a short story about dragons.",
        "Explain quantum mechanics in simple terms.",
        "asdfasdfasdfasdf",  # nonsense
        "The capital of France is",
        "def foo(x): return x**2",
        "Translate 'hello' to Spanish.",
        "Summarize the following text:",
        "Why is the sky blue?"
    ]

    manual_log_probs = compute_log_probs_for_texts(
        manual_texts,
        model,
        tokenizer,
        gmm_prior,
        model.device
    )

    for t, lp in zip(manual_texts, manual_log_probs):
        logger.info(f"{lp:.2f} | {t}")

    logger.info(f"Mean manual log prob: {manual_log_probs.mean()}")

    logger.info(
        f"{'='*100}\n\t\t\t\tCompleted script: {exp_name}\n{'='*100}"
    )








