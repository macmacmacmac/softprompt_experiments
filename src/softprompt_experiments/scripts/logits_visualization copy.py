import torch
import argparse
import os

from torch.utils.data import DataLoader

from tqdm import tqdm

import glob
import torch

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from softprompt_experiments.models.logit_priors import GMM_prior
import joblib

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
    print(
        "="*100, "\n", 
        f"\t\t\t\tRunning script: {exp_name}", "\n",
        "="*100,"\n"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualizations_dir", type=str, default="./visualizations") 
    parser.add_argument("--save_directory", type=str, default="./datasets/inversion_dataset/instructions_logits_dataset")
    args, _ = parser.parse_known_args(args_list)
    
    SAVE_DIR = args.save_directory
    VISUALIZATIONS_DIR = args.visualizations_dir

    dataset = LogitsDataset(SAVE_DIR) #directory to save visualization
    all_samples = []
    for f in tqdm(dataset.files):
        tensor = torch.load(f, map_location='cpu')
        idx = torch.randperm(tensor.size(0))[:500]  # sample 500 rows
        all_samples.append(tensor[idx])
    sample_tensor = torch.cat(all_samples, dim=0)
     
    def visualize_logits(sample_tensor, save_dir, title="logits_analysis"):
        os.makedirs(save_dir, exist_ok=True)

        # Convert dataset to a single tensor (N, D)
        X = sample_tensor.data.float().numpy()  # (num_samples, num_logits)
        N, D = X.shape
        print(f"Dataset shape: {X.shape}")

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

    visualize_logits(sample_tensor, VISUALIZATIONS_DIR)

    # sample_tensor = sample_tensor - sample_tensor[:,0].unsqueeze(1)
    
    # visualize_logits(sample_tensor, VISUALIZATIONS_DIR, "standardized_logits_analysis")
    def gmm_elbow_plot(
        logits_dataset, 
        save_dir, 
        title="gmm_elbow",
        max_components=5,
        pca_dim=20
    ):
        os.makedirs(save_dir, exist_ok=True)

        # -----------------------------
        # 1. Load / prepare data
        # -----------------------------
        X_sample = logits_dataset.data.float().numpy()
        print(f"Data shape: {X_sample.shape}")

        # -----------------------------
        # 2. PCA
        # -----------------------------
        pca = PCA(n_components=pca_dim, random_state=42)
        X_reduced = pca.fit_transform(X_sample)
        print(f"PCA reduced shape: {X_reduced.shape}")

        # -----------------------------
        # 3. Sweep over n_components
        # -----------------------------
        n_components_list = list(range(1, max_components + 1))
        log_likelihoods = []

        for k in n_components_list:
            print(f"Fitting GMM with {k} components...")
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=42
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

        print(f"Elbow plot saved to {save_path}")

        return n_components_list, log_likelihoods
    
    gmm_elbow_plot(sample_tensor, VISUALIZATIONS_DIR)

    def save_gmm_pca(gmm, pca, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(gmm, os.path.join(save_dir, "gmm.joblib"))
        joblib.dump(pca, os.path.join(save_dir, "pca.joblib"))

    def gmm_cluster_logits(logits_dataset, save_dir, title="logits_gmm", 
                        n_components=3, pca_dim=20, n_samples_per_shard=500):
        """
        Subsamples logits from each shard, reduces dimensionality with PCA, 
        fits a Gaussian Mixture Model, and visualizes cluster assignments.
        """
        os.makedirs(save_dir, exist_ok=True)

        # X_sample = torch.cat(logits_dataset, dim=0).numpy()
        X_sample = logits_dataset.data.float().numpy()
        print(f"Subsampled data shape: {X_sample.shape}")

        # -----------------------------
        # 2. PCA dimensionality reduction
        # -----------------------------
        pca = PCA(n_components=pca_dim, random_state=42)
        X_reduced = pca.fit_transform(X_sample)
        print(f"PCA reduced shape: {X_reduced.shape}")

        # -----------------------------
        # 3. Fit Gaussian Mixture Model
        # -----------------------------
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(X_reduced)
        labels = gmm.predict(X_reduced)

        print("GMM Weights:", gmm.weights_)
        print("GMM Means shape:", gmm.means_.shape)
        print("GMM Covariances shape:", gmm.covariances_.shape)

        log_likelihood = gmm.score(X_reduced)  # average log-likelihood per sample
        print("Average log-likelihood per sample:", log_likelihood)

        save_gmm_pca(gmm, pca, os.path.join(SAVE_DIR, "models/gmm"))
        gmm_prior = GMM_prior(os.path.join(SAVE_DIR, "models/gmm"))

        sanity_log_likelihood = gmm_prior.log_prob(logits_dataset).mean().item()
        print("SAnity check: average log-likelihood per sample:", sanity_log_likelihood)

        # -----------------------------
        # 4. Visualization (2D PCA scatter)
        # -----------------------------
        plt.figure(figsize=(6,6))
        plt.scatter(X_reduced[:,0], X_reduced[:,1], c=labels, cmap='tab10', s=5, alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"GMM Cluster Assignment: {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{title}_gmm_clusters.jpg"), dpi=300)
        plt.close()

        print(f"GMM cluster visualization saved to {save_dir}")
        return gmm, X_reduced, labels

    gmm_cluster_logits(sample_tensor, VISUALIZATIONS_DIR)


    print(
        "\n","="*100, "\n", 
        f"\t\t\t\tCompleted script: {exp_name}", "\n",
        "="*100,
    )









