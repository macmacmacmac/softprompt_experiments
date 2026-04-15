from softprompt_experiments.models.priors import logit_priors

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os


class GMM_prior(logit_priors.LogitPrior):
    """
    Prior assuming logits follow a Gaussian mixture model after PCA.
    """
    def __init__(self, gmm_and_pca_dir="./datasets/inversion_dataset/instructions_logits_dataset/models/gmm"):
        super().__init__()

        # -----------------------------
        # Load sklearn objects
        # -----------------------------
        gmm = joblib.load(os.path.join(gmm_and_pca_dir, "gmm.joblib"))
        pca = joblib.load(os.path.join(gmm_and_pca_dir, "pca.joblib"))

        # -----------------------------
        # Store PCA params
        # -----------------------------
        self.register_buffer("pca_mean", torch.tensor(pca.mean_, dtype=torch.float32))
        self.register_buffer("pca_components", torch.tensor(pca.components_, dtype=torch.float32))

        # -----------------------------
        # Store GMM params
        # -----------------------------
        self.K = gmm.n_components
        self.D = gmm.means_.shape[1]

        self.register_buffer("weights", torch.tensor(gmm.weights_, dtype=torch.float32))  # (K,)
        self.register_buffer("means", torch.tensor(gmm.means_, dtype=torch.float32))      # (K, D)
        self.register_buffer("covs", torch.tensor(gmm.covariances_, dtype=torch.float32)) # (K, D, D)

        # Precompute inverses + log dets
        cov_inv = torch.linalg.inv(self.covs)  # (K, D, D)
        log_det = torch.logdet(self.covs)      # (K,)

        self.register_buffer("cov_inv", cov_inv)
        self.register_buffer("log_det", log_det)

        # constant term
        self.log_2pi = torch.log(torch.tensor(2 * torch.pi))

    # -----------------------------
    # PCA transform (torch version)
    # -----------------------------
    def pca_transform(self, x):
        """
        x: (B, original_dim)
        """
        x_centered = x - self.pca_mean
        return x_centered @ self.pca_components.T  # (B, D)

    # -----------------------------
    # Log prob
    # -----------------------------
    def get_last_token_logits(self, logits, attention_mask):
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

    def log_prob(self, outputs, attention_mask, **kwargs):
        """
        logits: transformers model outputs
        attention_mask: (B, seq_len)
        returns: (B,) log probabilities
        """

        # 1. PCA reduce
        logits = outputs.logits #logits: [B, T, V]
        last_logits = self.get_last_token_logits(logits, attention_mask)
        x = self.pca_transform(last_logits)  # (B, D)

        B = x.shape[0]

        # Expand for broadcasting
        x = x.unsqueeze(1)              # (B, 1, D)
        means = self.means.unsqueeze(0) # (1, K, D)

        diff = x - means                # (B, K, D)

        # Mahalanobis term: (x - μ)^T Σ^{-1} (x - μ)
        # einsum: (B,K,D) x (K,D,D) x (B,K,D) → (B,K)
        mahal = torch.einsum("bkd,kde,bke->bk", diff, self.cov_inv, diff)

        # Log Gaussian per component
        log_probs = -0.5 * (
            mahal + self.log_det.unsqueeze(0) + self.D * self.log_2pi
        )  # (B, K)

        # Add log weights
        log_weights = torch.log(self.weights).unsqueeze(0)  # (1, K)
        log_probs = log_probs + log_weights  # (B, K)

        # Log-sum-exp over components
        return torch.logsumexp(log_probs, dim=1)  # (B,)

