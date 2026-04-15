from softprompt_experiments.models.priors import logit_priors
from softprompt_experiments.models.LM_inverter import LM_inverter, load_model

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

import transformers
from transformers.modeling_outputs import BaseModelOutput

def args_from_config(args_cls, config):
    args = args_cls()
    for key, value in vars(config).items():
        if key in dir(args):
            setattr(args, key, value)
    return args

class Inversion_Prior(logit_priors.LogitPrior):
    """
    Prior assuming logits follow a Gaussian mixture model after PCA.
    """
    def __init__(self, base_model, base_tokenizer):
        super().__init__()
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.word_embeddings = base_model.get_input_embeddings()
        self.inversion_model = load_model(base_model, base_tokenizer)
        self.inversion_model.to(base_model.device)
        self.gen_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
            'min_length': 1,
            'max_length': 128
        }

    def sample_z_prime_from_q(
            self, 
            x_z: BaseModelOutput, 
            attention_mask: torch.Tensor
        ):
        """
        Samples a z' ~ q(z'|x,z)
        """
        with torch.no_grad():
            # auto regressively generates z prime
            z_prime = self.inversion_model.generate_from_output(
                x_z,
                attention_mask,
                self.gen_kwargs
            )[0]
            # decoded = self.inversion_model.tokenizer.batch_decode(
            #     inversion,
            #     skip_special_tokens=True
            # )
            return z_prime
        
    def log_prob_q_of_z_prime(
            self,
            z_prime: torch.Tensor, 
            x_z: BaseModelOutput, 
            attention_mask: torch.Tensor
        ):
        """
        Computes log q(z'|x,z)
        """
        outputs = self.inversion_model(x_z, z_prime, attention_mask)
        
        # HF logits are shifted left like [b^,c^,d^,next_token^]
        logits = outputs.logits[:, :-1]
        labels = z_prime[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)

        # alternatively
        # log_q = outputs.loss

        log_q = token_log_probs.sum(dim=-1)
        return log_q
    
    def log_p_y_x_z_prime(
        self,
        z_prime: torch.Tensor,
        x_y: torch.Tensor,
        attention_mask: torch.tensor,
        labels: torch.Tensor,
    ):
        #TODO: build input_embeds somehow
        # probably decode z_prime (tokenized in T5's tokenzier), then tokenize in Llama2's tokenizer
        # 
        output = self.inversion_model.embedder(
            input_embeds=input_embeds,
            attention_mask=None,
            labels=labels
        )
        
        return output.loss
        

    """
        Log_prob v1:
        - Sample a hard prompt z' using the inverter
        - Compute CE between log p(y|z',x) and log(p|z,x)
        - Treat z^ as a detached random variable, no gradient
    """
    def log_prob(self, output, attention_mask, **kwargs):
        """
        output: transformers llm output
        attention_mask: [B, T]
        input_embeds: input sequence embeds including softprompt and Y [B, T, D]
        labels: tokenized labels [B, T]
        softprompt_len: number of softprompt tokens (len of Z)
        returns: (B,) log probabilities
        """

        device = output.logits.device

        input_embeds = kwargs["input_embeds"]   # [B, T, D]
        attention_mask = attention_mask         # [B, T]
        labels = kwargs["labels"]
        S = kwargs["softprompt_len"]

        # -------------------------------------------------------
        # 1. Sample Z' for each batch element (DETACHED)
        # -------------------------------------------------------

        with torch.no_grad():
            inversion = self.inversion_model.generate_from_output(
                output,
                attention_mask,
                self.gen_kwargs
            )

            decoded = self.inversion_model.tokenizer.batch_decode(
                inversion,
                skip_special_tokens=True
            )

            hardprompt = self.base_tokenizer(
                decoded,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            hardprompt = {k: v.to(device) for k, v in hardprompt.items()}

            # IMPORTANT: per-sample lengths
            input_ids = hardprompt["input_ids"]          # [B, L_i (padded)]
            attn = hardprompt["attention_mask"]

            Z_prime_embeds = self.word_embeddings(input_ids)  # [B, L_i, D]

        
        return loss