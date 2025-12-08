import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW, Adam
import random
import os
import copy
from softprompt_experiments.models.softprompt import SoftPrompt
class SquishyPrompt(SoftPrompt):
    """
    An implementation of softprompt for prompt-tuning, now with parsability loss term
    - model: a huggingface decoder model
    - word_embeddings: it's word_embedding matrix from model.get_input_embeddings
    - tokenizer: the tokenizer to be used
    - path_to_model: if passed, loads a saved softprompt model instead of initializing one
    - num_tokens: number of virtual tokens in the softprompt
    """
    def __init__(self, model=None, tokenizer=None, word_embeddings=None, path_to_model=None, num_tokens=8, lambd=1.):
        super().__init__(
            model=model, 
            tokenizer=tokenizer, 
            word_embeddings=word_embeddings, 
            path_to_model=path_to_model, 
            num_tokens=num_tokens
        )
        self.lambd = lambd
    
    def loss_fn(self, input_embeds, labels):
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=None,    # attention is fully allowed
            labels=labels   # HF automatically computes CE
        )
        CE_loss =  outputs.loss
        parsability_loss = self.get_parsability()
        return CE_loss + self.lambd*parsability_loss
    
    def save_softprompt(self, path_to_save):
        state_dict = {
            'prompt_embeddings':self.forward(),
            'initial_tokens':self.initial_tokens,
            'initial_embeddings':self.initial_embeddings,
            'num_tokens':self.num_tokens
        }
        torch.save(state_dict, os.path.join(path_to_save, "squishy.pt"))
    
    def load_softprompt(self, path_to_load):
        state_dict = torch.load(os.path.join(path_to_load, "squishy.pt"))
        self.initial_tokens = state_dict['initial_tokens']
        self.initial_embeddings = state_dict['initial_embeddings']
        self.num_tokens = state_dict['num_tokens']
        self.prompt_embeddings = state_dict['prompt_embeddings'].squeeze(0)
