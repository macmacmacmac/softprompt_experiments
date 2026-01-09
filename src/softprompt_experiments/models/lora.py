import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW, Adam
import random
import os
import copy
import math
from peft import get_peft_model, PeftModel

class LoRa(nn.Module):
    """
    An implementation of LoRa, a wrapper around PEFT
    - model: a huggingface decoder model
    - path_to_model: if passed, loads a saved softprompt model instead of initializing one
    - lora_config: a peft LoraConfig object

    """
    def __init__(self, model=None, lora_config=None, path_to_model=None):
        super().__init__()

        object.__setattr__(self, "_model", model)

        if path_to_model is None:
            self.lora_model = get_peft_model(self.base_model, lora_config)
        else:
            self.lora_model = PeftModel.from_pretrained(
                self.base_model, path_to_model
            )

    def generate_from_embeds(self, embeds=None, max_new_tokens=20, do_sample=True, suffix_str=None):
        """
        Generate text given softprompt embeddings.
        Args:
            embeds: [1, seq_len, hidden_dim] softprompt embeddings
            max_new_tokens: number of tokens to generate
            do_sample: whether to sample or use greedy decoding
            suffix_str: some string to be appended after the embeds
        Returns:
            generated string
        """
        with torch.no_grad():
            if embeds is not None:
                if suffix_str:
                    ids = self._tokenizer(suffix_str, return_tensors="pt").input_ids.to(self._model.device)
                    suffix_embs = self._word_embeddings(ids).to(dtype=self._model.dtype)
                    full_embs = torch.cat([embeds, suffix_embs], dim=1)
                attention_mask = torch.ones(embeds.size()[:-1], device=full_embs.device, dtype=torch.long)
                output_ids = self._model.generate(
                    inputs_embeds=full_embs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            elif suffix_str:
                ids = self._tokenizer(suffix_str, return_tensors="pt").input_ids.to(self._model.device)
                suffix_embs = self._word_embeddings(ids).to(dtype=self._model.dtype)
                full_embs = suffix_embs
                attention_mask = torch.ones(full_embs.size()[:-1], device=full_embs.device, dtype=torch.long)
                output_ids = self._model.generate(
                    inputs_embeds=full_embs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            else:
                raise ValueError("At least one of embs or suffix_str must not be None")
            
        output = self._tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return output


        