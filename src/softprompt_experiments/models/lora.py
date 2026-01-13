import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW, Adam
import random
import os
import copy
import math

from contextlib import contextmanager

@contextmanager
def disable_lora(model):
    old = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            old.append(m.scaling)
            m.scaling = 0.0
    try:
        yield
    finally:
        for m, s in zip(
            (m for m in model.modules() if isinstance(m, LoRALinear)),
            old
        ):
            m.scaling = s

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: float = 16.0):
        super().__init__()
        self.linear = orig_linear
        for p in self.linear.parameters():
            p.requires_grad = False

        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.bias = self.linear.bias is not None

        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / max(1, self.r)

        if self.r > 0:
            self.A = nn.Parameter(torch.zeros((self.r, self.in_features), dtype=self.linear.weight.dtype))
            self.B = nn.Parameter(torch.zeros((self.out_features, self.r), dtype=self.linear.weight.dtype))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        # IMPORTANT: mark this as LoRA
        self.is_lora = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., in_features)
        # base output
        base = F.linear(x, self.linear.weight, self.linear.bias)

        if self.r <= 0:
            return base

        # compute LoRA delta: (x @ A.T) -> (..., r) then @ B.T -> (..., out_features)
        # A: (r, in_features) => A.T: (in_features, r)
        # x @ A.T -> (..., r)
        # then @ B.T (r, out_features) => (..., out_features)
        # multiply by scaling
        # ensure dtype matches linear weight dtype
        xa = torch.matmul(x, self.A.t())
        delta = torch.matmul(xa, self.B.t()) * self.scaling
        return base + delta

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 4, alpha: float = 16.0):
        return cls(linear, r=r, alpha=alpha)
    # LoRA hyperparams (you can tweak)


class LoRa(nn.Module):
    """
    An implementation of LoRa, a wrapper around PEFT
    - model: a huggingface decoder model
    - path_to_model: if passed, loads a saved softprompt model instead of initializing one
    - r: lora rank
    - alpha: lora alpha

    """
    def __init__(self, model=None, r=16, alpha=16, path_to_model=None):
        # Helper to replace modules in-place by name
        def replace_linear_with_lora(model: nn.Module, r: int, alpha: float):
            """
            Replace ONLY q_proj and v_proj linear layers with LoRALinear wrappers.
            Everything else stays frozen.
            """
            name_to_module = dict(model.named_modules())

            for name, module in list(model.named_modules()):
                # only replace torch.nn.Linear
                if not isinstance(module, nn.Linear):
                    continue

                lowered = name.lower()

                # Only patch q_proj and v_proj
                if not ("q_proj" in lowered or "v_proj" in lowered):
                    continue

                # Avoid embedding layers or lm_head (safety)
                if "embed" in lowered or "lm_head" in lowered:
                    continue

                # Find parent module
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = name_to_module.get(parent_name, None)
                else:
                    parent = model
                    child_name = name

                if parent is None:
                    continue

                orig_linear = getattr(parent, child_name)

                # Avoid double-wrapping in case of re-entry
                if isinstance(orig_linear, LoRALinear):
                    continue

                # Wrap with LoRA
                lora_module = LoRALinear.from_linear(
                    orig_linear,
                    r=r,
                    alpha=alpha
                )

                # Ensure device/dtype matches original
                w = orig_linear.weight
                lora_module.to(device=w.device, dtype=w.dtype)

                # Replace in parent
                setattr(parent, child_name, lora_module)

        super().__init__()

        object.__setattr__(self, "_model", model)

        replace_linear_with_lora(model, r=r, alpha=alpha)
        if path_to_model is not None:
            self.load_lora(path_to_model)
    
    def load_lora(self, path: str):
        model = self._model
        device = self._model.device

        checkpoint = torch.load(path, map_location="cpu")
        lora_adapters = checkpoint["lora_adapters"]

        name_to_module = dict(model.named_modules())

        for name, payload in lora_adapters.items():
            if name not in name_to_module:
                raise KeyError(f"LoRA module '{name}' not found in model")

            module = name_to_module[name]

            if not isinstance(module, LoRALinear):
                raise TypeError(f"Module '{name}' is not LoRALinear")

            if module.r != payload["r"] or module.alpha != payload["alpha"]:
                raise ValueError(
                    f"LoRA config mismatch at '{name}': "
                    f"(r, alpha)=({module.r}, {module.alpha}) "
                    f"!= ({payload['r']}, {payload['alpha']})"
                )

            module.A.data.copy_(payload["A"])
            module.B.data.copy_(payload["B"])

            if device is not None:
                module.to(device)
    
    def save_lora(self, save_path: str):
        """
        Returns a dict that can be saved with torch.save(...)
        """
        lora_state = {}
        model = self._model
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear) and module.r > 0:
                lora_state[name] = {
                    "A": module.A.detach().cpu(),
                    "B": module.B.detach().cpu(),
                    "r": module.r,
                    "alpha": module.alpha,
                }

        torch.save(lora_state, os.path.join(save_path,"lora.pt"))
        

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

    def gen_without_lora(self, embeds=None,max_new_tokens=20, do_sample=True, suffix_str=None):
        with disable_lora(self._model):
            return self.generate_from_embeds(
                embeds=embeds, 
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample, 
                suffix_str=suffix_str
            )


        