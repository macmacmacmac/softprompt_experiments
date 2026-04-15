from abc import ABC, abstractmethod
import torch.nn as nn

class LogitPrior(nn.Module, ABC):
    @abstractmethod
    def log_prob(self, outputs, attention_mask=None, **kwargs):
        """Return log probability of x"""
        pass

