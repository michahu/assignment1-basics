import math

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(in_features, out_features, dtype=dtype, device=device))
        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        nn.init.trunc_normal_(self.weights, 0, var, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "in_features out_features, ... in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weights, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


# TODO: assume dtype is smaller than torch.float32
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.gain).to(in_dtype)


if __name__ == "__main__":
    rmsnorm = RMSNorm(5)
    x = torch.randn(5)
    print(rmsnorm(x))
