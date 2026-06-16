import math

import torch
from einops import einsum


def softmax(x: torch.Tensor, dim: int):
    v = x - x.max(dim=dim, keepdim=True).values
    exp_v = v.exp()
    return exp_v / exp_v.sum(dim=dim, keepdim=True)


def cross_entropy(logits, targets):
    v = logits - logits.max(dim=-1, keepdim=True).values
    log_softmax = v - torch.log(v.exp().sum(dim=-1, keepdim=True))
    return -log_softmax.gather(-1, targets.unsqueeze(-1)).mean()


def sdpa(keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = keys.size(-1)
    scores = einsum(queries, keys, "bsz ... q_len d_k, bsz ... k_len d_k -> bsz ... q_len k_len") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return einsum(softmax(scores, dim=-1), values, "bsz ... q_len k_len, bsz ... k_len d_v -> bsz ... q_len d_v")
