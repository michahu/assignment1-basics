import math

import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.utils import sdpa, softmax


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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = einsum(positions, freqs, "seq, half -> seq half")
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if not d_ff:
            d_ff = 64 * round(d_model * 8 / 3 / 64)
        self.d_ff = d_ff
        self.w1 = Linear(d_model, self.d_ff)
        self.w3 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        d_k = int(d_model / num_heads)

        self.w_k = Linear(d_model, d_model)
        self.w_q = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)

        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len) if theta is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)

        # compute matmuls
        k = rearrange(self.w_k(x), "... t (h d) -> ... h t d", h=self.num_heads)
        q = rearrange(self.w_q(x), "... t (h d) -> ... h t d", h=self.num_heads)
        v = rearrange(self.w_v(x), "... t (h d) -> ... h t d", h=self.num_heads)

        if self.rope is not None:
            k = self.rope(k)
            q = self.rope(q)

        # construct attention mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        # do sdpa
        h = rearrange(sdpa(k, q, v, mask), "... h t d -> ... t (h d)", h=self.num_heads)
        return self.w_o(h)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len=2048):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x += self.attn(h)
        h = self.ln2(x)
        return x + self.ffn(h)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len=context_length) for i in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        x = self.layers(x)
        return self.lm_head(self.ln_final(x))


if __name__ == "__main__":
    rmsnorm = RMSNorm(5)
    x = torch.randn(5)
    print(rmsnorm(x))
