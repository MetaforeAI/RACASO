"""NanoGPT — vendored minimal byte-level GPT for R3.

Byte-level (vocab 256) keeps the bench dependency-free — no
``transformers``, no ``tokenizers``, no merges JSON. Byte-level
modeling is a legitimate industry practice (ByT5, MEGABYTE,
Karpathy's nanoGPT examples) and gives us a credible LM signal
without external tokenizer state.

Architecture (default):
    vocab=256, hidden=384, heads=6, layers=6, seq_len=256
    ~30M parameters total.

This is structurally the same model as `bench/models/charlm.py` but
deeper/wider, and at byte level rather than ASCII-char level — the file
exists separately so the R2 char-LM and R3 byte-LM benches can ablate
size while staying readable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(att)


class _Block(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = _CausalSelfAttention(hidden, heads)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    """Byte-level GPT, ~30M params at default config."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden: int = 384,
        heads: int = 6,
        layers: int = 6,
        seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([_Block(hidden, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)
