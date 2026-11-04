"""Task 4: reference and tiled (online-softmax) scaled dot-product attention.

Shapes: q [B, H, Lq, D], k/v [B, H, Lk, D]. Output [B, H, Lq, D].
Causal masking uses the standard "bottom-right aligned" convention so that it also
works when Lq < Lk (e.g. decoding with a KV cache): query i (0-based) may attend to
key j iff  j <= i + (Lk - Lq).
"""
import math

import torch


def causal_mask(Lq, Lk, device):
    i = torch.arange(Lq, device=device)[:, None]
    j = torch.arange(Lk, device=device)[None, :]
    return j <= i + (Lk - Lq)  # True = allowed


def reference_attention(q, k, v, causal=True):
    """Naive attention: materialises the full [Lq, Lk] score matrix.
    softmax(QK^T / sqrt(D)) V with a numerically stable softmax (row-max subtracted)."""
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    if causal:
        scores = scores.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)  # stability
    p = torch.exp(scores)
    p = p / p.sum(dim=-1, keepdim=True)
    return torch.matmul(p, v)


def tiled_attention(q, k, v, causal=True, block_q=64, block_k=64):
    """FlashAttention-style forward pass in plain PyTorch.

    For every query block we sweep over key/value blocks keeping, per query row,
      m : running max of the scores seen so far
      l : running sum of exp(score - m)           (softmax normaliser)
      o : running un-normalised weighted sum of V  (sum exp(score - m) * v)
    When a new block has a larger max m_new, the old statistics are rescaled by
    exp(m_old - m_new).  The full [Lq, Lk] matrix is never built (only a
    [block_q, block_k] tile), and the result equals exact softmax attention.
    Statistics are accumulated in float32 for stability.
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    offset = Lk - Lq
    for qs in range(0, Lq, block_q):
        qe = min(qs + block_q, Lq)
        qb = q[..., qs:qe, :].float()
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, qe - qs, D), device=q.device, dtype=torch.float32)
        # with causal masking, keys beyond the last query's position are never needed
        k_end = min(Lk, qe + offset) if causal else Lk
        for ks in range(0, k_end, block_k):
            ke = min(ks + block_k, k_end)
            s = torch.matmul(qb, k[..., ks:ke, :].float().transpose(-2, -1)) * scale
            if causal:
                qi = torch.arange(qs, qe, device=q.device)[:, None]
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(~(kj <= qi + offset), float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # rows that are fully masked so far keep m=-inf; avoid (-inf)-(-inf)=nan
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m - m_safe)            # rescale factor for old stats
            p = torch.exp(s - m_safe)                # tile probabilities (un-normalised)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            o = o * alpha + torch.matmul(p, v[..., ks:ke, :].float())
            m = m_new
        out[..., qs:qe, :] = (o / l).to(q.dtype)
    return out


def sdpa_attention(q, k, v, causal=True):
    """PyTorch's fused/optimised backend (reference for comparison only).
    Note: F.scaled_dot_product_attention's is_causal is top-left aligned, so for
    Lq != Lk we pass an explicit bottom-right-aligned mask."""
    if q.shape[-2] == k.shape[-2]:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    mask = causal_mask(q.shape[-2], k.shape[-2], q.device) if causal else None
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
