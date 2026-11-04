"""Task 5: greedy speculative decoding (draft model proposes, target model verifies).

Invariant at the start of each round (c = number of positions in both caches):
    seq  = [t_0 ... t_c]          # c+1 tokens; the LAST token `last` = t_c is NOT cached yet
    target cache and draft cache both hold K,V for t_0..t_{c-1}   (length c)

One round:
 1. DRAFT  : feed `last` to the draft, greedily propose d_1..d_k (k sequential draft
             forwards; the draft cache now holds last,d_1..d_{k-1}  -> length c+k).
 2. VERIFY : ONE target forward over [last, d_1..d_k] (k+1 tokens, positions c..c+k) gives
             target greedy predictions p_0..p_k, where p_i is the target's choice for the
             token that follows position c+i.  (p_0 should equal d_1, ..., p_{k-1} d_k).
 3. ACCEPT : m = length of the longest prefix with d_{i+1} == p_i.  Emit d_1..d_m and then
             p_m : if m<k, p_m is the target's CORRECTION for the first rejected draft token;
             if m==k, p_k is a free BONUS token.   => m+1 new tokens per target forward,
             and every emitted token is exactly what the target's greedy decoding emits.
 4. SYNC   : target cache holds c+k+1 entries but only last,d_1..d_m (c+m+1) are valid ->
             crop to c+m+1.  Draft cache: crop to c+m+1 too; if m==k the draft never saw d_k
             so we feed it d_k (one extra draft forward) to reach length c+k+1.
             New `last` = p_m (uncached), c <- c+m+1.  Invariant restored.
Output equals the target model's plain greedy output (up to floating-point differences
between a (k+1)-token forward and k+1 single-token forwards; we measure this).
"""
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from kv_decode import _keep_kw, _sync


@dataclass
class SpecResult:
    tokens: list
    seconds: float = 0.0
    prefill_s: float = 0.0
    peak_mem_MiB: float = 0.0
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0                      # accepted draft tokens
    target_forwards: int = 0               # incl. prefill
    draft_forwards: int = 0
    accept_hist: dict = field(default_factory=dict)   # m -> count

    @property
    def acceptance_rate(self):
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_round(self):
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0


def _fwd(model, ids, cache, start, keep_all=True):
    """Forward `ids` [1,n] placed at absolute positions start..start+n-1 using `cache`."""
    dev = ids.device
    n = ids.shape[1]
    pos = torch.arange(start, start + n, device=dev)
    return model(input_ids=ids, attention_mask=torch.ones((1, start + n), device=dev, dtype=torch.long),
                 position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=True)


@torch.no_grad()
def speculative_greedy(target, draft, input_ids, max_new_tokens, k=4, eos_id=None):
    dev = input_ids.device
    P = input_ids.shape[1]
    tc, dc = DynamicCache(), DynamicCache()
    torch.cuda.reset_peak_memory_stats()
    res = SpecResult(tokens=[])
    _sync(); t0 = time.perf_counter()
    # prefill both models with all prompt tokens except the last one (kept uncached as `last`)
    if P > 1:
        _fwd(target, input_ids[:, :-1], tc, 0)
        _fwd(draft, input_ids[:, :-1], dc, 0)
    res.target_forwards += 1; res.draft_forwards += 1
    _sync(); res.prefill_s = time.perf_counter() - t0
    last = input_ids[:, -1:]
    c = P - 1
    out = []
    while len(out) < max_new_tokens:
        kk = min(k, max_new_tokens - len(out))     # do not draft more than we can still emit
        # ---- 1. draft k tokens ----
        cur, drafts = last, []
        for j in range(kk):
            o = _fwd(draft, cur, dc, c + j)
            cur = o.logits[:, -1].argmax(-1, keepdim=True)
            drafts.append(cur)
            res.draft_forwards += 1
        d = torch.cat(drafts, dim=1)                                  # [1, kk]
        # ---- 2. verify with ONE target forward ----
        o = _fwd(target, torch.cat([last, d], dim=1), tc, c)
        res.target_forwards += 1
        p = o.logits[0].argmax(-1)                                     # [kk+1] target greedy picks
        # ---- 3. accept longest matching prefix ----
        match = (p[:kk] == d[0]).long()
        m = int(match.cumprod(0).sum().item())
        new = d[0, :m].tolist() + [int(p[m].item())]
        res.rounds += 1; res.drafted += kk; res.accepted += m
        res.accept_hist[m] = res.accept_hist.get(m, 0) + 1
        # ---- 4. cache sync ----
        tc.crop(c + m + 1)
        if m == kk:                          # draft never consumed its last proposal
            _fwd(draft, d[:, -1:], dc, c + kk)
            res.draft_forwards += 1
        dc.crop(c + m + 1)
        c += m + 1
        out += new
        last = torch.tensor([[new[-1]]], device=dev)
        if eos_id is not None and eos_id in new:
            out = out[:out.index(eos_id) + 1]
            break
    res.tokens = out[:max_new_tokens]
    _sync(); res.seconds = time.perf_counter() - t0
    res.peak_mem_MiB = torch.cuda.max_memory_allocated() / 2**20
    return res
