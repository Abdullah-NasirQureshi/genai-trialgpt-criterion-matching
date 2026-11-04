"""Task 3: KV-cached autoregressive greedy decoding, written as an explicit loop.

Two decoders over the SAME model and weights:

* greedy_nocache : reference. Every step re-runs the model on the WHOLE sequence so
                   far (prompt + generated tokens) and takes argmax of the last logits.
                   Cost per step grows with the sequence length (O(n) tokens re-processed).
* greedy_kv      : prefill once over the prompt, then every step feeds ONLY the newest
                   token. We manage the cache ourselves:
                     - cache object      : transformers DynamicCache, created by us, passed in
                                           and updated in place by every attention layer
                                           (K,V of the new token are concatenated to the stored
                                           K,V of layer l -> cache grows by 1 position / step)
                     - position_ids      : we pass the absolute position (= current cache
                                           length) so RoPE rotates the new token correctly
                     - cache_position    : same index, used by the cache for bookkeeping
                     - attention_mask    : all-ones [1, cache_len + 1]; the new query attends to
                                           every cached key. Causality is automatic because the
                                           single query is the LAST position (no future keys
                                           exist); during prefill the model builds the causal
                                           lower-triangular mask itself.
Both return generated token ids and timing (prefill / decode) plus peak GPU memory.
"""
import inspect
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


def _keep_kw(model):
    """name of the 'only compute logits for last n positions' kwarg (varies by version)."""
    params = inspect.signature(model.forward).parameters
    for n in ("logits_to_keep", "num_logits_to_keep"):
        if n in params:
            return n
    return None


@dataclass
class DecodeResult:
    tokens: list
    prefill_s: float = 0.0
    decode_s: float = 0.0
    peak_mem_MiB: float = 0.0
    cache_len_trace: list = field(default_factory=list)  # cache length after each step (KV only)
    cache_bytes: int = 0


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cache_nbytes(cache):
    total = 0
    for layer in range(len(cache)):
        k, v = cache[layer]
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


@torch.no_grad()
def greedy_nocache(model, input_ids, max_new_tokens, eos_id=None):
    """Reference decoder: NO cache. Re-encodes the entire sequence at every step."""
    keep = _keep_kw(model)
    kw = {keep: 1} if keep else {}
    ids = input_ids
    out = []
    torch.cuda.reset_peak_memory_stats()
    _sync(); t0 = time.perf_counter()
    prefill_s = 0.0
    for step in range(max_new_tokens):
        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **kw)
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        if step == 0:
            _sync(); prefill_s = time.perf_counter() - t0
        out.append(nxt.item())
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and out[-1] == eos_id:
            break
    _sync()
    total = time.perf_counter() - t0
    return DecodeResult(out, prefill_s, total - prefill_s, torch.cuda.max_memory_allocated() / 2**20)


@torch.no_grad()
def greedy_kv(model, input_ids, max_new_tokens, eos_id=None, trace=False):
    """Cache-aware decoder: prefill once, then one-token forward passes reusing the cache."""
    dev = input_ids.device
    keep = _keep_kw(model)
    kw = {keep: 1} if keep else {}
    P = input_ids.shape[1]
    cache = DynamicCache()
    torch.cuda.reset_peak_memory_stats()
    # ---- prefill: whole prompt, builds K,V for all prompt positions in every layer ----
    _sync(); t0 = time.perf_counter()
    pos = torch.arange(P, device=dev)
    o = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
              position_ids=pos[None], cache_position=pos, past_key_values=cache,
              use_cache=True, **kw)
    nxt = o.logits[:, -1].argmax(-1, keepdim=True)
    _sync(); prefill_s = time.perf_counter() - t0
    out = [nxt.item()]
    trace_l = [cache.get_seq_length()] if trace else []
    # ---- decode: feed only the last generated token ----
    t1 = time.perf_counter()
    while len(out) < max_new_tokens and not (eos_id is not None and out[-1] == eos_id):
        L = cache.get_seq_length()                       # tokens already cached
        p = torch.tensor([L], device=dev)                # absolute position of the new token
        o = model(input_ids=nxt, attention_mask=torch.ones((1, L + 1), device=dev, dtype=torch.long),
                  position_ids=p[None], cache_position=p, past_key_values=cache,
                  use_cache=True, **kw)
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
        if trace:
            trace_l.append(cache.get_seq_length())
    _sync()
    decode_s = time.perf_counter() - t1
    return DecodeResult(out, prefill_s, decode_s, torch.cuda.max_memory_allocated() / 2**20,
                        trace_l, cache_nbytes(cache))


def make_prompt_of_length(tok, text_pool, n_tokens):
    """Real clinical text repeated/cut to exactly n_tokens (for controlled length sweeps)."""
    ids = []
    i = 0
    while len(ids) < n_tokens:
        ids += tok(text_pool[i % len(text_pool)] + "\n", add_special_tokens=False).input_ids
        i += 1
    return torch.tensor([ids[:n_tokens]])
