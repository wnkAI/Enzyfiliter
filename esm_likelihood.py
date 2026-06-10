import numpy as np
import torch


def _load(model_name):
    import esm
    fn = getattr(esm.pretrained, model_name)
    model, alphabet = fn()
    model.eval()
    return model, alphabet


@torch.no_grad()
def esm_scores(ids, seqs, model_name="esm2_t33_650M_UR50D",
               device="cuda", masked=False, batch_size=8, max_len=1022):
    """Viewpoint 2: protein language-model pseudo-log-likelihood.
    Per-sequence 'naturalness' under ESM-2. Higher = better.
    masked=False : single forward, mean log-prob of the true residues (fast).
    masked=True  : true pseudo-LL, mask each position (slow, more rigorous)."""
    model, alphabet = _load(model_name)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = model.to(device)
    bc = alphabet.get_batch_converter()
    layers = model.num_layers
    out = {}

    clipped = [(i, s[:max_len]) for i, s in zip(ids, seqs)]

    if not masked:
        for k in range(0, len(clipped), batch_size):
            chunk = clipped[k:k + batch_size]
            data = [(sid, s) for sid, s in chunk]
            _, _, toks = bc(data)
            toks = toks.to(device)
            logits = model(toks)["logits"]
            logp = torch.log_softmax(logits, dim=-1)
            for r, (sid, s) in enumerate(chunk):
                L = len(s)
                tot = 0.0
                for pos in range(1, L + 1):                # skip BOS at 0
                    tot += logp[r, pos, toks[r, pos]].item()
                out[sid] = tot / L
            print(f"  esm {min(k+batch_size,len(clipped))}/{len(clipped)}")
        return out

    mask_idx = alphabet.mask_idx
    for sid, s in clipped:
        _, _, toks = bc([(sid, s)])
        toks = toks.to(device)
        L = len(s)
        tot = 0.0
        for pos in range(1, L + 1):
            masked_toks = toks.clone()
            true = toks[0, pos].item()
            masked_toks[0, pos] = mask_idx
            logits = model(masked_toks)["logits"]
            logp = torch.log_softmax(logits[0, pos], dim=-1)
            tot += logp[true].item()
        out[sid] = tot / L
        print(f"  esm(masked) {sid}")
    return out
