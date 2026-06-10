import os
import shutil
import subprocess
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
GAP = 20  # gap / unknown


def read_fasta(path):
    ids, seqs, cur, buf = [], [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip()
        if not line:
            continue
        if line[0] == ">":
            if cur is not None:
                ids.append(cur); seqs.append("".join(buf))
            cur = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if cur is not None:
        ids.append(cur); seqs.append("".join(buf))
    return ids, seqs


def build_msa(fasta, out=None, threads=8, backend="auto"):
    """Align sequences. Prefers MAFFT (servers), falls back to pyfamsa (pip,
    no system binary). Returns path to aligned fasta."""
    out = out or os.path.splitext(fasta)[0] + ".aln.fasta"
    if backend in ("auto", "mafft") and shutil.which("mafft"):
        with open(out, "w") as fh:
            subprocess.run(["mafft", "--auto", "--thread", str(threads), fasta],
                           check=True, stdout=fh, stderr=subprocess.DEVNULL)
        return out
    from pyfamsa import Aligner, Sequence
    ids, seqs = read_fasta(fasta)
    msa = Aligner(guide_tree="upgma").align(
        [Sequence(i.encode(), s.encode()) for i, s in zip(ids, seqs)])
    with open(out, "w") as fh:
        for r in msa:
            fh.write(f">{r.id.decode()}\n{r.sequence.decode()}\n")
    return out


def _encode_alignment(aln_fasta):
    ids, gseqs = read_fasta(aln_fasta)
    L = len(gseqs[0])
    M = np.full((len(gseqs), L), GAP, dtype=np.int8)
    for r, s in enumerate(gseqs):
        for c, ch in enumerate(s):
            M[r, c] = AA_IDX.get(ch.upper(), GAP)
    return ids, M


def _column_freqs(M, pseudo=0.5):
    N, L = M.shape
    freqs = np.zeros((L, 21))
    for c in range(L):
        col = M[:, c]
        for a in range(21):
            freqs[c, a] = np.count_nonzero(col == a)
    freqs += pseudo
    freqs /= freqs.sum(axis=1, keepdims=True)
    return freqs


def conservation_scores(aln_fasta):
    """Viewpoint 1: data-driven function-critical-site weighting.
    High-conservation columns get more weight (no catalytic-site prior needed).
    Each sequence scored by how faithfully it matches the conserved consensus."""
    ids, M = _encode_alignment(aln_fasta)
    freqs = _column_freqs(M)
    aa_freq = freqs[:, :20]
    aa_freq = aa_freq / aa_freq.sum(axis=1, keepdims=True)
    # column conservation = 1 - normalized Shannon entropy over 20 aa
    ent = -(aa_freq * np.log(aa_freq + 1e-12)).sum(axis=1)
    conservation = 1.0 - ent / np.log(20)          # 0..1, high = conserved
    gap_frac = (M == GAP).mean(axis=0)
    w = conservation * (1.0 - gap_frac)            # downweight gappy columns
    denom = w.sum() + 1e-9
    scores = {}
    for r, sid in enumerate(ids):
        res = M[r]
        contrib = 0.0
        for c in range(M.shape[1]):
            a = res[c]
            if a == GAP:
                continue                            # missing a key position hurts
            contrib += w[c] * aa_freq[c, a]         # reward matching frequent residue
        scores[sid] = contrib / denom
    return scores


def coevolution_scores(aln_fasta, top_pairs=200, max_gap=0.5, pseudo=0.5):
    """Viewpoint 3: coupled-residue network (APC-corrected mutual information).
    Picks the strongest co-evolving column pairs (a data-driven proxy for the
    functional/active-site network) and scores each sequence by how well it
    preserves the high-frequency coupled residue combinations."""
    ids, M = _encode_alignment(aln_fasta)
    N, L = M.shape
    gap_frac = (M == GAP).mean(axis=0)
    aa_freq = _column_freqs(M)[:, :20]
    aa_freq = aa_freq / aa_freq.sum(axis=1, keepdims=True)
    ent = -(aa_freq * np.log(aa_freq + 1e-12)).sum(axis=1)
    keep = np.where((gap_frac < max_gap) & (ent > 0.1))[0]   # informative columns

    # pairwise MI on kept columns
    def col_onehot(c):
        oh = np.zeros((N, 20))
        col = M[:, c]
        m = col != GAP
        oh[np.arange(N)[m], col[m]] = 1.0
        return oh, m

    min_obs = max(5, int(0.1 * N))      # adaptive: enough pairs to estimate MI
    mi = {}
    cache = {c: col_onehot(c) for c in keep}
    for ii in range(len(keep)):
        ci = keep[ii]
        ohi, mi_mask = cache[ci]
        for jj in range(ii + 1, len(keep)):
            cj = keep[jj]
            ohj, mj_mask = cache[cj]
            both = mi_mask & mj_mask
            n = both.sum()
            if n < min_obs:
                continue
            pi = ohi[both].mean(0) + 1e-9
            pj = ohj[both].mean(0) + 1e-9
            joint = (ohi[both].T @ ohj[both]) / n + 1e-9
            val = (joint * np.log(joint / np.outer(pi, pj))).sum()
            mi[(ci, cj)] = val
    if not mi:
        return {sid: 0.0 for sid in ids}

    # APC correction
    cols = sorted(keep.tolist())
    idx = {c: k for k, c in enumerate(cols)}
    mimat = np.zeros((len(cols), len(cols)))
    for (a, b), v in mi.items():
        mimat[idx[a], idx[b]] = v; mimat[idx[b], idx[a]] = v
    mean_i = mimat.mean(1)
    mean_all = mimat.mean()
    apc = np.outer(mean_i, mean_i) / (mean_all + 1e-9)
    corrected = {}
    for (a, b), v in mi.items():
        corrected[(a, b)] = v - apc[idx[a], idx[b]]

    top = sorted(corrected, key=corrected.get, reverse=True)[:top_pairs]

    # pointwise mutual information tables for top pairs.
    # PMI = log p(a,b) - log p(a) - log p(b) removes the single-site marginals
    # (which viewpoint 1 already captures), leaving only the pairwise epistatic
    # signal -> keeps this view orthogonal to conservation.
    jtab = {}
    for (a, b) in top:
        col_a, col_b = M[:, a], M[:, b]
        both = (col_a != GAP) & (col_b != GAP)
        t = np.zeros((20, 20)) + pseudo
        for x, y in zip(col_a[both], col_b[both]):
            t[x, y] += 1
        t /= t.sum()
        pa = t.sum(1, keepdims=True)
        pb = t.sum(0, keepdims=True)
        jtab[(a, b)] = np.log(t) - np.log(pa) - np.log(pb)

    scores = {}
    for r, sid in enumerate(ids):
        res = M[r]
        s, n = 0.0, 0
        for (a, b) in top:
            ra, rb = res[a], res[b]
            if ra == GAP or rb == GAP:
                continue
            s += jtab[(a, b)][ra, rb]; n += 1
        scores[sid] = s / n if n else -50.0          # no coverage = penalized
    return scores
