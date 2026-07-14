# Enzyfiliter — label-free enzyme ranking (3-view consensus)

Sequence-only, activity-label-free, training-free prioritization of enzyme candidates. Three mechanistically orthogonal weak signals vote:

1. **conservation** — consensus fidelity at highly conserved MSA columns (data-driven functionally critical sites)
2. **esm** — ESM-2 sequence pseudo-likelihood (naturalness)
3. **coevolution** — compatibility with the strongly coupled residue network from APC-corrected MI (PMI form, orthogonal to conservation)

Parameter-free RRF fusion, with cross-view disagreement as an uncertainty estimate. This is candidate *prioritization*, not activity *prediction*.

## Installation

```bash
pip install torch fair-esm numpy pandas pyfamsa
```

MSA uses pyfamsa (bundled via pip, no system binary required); if MAFFT is installed locally it is preferred.

## Usage

```bash
python run_ranking.py --fasta data/alse_154_candidates.fasta --out ranking.csv
```

Defaults to **ESM-2 8M on CPU**, finishing in seconds to minutes, no GPU needed. The 8M weights (~30 MB) are downloaded automatically on first run.

Optional arguments:
- `--esm-model esm2_t33_650M_UR50D --device cuda` — switch to a larger model + GPU
- `--masked` — strict per-position masked pseudo-LL (much slower)
- `--msa aligned.fasta` — skip alignment and use a pre-aligned file
- `--top-pairs 200` — number of strongly coupled column pairs used for coevolution
- `--penalty 0.5` — disagreement penalty strength (0 = off)
- `--no-esm` — run only the two MSA views (debugging)

## Output ranking.csv

| Column | Meaning |
|---|---|
| `final_rank` | final priority (1 = highest) |
| `adj_score` / `rrf` | penalized score (ranking basis) / raw RRF score |
| `rank_conservation` / `rank_esm` / `rank_coevolution` | per-view individual ranks |
| `score_*` | per-view raw scores |
| `disagreement` | cross-view disagreement (0–1, higher = more uncertain) |

Picking candidates: take the top-K by `final_rank`, preferring those with low `disagreement`.
