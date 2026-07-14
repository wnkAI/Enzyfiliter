# Enzyfiliter

Label-free, training-free ranking of enzyme candidates from sequence alone. Three orthogonal signals (conservation, ESM-2 pseudo-likelihood, coevolution) vote via RRF fusion.

## Install

```bash
pip install torch fair-esm numpy pandas pyfamsa
```

## Run

```bash
python run_ranking.py --fasta data/alse_154_candidates.fasta --out ranking.csv
```

Runs on CPU in seconds to minutes (ESM-2 8M, weights auto-downloaded on first run).

Pick the top candidates by `final_rank` in `ranking.csv`.
