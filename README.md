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


## Layout

```
run_ranking.py      Main entry: computes the three signals and fuses them into one ranking
esm_likelihood.py   ESM-2 pseudo-likelihood signal (single sequence, no MSA)
msa_signals.py      MSA-based signals: residue conservation + coevolution
fuse.py             Reciprocal-rank fusion (RRF) of the three per-signal rankings
```
