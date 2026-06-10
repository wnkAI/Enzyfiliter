import numpy as np
import pandas as pd


def _ranks(score_dict, ids):
    """rank 1 = best (highest score)."""
    vals = np.array([score_dict[i] for i in ids], dtype=float)
    order = np.argsort(-vals)            # desc
    rank = np.empty(len(ids), dtype=int)
    rank[order] = np.arange(1, len(ids) + 1)
    return rank


def aggregate(ids, views, k=60, penalty=0.5):
    """Parameter-free rank aggregation (Reciprocal Rank Fusion) over the views.
    No learned weights -> no label needed, no overfitting attack surface.
    Cross-view disagreement is reported as a calibrated uncertainty AND used to
    down-weight candidates the views can't agree on (penalty), so a sequence two
    views love but the third rejects no longer rockets to the top on RRF alone."""
    names = list(views.keys())
    rank_cols = {f"rank_{n}": _ranks(views[n], ids) for n in names}

    R = np.vstack([rank_cols[f"rank_{n}"] for n in names]).T   # N x V
    rrf = (1.0 / (k + R)).sum(axis=1)

    # disagreement: spread of per-view ranks, normalized by N
    disagree = (R.max(1) - R.min(1)) / len(ids)
    adj = rrf * (1.0 - penalty * disagree)        # consistency-weighted score

    df = pd.DataFrame({"id": ids})
    for n in names:
        df[f"score_{n}"] = [views[n][i] for i in ids]
    for c, v in rank_cols.items():
        df[c] = v
    df["rrf"] = rrf
    df["disagreement"] = disagree
    df["adj_score"] = adj
    df = df.sort_values("adj_score", ascending=False).reset_index(drop=True)
    df.insert(1, "final_rank", np.arange(1, len(df) + 1))
    return df
