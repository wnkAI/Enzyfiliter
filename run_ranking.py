import argparse
from msa_signals import read_fasta, build_msa, conservation_scores, coevolution_scores
from esm_likelihood import esm_scores
from fuse import aggregate


def main():
    ap = argparse.ArgumentParser(description="Label-free enzyme ranking (3-view consensus)")
    ap.add_argument("--fasta", required=True, help="candidate sequences (e.g. 154 seqs)")
    ap.add_argument("--out", default="ranking.csv")
    ap.add_argument("--msa", default=None, help="precomputed aligned fasta; else run mafft")
    ap.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--masked", action="store_true", help="true pseudo-LL (slow)")
    ap.add_argument("--top-pairs", type=int, default=200)
    ap.add_argument("--penalty", type=float, default=0.5,
                    help="how hard to down-weight high-disagreement candidates (0=off)")
    ap.add_argument("--no-esm", action="store_true", help="skip ESM (debug)")
    a = ap.parse_args()

    ids, seqs = read_fasta(a.fasta)
    print(f"{len(ids)} sequences")

    aln = a.msa or build_msa(a.fasta)
    print(f"MSA: {aln}")

    print("view 1: conservation ...")
    cons = conservation_scores(aln)
    print("view 3: coevolution ...")
    coev = coevolution_scores(aln, top_pairs=a.top_pairs)

    views = {"conservation": cons, "coevolution": coev}
    if not a.no_esm:
        print("view 2: ESM-2 likelihood ...")
        views["esm"] = esm_scores(ids, seqs, model_name=a.esm_model,
                                  device=a.device, masked=a.masked)

    df = aggregate(ids, views, penalty=a.penalty)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
