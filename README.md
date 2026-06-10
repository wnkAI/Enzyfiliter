# seqrank — label-free enzyme ranking (3-view consensus)

纯序列、无活性标签的酶候选优先级排序。三个机制正交的弱信号投票:

1. **conservation** — MSA 高保守列(数据驱动的功能关键位点)的共识保真度
2. **esm** — ESM-2 650M 序列伪似然(自然度)
3. **coevolution** — APC-corrected MI 强耦合残基网络的兼容性

无参数 RRF 融合(无标签 → 不学权重 → 无过拟合攻击面),并输出跨视角分歧度作为不确定性。
**定位是「候选优先级」而非「预测活性」**:三视角都排高 = 高置信;分歧大 = 高不确定、不强排。

## 安装(服务器)

```bash
pip install torch fair-esm numpy pandas
conda install -c bioconda mafft        # 或 apt-get install mafft
```

## 运行

```bash
python run_ranking.py --fasta data/seqs154.fasta --out ranking.csv --device cuda
```

首次会自动下载 ESM-2 650M (~2.5GB) 权重。GPU 显存 ~3GB 足够。

可选参数:
- `--masked` 用严格 pseudo-LL(逐位 mask,慢很多,更 rigorous;论文终版用)
- `--msa aligned.fasta` 跳过 mafft,用已对齐文件
- `--top-pairs 200` 共进化取多少强耦合列对
- `--no-esm` 只跑 MSA 两个视角(无 GPU 时调试)

## 输出 ranking.csv

| 列 | 含义 |
|---|---|
| `final_rank` | 最终优先级(1=最高) |
| `rrf` | RRF 融合分(越大越前) |
| `rank_conservation` / `rank_esm` / `rank_coevolution` | 各视角单独排名 |
| `score_*` | 各视角原始分 |
| `disagreement` | 跨视角分歧度(0~1,越大越不确定) |

挑湿实验候选:按 `final_rank` 取 top-K,优先选 `disagreement` 低的。
