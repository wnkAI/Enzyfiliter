# Enzyfiliter — label-free enzyme ranking (3-view consensus)

纯序列、无活性标签、无训练的酶候选优先级排序。三个机制正交的弱信号投票:

1. **conservation** — MSA 高保守列(数据驱动的功能关键位点)的共识保真度
2. **esm** — ESM-2 序列伪似然(自然度)
3. **coevolution** — APC-corrected MI 强耦合残基网络的兼容性(PMI 形式,与 conservation 正交)

无参数 RRF 融合 + 跨视角分歧度作为不确定性。定位是「候选优先级」而非「预测活性」。

## 安装

```bash
pip install torch fair-esm numpy pandas pyfamsa
```

MSA 用 pyfamsa(pip 自带,无需系统二进制);若本机装了 MAFFT 会优先用 MAFFT。

## 运行

```bash
python run_ranking.py --fasta data/alse_154_candidates.fasta --out ranking.csv
```

默认用 **ESM-2 8M、CPU**,几秒到几分钟跑完,无需 GPU。首次自动下载 8M 权重(~30 MB)。

可选参数:
- `--esm-model esm2_t33_650M_UR50D --device cuda` 换更大模型 + GPU
- `--masked` 严格逐位 mask 的 pseudo-LL(慢很多)
- `--msa aligned.fasta` 跳过对齐,用已对齐文件
- `--top-pairs 200` 共进化取多少强耦合列对
- `--penalty 0.5` disagreement 惩罚强度(0 = 关闭)
- `--no-esm` 只跑 MSA 两个视角(调试)

## 输出 ranking.csv

| 列 | 含义 |
|---|---|
| `final_rank` | 最终优先级(1 = 最高) |
| `adj_score` / `rrf` | 惩罚后分(排序依据)/ 原始 RRF 分 |
| `rank_conservation` / `rank_esm` / `rank_coevolution` | 各视角单独排名 |
| `score_*` | 各视角原始分 |
| `disagreement` | 跨视角分歧度(0~1,越大越不确定) |

挑候选:按 `final_rank` 取 top-K,优先选 `disagreement` 低的。
