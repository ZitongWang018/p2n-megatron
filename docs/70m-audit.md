# 70M results and implementation audit

## Training and evaluation

Both runs completed 2,836 optimizer steps with 74,325,248 parameters, six
physical layers, a two-layer recurrent core, sequence length 2,048, global
batch 256, eight GPUs, microbatch 8, and four accumulation steps. They consumed
1,486,880,768 tokens (20.005 tokens per parameter) without dataset wrapping.
The original recipe used peak LR 1.5e-3, weight decay 0.01, 1% warmup, cosine
decay to 10% of peak, BF16, seed 42, and activation checkpointing.

Final checkpoints were evaluated on the same first 4,096 held-out sequences
(8,388,608 tokens), using causal next-token cross-entropy over all positions.

| Model | Jacobi updates | Validation loss | Perplexity |
| --- | ---: | ---: | ---: |
| Vanilla | 0 | 3.430757 | 30.9000 |
| P2N | 3 | 3.504279 | 33.2574 |

P2N's loss is 0.073522 higher; its perplexity is 7.63% higher. These are
single-seed results. The online 256-sequence validation gave 3.633560 and
3.703917 respectively; its different absolute values reflect a smaller
validation subset, not another metric.

The separate 1,024-sequence diagnostic sweep gave P2N losses of 5.027401,
3.644157, 3.461118, 3.462116, 3.494158, and 3.548081 for K=0 through K=5.
K=0 on P2N-trained weights is not a parameter-matched baseline: the baseline
has independently trained weights. More iterations are not guaranteed to
improve an extrapolation beyond the training range K in {2,3}.

## Paper protocol differences

The paper's smallest reported dense model is 150M, not this six-layer model.
Its backbone ties embeddings; the borrowed 70M configuration does not.
The paper uses global batch 1,024, weight decay 0.1, 5% warmup, and TPP200.
Batch 256 and TPP20 were chosen for this exploratory experiment. Weight
decay and warmup were additional deviations in the original launcher.
The paper disables activation recomputation; this run enables it for 24GB GPUs.
The existing Pile token stream was reused; exact identity with the paper's
data preprocessing has not been established.

The results support a comparison of these two short experiments; they do
not reproduce the paper's reported scaling results.

## Independent implementation checks

On 2026-10-02, Slurm audit job 133443 loaded the actual final P2N checkpoint
and passed all checks below. Reference and gradient checks use two 64-token
sequences, including document boundaries; the reported final evaluation above
uses full 2,048-token sequences.

| Check | Observed result |
| --- | --- |
| Optimizer update counters | Both runs have all 51 parameter states at step 2,836, including a nonzero output-head momentum |
| Raw data and one-token-ahead labels | Match independent memmap slices |
| Independent Hugging Face Qwen3 reference, K=0/2/3 | Maximum logits errors 1.91e-5 / 1.53e-5 / 1.62e-5 |
| Future-token perturbation | Earlier logits unchanged |
| Previous-document perturbation | Following document logits unchanged |
| BF16 checkpointing on/off | All 51 parameter-gradient tensors match exactly |
| Full batch versus gradient accumulation | Maximum gradient error 1.49e-7 |
| Prefix/core/suffix calls with K=3 | [1, 1, 4, 4, 1, 1] |
| Gradients through warm pass and every update | All four state gradient norms nonzero |

The HF reference maps the trained QKV, GQA, Q/K norms, RoPE, SwiGLU, and
output weights, then independently implements the paper's recurrence and
document masks. The production P2N model was unchanged between the original
70M run and this audit; later changes only added another model profile.
These checks found no recurrence, label-shift, causality, checkpointing,
or accumulation defect explaining the observed loss gap. They do not prove
equivalence at every input or eliminate statistical and protocol effects.

```bash
sbatch scripts/audit_p2n_1gpu.sbatch
```

## Corrections

The original `--verify` checked gradients in only three representative layers.
It now also checks every trainable parameter, including both embeddings and
the output head, for missing or nonfinite gradients. The 70M launcher's
defaults now use weight decay 0.1 and 5% warmup, with a separate checkpoint
directory. The original checkpoint profile and results are preserved.
The revised verification guard was also checked by deliberately removing
the output-head gradient; it correctly raised an assertion.
No new 70M training was launched as part of this audit.

Training curves: [Vanilla](https://swanlab.cn/@ZitongWang/P2N/runs/t3rd8bv7),
[P2N](https://swanlab.cn/@ZitongWang/P2N/runs/26c6cks2).
