# P2N Megatron: Qwen3-style 600M

从随机初始化训练 P2N 的 600M Qwen3 风格模型。模型使用 [Megatron Core](https://github.com/NVIDIA/Megatron-LM)；四卡训练通过 PyTorch DDP 同步，TP=PP=1。不加载 Qwen 权重，不使用 LLaMA-Factory。

## Quick start · Lumia

仓库放在 `/home/ztwang/p2n-megatron`；数据、缓存和检查点放在 `/data3/ztwang/p2n-megatron`。

```bash
cd /home/ztwang/p2n-megatron
bash scripts/setup_megatron.sh
sbatch scripts/smoke_4gpu.sbatch
squeue -u ztwang
tail -f /data3/ztwang/p2n-megatron/logs/smoke-<JOBID>.out
```

Smoke 使用集群共享的只读 5B-token smallpile，运行 4×4090、2 step、64-token 序列。它验证参数量、P2N 迭代、有限梯度、四卡同步和检查点。若要测试论文的 2,048-token 长度：

```bash
SMOKE_SEQ_LEN=2048 SMOKE_STEPS=1 SMOKE_SAVE_EVERY=0 \
SMOKE_CHECKPOINTING=1 sbatch scripts/smoke_4gpu.sbatch
```

## Training

准备由目标 tokenizer 编码的 Pile token 流，格式为 Megatron `uint16 .bin`，token ID 小于 50,304。正式训练约需 120B token；现有 smallpile 不够。论文正文没有唯一指定 tokenizer，因此必须记录正式数据所用 tokenizer 与 EOD ID。

```bash
TRAIN_BIN=/data3/ztwang/p2n-megatron/data/pile_train.bin \
sbatch scripts/train_4gpu.sbatch

# 相同架构和数据顺序的 Vanilla 对照
TRAIN_BIN=/data3/ztwang/p2n-megatron/data/pile_train.bin \
METHOD=vanilla sbatch scripts/train_4gpu.sbatch
```

可选变量：`VALID_BIN` 设置独立验证集；`EOD_ID` 设置文档结束 token（默认 `50256`）；`RESUME_CHECKPOINT` 从保存的 `step_*.pt` 继续；`TRAIN_CHECKPOINTING=0` 关闭激活重计算。日志和检查点保存在 `/data3/ztwang/p2n-megatron/{logs,checkpoints}`。默认 57,221 step、序列 2,048、全局 batch 1,024、每 500 step 保存一次。

## Method & configuration

| 项目 | 设置 |
|---|---|
| 架构 | 21 层，hidden 1,280，FFN 5,632，Q/KV heads 20/5，RoPE、Q/K RMSNorm、SwiGLU |
| 参数 | 604,627,328；词表 50,304；共享输入/输出词嵌入 |
| P2N | 前 7 层 + 共享 core 7 层 + 后 7 层；core warm pass 后每 batch 采样 K=2/3 次 Jacobi 更新，完整反向传播 |
| 优化 | AdamW，peak LR 1.5e-3，5% warmup，cosine 至 10%，BF16，seed 42 |

P2N 反馈为 `core_input + ShiftPrev(previous_core_output)`；序列首位和 EOD 后清零，注意力也隔离 packed 文档。`method=vanilla` 使用相同参数层和 token 预算，不执行反馈迭代。

## Files

```text
p2n/model.py                 Megatron 模型与 P2N 递推
p2n/data.py                  uint16 token 流读取
train.py                     训练、验证、检查点
scripts/setup_megatron.sh    固定 Megatron Core 版本并检查环境
scripts/env.sh               Lumia 运行环境
scripts/smoke_4gpu.sbatch    四卡 smoke
scripts/train_4gpu.sbatch    四卡预训练
```

## Reproduction status

四卡 smoke 已验证：K=2/3、保存与恢复、独立验证集、Vanilla 对照、2,048-token 长度和梯度累积。24 GiB 4090 在 2,048 token 下需要激活重计算，正式脚本默认开启。该项目实现**预训练阶段**；完整 Pile 训练、后续 SFT/评测和带 KV cache 的生成尚未完成，smoke 结果不能视为论文指标。

Megatron 固定为 `core_v0.13.0`（`c550cf6c41c31cd3ec72e05c25ea0c979f2b6631`）。Lumia 脚本复用账号已有的 PyTorch 2.6.0 环境及 `/data3/ztwang/p2n-megatron/shim`；在新机器上需先准备等效的 PyTorch/CUDA 环境并修改 `scripts/env.sh`。项目结构参考 [PonderLM-2](https://github.com/LUMIA-Group/PonderLM-2)。
