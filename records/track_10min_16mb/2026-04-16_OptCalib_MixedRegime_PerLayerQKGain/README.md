# Record: Optimized Calibration + Mixed-Regime GPTQ + Per-Layer QK-Gain

**val_bpb: TODO** (3-seed mean, std TODO) | **~15.9 MB** | 8xH100 SXM, 600s

**Improvement over current SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), 1.1147 BPB):** TODO nats (TODO BPB)

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 42 | TODO | TODO | TODO | **TODO** | TODO |
| 314 | TODO | TODO | TODO | **TODO** | TODO |
| 999 | TODO | TODO | TODO | **TODO** | TODO |
| **Mean** | | | | **TODO** | |

## Main Changes

The comparison baseline is [PR #1019](https://github.com/openai/parameter-golf/pull/1019), the current SOTA at **1.1147 BPB**.

### 1. Mixed-Regime GPTQ Calibration (Novel)

Prior work uses a single (num_seqs, seq_len) setting for all GPTQ Hessian collection. We observe that attention and MLP layers have different Hessian statistics requirements:
- **Attention layers** benefit from longer sequences that capture long-range positional correlations in Q/K matrices.
- **MLP layers** benefit from shorter, more numerous sequences that provide better IID sampling of the activation distribution.

We split the calibration into two independent passes: attention-module hooks are run on (attn_num_seqs x attn_seq_len) sequences, MLP-module hooks on (mlp_num_seqs x mlp_seq_len) sequences, each generated with independent random seeds. The total token budget is preserved per module type.

Best config: TODO

### 2. Per-Layer QK-Gain Init Schedule

The global `qk_gain_init=1.5` is replaced with a per-layer schedule. Since `q_gain` is a learnable parameter, this is a strictly more expressive initialization — the optimizer can converge to the global value if that happens to be optimal for all layers. In practice, early layers (broad attention) and late layers (content-specific attention) have different optimal scales.

Best schedule: TODO

### 3. Optimized Calibration Token Design

Swept (num_seqs, seq_len) grid at fixed 131K total tokens, plus temperature, top-k/top-p, and GPTQ damping/block-size hypers.

Best config: TODO

## Architecture

Same as PR #1019 (11L 512d, 8 GQA heads, 4 KV heads, XSA-all, LeakyReLU(0.5)^2 MLP3x, BigramHash 3072x112, Partial RoPE 16/64, Parallel Muon, EMA+SWA, Full Hessian GPTQ int6, LZMA9). The only changes are to calibration and QK-gain initialization.

## Requirements

Same as PR #1019. Flash Attention 3 (Hopper) required.

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard
```

## Run Command

```bash
# Fill in winning settings
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=42 \
CALIB_SPLIT_BY_MODULE=1 \
CALIB_ATTN_NUM_SEQS=TODO CALIB_ATTN_SEQ_LEN=TODO \
CALIB_MLP_NUM_SEQS=TODO CALIB_MLP_SEQ_LEN=TODO \
QK_GAIN_INIT_SCHEDULE="TODO" \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

```
PR #1019 (SOTA, 1.1147) — AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112
    └── This work adds:
        ├── Mixed-regime GPTQ calibration (independent attn/MLP Hessian data)
        ├── Per-layer QK-gain init schedule
        └── Optimized calibration token design (swept num_seqs x seq_len x temp x damping x block_size)
```
