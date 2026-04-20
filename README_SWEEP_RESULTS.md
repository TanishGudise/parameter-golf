# Calibration Sensitivity Sweep — Null Result

## Context

Extended abaybektursun's PR #1019 (Mar 25, 1.1147 BPB) with three novel levers ported into `train_gpt_calib_sweep.py` (2339 lines):
1. **QUANT_ONLY_CHECKPOINT** — skip training, load checkpoint, run calibration → GPTQ → eval (~15 min vs 6-8 hr retrain)
2. **CALIB_SPLIT_BY_MODULE** — independent `(num_seqs, seq_len)` for attention vs MLP Hessian collection
3. **QK_GAIN_INIT_SCHEDULE** — per-layer QK-gain init schedule (comma-separated float list)

## Methodology

Trained a smoke checkpoint on the PR #1019 base (3k iterations, 1×H100, ~30 min). Then ran an A/B/C sensitivity sweep in QUANT_ONLY mode — each calibration config takes ~15 minutes — varying the shape of the calibration data while keeping total tokens constant at 131,072.

| Config | num_seqs | seq_len | Total tokens |
|--------|----------|---------|--------------|
| A      | 64       | 2048    | 131,072      |
| C      | 128      | 1024    | 131,072      |
| B      | 256      | 512     | 131,072      |

Scripts: `sweep_baseline_sensitivity.sh`, `sweep_mixed_regime.sh`
Checkpoint: `final_model.pt` (smoke, ~3k iters)
Logs: `logs/baseline_sensitivity/`

## Results

| Config      | roundtrip_exact | sliding_window_exact |
|-------------|-----------------|----------------------|
| A (64×2048) | 1.19148724      | 1.16799711           |
| C (128×1024)| 1.19124363      | 1.16775830           |
| B (256×512) | 1.19117343      | 1.16770591           |

**Signal: monotonic (A > C > B, fewer-longer-seqs = worse) but ~200× smaller than expected.**

On a pre-SDClip base tested earlier, the same schedule shift produced −0.069 BPB. On the PR #1019 stack with GPTQ SDClip, the same shift yields only −0.00029 BPB — three orders of magnitude smaller.

## Interpretation

GPTQ SDClip (`c * std(row)` threshold, introduced in PR #1394's lineage) appears to saturate calibration sensitivity. The outlier-clipping mechanism makes the choice of calibration token distribution largely irrelevant to final quantization error: SDClip normalizes outliers independently of the Hessian signal, so changing the Hessian calibration data has minimal effect on which rows get clipped and how.

The mixed-regime lever (Lever 2) has limited ceiling on the PR #1019 stack.

## Implication

Pivoting to SP8192 base (PR #1394, 1.0856 BPB) where:
1. The attention regime is fundamentally different (vocab 8192 → larger embedding space, q_gain=4.0 → stronger attention sharpening)
2. The calibration sensitivity landscape may be different — the SP8192 model has different Hessian structure due to architecture changes
3. The per-layer QK-gain schedule (Lever 3) is the primary novel contribution worth testing on the new base

See branch `sp8192-rebase` and `train_gpt_sp8192_opt.py`.

## BPB Gap

| Milestone | BPB | Notes |
|-----------|-----|-------|
| PR #1019 base (our prior) | 1.1147 | Mar 25, abaybektursun |
| SP8192 base (rebase target) | 1.0856 | Apr 5, Kevin Clark (PR #1394) |
| bigbag SOTA | 1.0810 | Apr 9, PR #1493 |
| Gap to close vs SOTA | -0.0046 | after rebase |

Rebase alone closes −0.0291 BPB (26% of original gap to SOTA).

## Reproducibility

```bash
# On the pod, with checkpoint at final_model.pt:
QUANT_ONLY_CHECKPOINT=final_model.pt \
  CALIB_NUM_SEQS=64 CALIB_SEQ_LEN=2048 \
  python train_gpt_calib_sweep.py 2>&1 | tee logs/config_A.log

QUANT_ONLY_CHECKPOINT=final_model.pt \
  CALIB_NUM_SEQS=128 CALIB_SEQ_LEN=1024 \
  python train_gpt_calib_sweep.py 2>&1 | tee logs/config_C.log

QUANT_ONLY_CHECKPOINT=final_model.pt \
  CALIB_NUM_SEQS=256 CALIB_SEQ_LEN=512 \
  python train_gpt_calib_sweep.py 2>&1 | tee logs/config_B.log
```
