#!/bin/bash
# Sweep mixed-regime calibration: independent (num_seqs, seq_len) for attn vs mlp.
# Hypothesis: attn Hessians want long-range context; MLP Hessians want IID sampling.
#
# Usage:
#   QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt bash sweep_mixed_regime.sh
#
# Fill in CALIB_TEMPERATURE etc. with the best settings from sweep_token_schedules.sh.

set -euo pipefail

SCRIPT="train_gpt_calib_sweep.py"
LOGDIR="logs/mixed_regime_sweep"
mkdir -p "$LOGDIR"

if [ -z "${QUANT_ONLY_CHECKPOINT:-}" ]; then
    echo "ERROR: Set QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt"
    exit 1
fi

CALIB_TEMPERATURE="${CALIB_TEMPERATURE:-0.8}"
GPTQ_BLOCK_SIZE="${GPTQ_BLOCK_SIZE:-128}"
GPTQ_DAMP_RATIO="${GPTQ_DAMP_RATIO:-0.01}"

run_sweep() {
    local name="$1"
    shift
    echo "=== Running: $name ==="
    RUN_ID="$name" QUANT_ONLY_CHECKPOINT="$QUANT_ONLY_CHECKPOINT" \
        CALIB_SPLIT_BY_MODULE=1 \
        CALIB_TEMPERATURE="$CALIB_TEMPERATURE" \
        GPTQ_BLOCK_SIZE="$GPTQ_BLOCK_SIZE" \
        GPTQ_DAMP_RATIO="$GPTQ_DAMP_RATIO" \
        "$@" python3 "$SCRIPT" 2>&1 | tee "$LOGDIR/${name}.log"
    echo ""
}

# --- Mixed regime: attn gets long seqs, MLP gets short+many seqs ---
# Total tokens per module type held at ~131072

# Attn=(64,2048) MLP=(256,512)
run_sweep "attn64x2048_mlp256x512" \
    CALIB_ATTN_NUM_SEQS=64 CALIB_ATTN_SEQ_LEN=2048 \
    CALIB_MLP_NUM_SEQS=256 CALIB_MLP_SEQ_LEN=512

# Attn=(64,2048) MLP=(512,256)
run_sweep "attn64x2048_mlp512x256" \
    CALIB_ATTN_NUM_SEQS=64 CALIB_ATTN_SEQ_LEN=2048 \
    CALIB_MLP_NUM_SEQS=512 CALIB_MLP_SEQ_LEN=256

# Attn=(64,2048) MLP=(1024,128)
run_sweep "attn64x2048_mlp1024x128" \
    CALIB_ATTN_NUM_SEQS=64 CALIB_ATTN_SEQ_LEN=2048 \
    CALIB_MLP_NUM_SEQS=1024 CALIB_MLP_SEQ_LEN=128

# Attn=(128,1024) MLP=(256,512)
run_sweep "attn128x1024_mlp256x512" \
    CALIB_ATTN_NUM_SEQS=128 CALIB_ATTN_SEQ_LEN=1024 \
    CALIB_MLP_NUM_SEQS=256 CALIB_MLP_SEQ_LEN=512

# Attn=(128,1024) MLP=(512,256)
run_sweep "attn128x1024_mlp512x256" \
    CALIB_ATTN_NUM_SEQS=128 CALIB_ATTN_SEQ_LEN=1024 \
    CALIB_MLP_NUM_SEQS=512 CALIB_MLP_SEQ_LEN=256

# Attn=(128,1024) MLP=(1024,128)
run_sweep "attn128x1024_mlp1024x128" \
    CALIB_ATTN_NUM_SEQS=128 CALIB_ATTN_SEQ_LEN=1024 \
    CALIB_MLP_NUM_SEQS=1024 CALIB_MLP_SEQ_LEN=128

# Control: same config for both (equivalent to non-split), tests overhead
run_sweep "control_64x2048_both" \
    CALIB_ATTN_NUM_SEQS=64 CALIB_ATTN_SEQ_LEN=2048 \
    CALIB_MLP_NUM_SEQS=64 CALIB_MLP_SEQ_LEN=2048

echo ""
echo "=== RESULTS SUMMARY ==="
echo "Run | BPB (sliding_window_exact) | Gen Time (s) | Hess Time (s)"
echo "--- | --- | --- | ---"
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    bpb=$(grep "final_int6_sliding_window_exact" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    gen_time=$(grep "gptq:generated" "$logfile" | tail -1 | grep -oP 'in \K[0-9.]+' || echo "N/A")
    hess_time=$(grep "gptq:collected" "$logfile" | tail -1 | grep -oP 'in \K[0-9.]+' || echo "N/A")
    echo "$name | $bpb | $gen_time | $hess_time"
done
