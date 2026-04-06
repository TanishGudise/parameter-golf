#!/bin/bash
# (Low priority) Sweep GPTQ damping and block size on the best token schedule.
# Run AFTER sweep_token_schedules.sh to find the best calibration settings.
#
# Usage:
#   QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt \
#   CALIB_NUM_SEQS=... CALIB_SEQ_LEN=... CALIB_TEMPERATURE=... \
#   bash sweep_gptq_hypers.sh
#
# Set the CALIB_* env vars to your best token schedule from the previous sweep.

set -euo pipefail

SCRIPT="train_gpt_calib_sweep.py"
LOGDIR="logs/gptq_hyper_sweep"
mkdir -p "$LOGDIR"

if [ -z "${QUANT_ONLY_CHECKPOINT:-}" ]; then
    echo "ERROR: Set QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt"
    exit 1
fi

CALIB_NUM_SEQS="${CALIB_NUM_SEQS:-64}"
CALIB_SEQ_LEN="${CALIB_SEQ_LEN:-2048}"
CALIB_TEMPERATURE="${CALIB_TEMPERATURE:-0.8}"

echo "Using calibration: ${CALIB_NUM_SEQS}x${CALIB_SEQ_LEN} temp=${CALIB_TEMPERATURE}"

run_sweep() {
    local name="$1"
    shift
    echo "=== Running: $name ==="
    RUN_ID="$name" QUANT_ONLY_CHECKPOINT="$QUANT_ONLY_CHECKPOINT" \
        CALIB_NUM_SEQS="$CALIB_NUM_SEQS" CALIB_SEQ_LEN="$CALIB_SEQ_LEN" \
        CALIB_TEMPERATURE="$CALIB_TEMPERATURE" \
        "$@" python3 "$SCRIPT" 2>&1 | tee "$LOGDIR/${name}.log"
    echo ""
}

# --- Damping ratio sweep ---
for damp in 0.001 0.003 0.01 0.03 0.1; do
    run_sweep "damp_${damp}" "GPTQ_DAMP_RATIO=$damp"
done

# --- Block size sweep ---
for bs in 32 64 128 256; do
    run_sweep "blocksize_${bs}" "GPTQ_BLOCK_SIZE=$bs"
done

# --- Best damping x best block size cross ---
# Fill these in after running the individual sweeps above
# run_sweep "best_combo" "GPTQ_DAMP_RATIO=0.01" "GPTQ_BLOCK_SIZE=128"

echo ""
echo "=== RESULTS SUMMARY ==="
echo "Run | BPB (sliding_window_exact) | Quant Time (s)"
echo "--- | --- | ---"
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    bpb=$(grep "final_int6_sliding_window_exact" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    quant_time=$(grep "gptq:quantization complete" "$logfile" | tail -1 | grep -oP 'in \K[0-9.]+' || echo "N/A")
    echo "$name | $bpb | $quant_time"
done
