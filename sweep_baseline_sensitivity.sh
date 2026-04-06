#!/bin/bash
# Baseline sensitivity check: confirm calibration settings move final_int6_sliding_window_exact.
# Requires QUANT_ONLY_CHECKPOINT to point to a saved final_model.pt from a full training run.
#
# Usage:
#   QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt bash sweep_baseline_sensitivity.sh
#
# Decision rule:
#   - If BPB differs by >=1e-4 between A and B, calibration is a live lever. Proceed with sweep.
#   - If BPB is unchanged, calibration is saturated. Pivot to a different novelty lane.

set -euo pipefail

SCRIPT="train_gpt_calib_sweep.py"
LOGDIR="logs/baseline_sensitivity"
mkdir -p "$LOGDIR"

if [ -z "${QUANT_ONLY_CHECKPOINT:-}" ]; then
    echo "ERROR: Set QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt"
    exit 1
fi

echo "=== Baseline A: 64 seqs x 2048 tokens (SOTA default) ==="
CALIB_NUM_SEQS=64 CALIB_SEQ_LEN=2048 CALIB_TEMPERATURE=0.8 \
QUANT_ONLY_CHECKPOINT="$QUANT_ONLY_CHECKPOINT" \
RUN_ID="baseline_A_64x2048" \
python3 "$SCRIPT" 2>&1 | tee "$LOGDIR/baseline_A_64x2048.log"

echo ""
echo "=== Baseline B: 256 seqs x 512 tokens (same total tokens) ==="
CALIB_NUM_SEQS=256 CALIB_SEQ_LEN=512 CALIB_TEMPERATURE=0.8 \
QUANT_ONLY_CHECKPOINT="$QUANT_ONLY_CHECKPOINT" \
RUN_ID="baseline_B_256x512" \
python3 "$SCRIPT" 2>&1 | tee "$LOGDIR/baseline_B_256x512.log"

echo ""
echo "=== Baseline C: 128 seqs x 1024 tokens (same total tokens) ==="
CALIB_NUM_SEQS=128 CALIB_SEQ_LEN=1024 CALIB_TEMPERATURE=0.8 \
QUANT_ONLY_CHECKPOINT="$QUANT_ONLY_CHECKPOINT" \
RUN_ID="baseline_C_128x1024" \
python3 "$SCRIPT" 2>&1 | tee "$LOGDIR/baseline_C_128x1024.log"

echo ""
echo "=== Results comparison ==="
echo "Extracting final_int6_sliding_window_exact from each run..."
for logfile in "$LOGDIR"/baseline_*.log; do
    name=$(basename "$logfile" .log)
    bpb=$(grep "final_int6_sliding_window_exact" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    gen_time=$(grep "gptq:generated" "$logfile" | tail -1 | grep -oP 'in \K[0-9.]+' || echo "N/A")
    echo "  $name: BPB=$bpb gen_time=${gen_time}s"
done

echo ""
echo "If BPB values differ by >=1e-4, proceed with sweep_token_schedules.sh"
echo "If BPB values are identical, calibration is saturated — pivot strategy needed."
