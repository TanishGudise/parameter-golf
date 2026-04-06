#!/bin/bash
# Final 3-seed record run on 8xH100.
# Once you've found the best calibration settings, fill them in below and run this.
#
# Usage (on an 8xH100 node):
#   bash run_record_3seed.sh

set -euo pipefail

SCRIPT="train_gpt_calib_sweep.py"
RECORD_DIR="records/track_10min_16mb/$(date +%Y-%m-%d)_OptCalib_GPTQ_XSA_BigramHash3072"
mkdir -p "$RECORD_DIR"

# === FILL IN YOUR BEST SETTINGS ===
export CALIB_NUM_SEQS="${CALIB_NUM_SEQS:-64}"
export CALIB_SEQ_LEN="${CALIB_SEQ_LEN:-2048}"
export CALIB_TEMPERATURE="${CALIB_TEMPERATURE:-0.8}"
export CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-8}"
export CALIB_TOP_P="${CALIB_TOP_P:-0.0}"
export CALIB_TOP_K="${CALIB_TOP_K:-0}"
export GPTQ_BLOCK_SIZE="${GPTQ_BLOCK_SIZE:-128}"
export GPTQ_DAMP_RATIO="${GPTQ_DAMP_RATIO:-0.01}"

echo "Record run config:"
echo "  CALIB: ${CALIB_NUM_SEQS}x${CALIB_SEQ_LEN} temp=${CALIB_TEMPERATURE} top_p=${CALIB_TOP_P} top_k=${CALIB_TOP_K}"
echo "  GPTQ: block_size=${GPTQ_BLOCK_SIZE} damp_ratio=${GPTQ_DAMP_RATIO}"
echo "  Output: $RECORD_DIR"
echo ""

for seed in 42 314 999; do
    echo "=== Seed $seed ==="
    export SEED="$seed"
    export RUN_ID="train_seed${seed}"

    torchrun --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "$RECORD_DIR/train_seed${seed}.log"

    if [ -f "final_model.int6.ptz" ]; then
        cp "final_model.int6.ptz" "$RECORD_DIR/final_model_seed${seed}.int6.ptz"
    fi
    echo ""
done

echo "=== Extracting results ==="
echo "Seed | BPB (sliding_window_exact) | Artifact Size"
echo "---- | --- | ---"
for seed in 42 314 999; do
    logfile="$RECORD_DIR/train_seed${seed}.log"
    bpb=$(grep "final_int6_sliding_window_exact" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    artifact=$(grep "Total submission size" "$logfile" | tail -1 | grep -oP ': \K[0-9]+' || echo "N/A")
    echo "  $seed | $bpb | $artifact bytes"
done

# Copy the script into the record directory
cp "$SCRIPT" "$RECORD_DIR/train_gpt.py"

echo ""
echo "Record directory: $RECORD_DIR"
echo "Don't forget to create README.md and submission.json in the record directory!"
