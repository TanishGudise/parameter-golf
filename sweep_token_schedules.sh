#!/bin/bash
# Sweep calibration token design: (num_seqs, seq_len), temperature, prompting.
# Run AFTER sweep_baseline_sensitivity.sh confirms calibration moves the needle.
#
# Usage:
#   QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt bash sweep_token_schedules.sh
#
# All runs hold total tokens constant at ~131072 unless noted.

set -euo pipefail

SCRIPT="train_gpt_calib_sweep.py"
LOGDIR="logs/token_schedule_sweep"
mkdir -p "$LOGDIR"

if [ -z "${QUANT_ONLY_CHECKPOINT:-}" ]; then
    echo "ERROR: Set QUANT_ONLY_CHECKPOINT=/path/to/final_model.pt"
    exit 1
fi

run_sweep() {
    local name="$1"
    shift
    echo "=== Running: $name ==="
    RUN_ID="$name" QUANT_ONLY_CHECKPOINT="$QUANT_ONLY_CHECKPOINT" \
        "$@" python3 "$SCRIPT" 2>&1 | tee "$LOGDIR/${name}.log"
    echo ""
}

# --- Sequence schedule sweep (constant total tokens ~131072, temp=0.8) ---
run_sweep "sched_64x2048_t0.8"   CALIB_NUM_SEQS=64   CALIB_SEQ_LEN=2048 CALIB_TEMPERATURE=0.8
run_sweep "sched_128x1024_t0.8"  CALIB_NUM_SEQS=128  CALIB_SEQ_LEN=1024 CALIB_TEMPERATURE=0.8
run_sweep "sched_256x512_t0.8"   CALIB_NUM_SEQS=256  CALIB_SEQ_LEN=512  CALIB_TEMPERATURE=0.8
run_sweep "sched_512x256_t0.8"   CALIB_NUM_SEQS=512  CALIB_SEQ_LEN=256  CALIB_TEMPERATURE=0.8
run_sweep "sched_1024x128_t0.8"  CALIB_NUM_SEQS=1024 CALIB_SEQ_LEN=128  CALIB_TEMPERATURE=0.8
run_sweep "sched_2048x64_t0.8"   CALIB_NUM_SEQS=2048 CALIB_SEQ_LEN=64   CALIB_TEMPERATURE=0.8

# --- Temperature sweep (best schedule from above, or default 64x2048 for now) ---
for temp in 0.6 0.7 0.8 0.9 1.0 1.2; do
    run_sweep "temp_64x2048_t${temp}" CALIB_NUM_SEQS=64 CALIB_SEQ_LEN=2048 "CALIB_TEMPERATURE=$temp"
done

# --- Temperature x schedule cross (128x1024 at various temps) ---
for temp in 0.7 0.9 1.0; do
    run_sweep "temp_128x1024_t${temp}" CALIB_NUM_SEQS=128 CALIB_SEQ_LEN=1024 "CALIB_TEMPERATURE=$temp"
done

# --- Increased total tokens (2x and 4x at best schedule) ---
run_sweep "more_tokens_128x2048_t0.8" CALIB_NUM_SEQS=128 CALIB_SEQ_LEN=2048 CALIB_TEMPERATURE=0.8
run_sweep "more_tokens_256x1024_t0.8" CALIB_NUM_SEQS=256 CALIB_SEQ_LEN=1024 CALIB_TEMPERATURE=0.8

# --- Top-k sampling variants ---
for topk in 50 100 200; do
    run_sweep "topk_${topk}_64x2048_t0.8" CALIB_NUM_SEQS=64 CALIB_SEQ_LEN=2048 CALIB_TEMPERATURE=0.8 "CALIB_TOP_K=$topk"
done

# --- Top-p (nucleus) sampling variants ---
for topp in 0.9 0.95; do
    run_sweep "topp_${topp}_64x2048_t0.8" CALIB_NUM_SEQS=64 CALIB_SEQ_LEN=2048 CALIB_TEMPERATURE=0.8 "CALIB_TOP_P=$topp"
done

echo ""
echo "=== RESULTS SUMMARY ==="
echo "Run | BPB (sliding_window_exact) | Gen Time (s) | Hess Time (s)"
echo "--- | --- | --- | ---"
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    bpb=$(grep "final_int6_sliding_window_exact" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    gen_time=$(grep "gptq:generated" "$logfile" | tail -1 | grep -oP 'in \K[0-9.]+' || echo "N/A")
    hess_time=$(grep "gptq:collected hessians.*in " "$logfile" | tail -1 | grep -oP 'in \K[0-9.]+' || echo "N/A")
    echo "$name | $bpb | $gen_time | $hess_time"
done
