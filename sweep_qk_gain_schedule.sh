#!/bin/bash
# Sweep per-layer QK-gain init schedules.
# These require full 8xH100 training runs (retraining, not QUANT_ONLY).
#
# Usage (on 8xH100):
#   bash sweep_qk_gain_schedule.sh

set -euo pipefail

SCRIPT="train_gpt_calib_sweep.py"
LOGDIR="logs/qk_gain_schedule_sweep"
mkdir -p "$LOGDIR"

# Fill in best calibration settings from prior sweeps
export CALIB_NUM_SEQS="${CALIB_NUM_SEQS:-64}"
export CALIB_SEQ_LEN="${CALIB_SEQ_LEN:-2048}"
export CALIB_TEMPERATURE="${CALIB_TEMPERATURE:-0.8}"
export GPTQ_BLOCK_SIZE="${GPTQ_BLOCK_SIZE:-128}"
export GPTQ_DAMP_RATIO="${GPTQ_DAMP_RATIO:-0.01}"

run_sweep() {
    local name="$1"
    local schedule="$2"
    echo "=== Running: $name (schedule: $schedule) ==="
    RUN_ID="$name" SEED=42 \
        QK_GAIN_INIT_SCHEDULE="$schedule" \
        torchrun --standalone --nproc_per_node=8 "$SCRIPT" 2>&1 | tee "$LOGDIR/${name}.log"
    echo ""
}

# Sanity: uniform 1.5 — should match baseline exactly
run_sweep "uniform_1.5" "1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5"

# Schedule A: lower early, higher mid, moderate late
run_sweep "sched_A" "1.3,1.3,1.3,1.6,1.6,1.6,1.6,1.6,1.4,1.4,1.4"

# Schedule B: lower early, moderate mid+late
run_sweep "sched_B" "1.2,1.2,1.2,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5"

# Schedule C: gradually increasing
run_sweep "sched_C" "1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.7,1.6,1.5"

echo ""
echo "=== RESULTS SUMMARY ==="
echo "Schedule | BPB (sliding_window_exact) | Train Steps"
echo "--- | --- | ---"
for logfile in "$LOGDIR"/*.log; do
    name=$(basename "$logfile" .log)
    bpb=$(grep "final_int6_sliding_window_exact" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    steps=$(grep "stopping_early\|^step:" "$logfile" | tail -1 | grep -oP 'step:\K[0-9]+' || echo "N/A")
    echo "$name | $bpb | $steps"
done
