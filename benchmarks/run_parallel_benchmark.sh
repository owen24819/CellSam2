#!/bin/bash
# Base vs small CellSAM2: how many positions can be processed in parallel on one GPU?
#
# Usage: bash benchmarks/run_parallel_benchmark.sh
# Override any setting via env vars, e.g.
#   MAX_FRAMES=20 N_LIST="1 2 4 8 16" bash benchmarks/run_parallel_benchmark.sh

source /home/max/.bashrc
conda activate /home/max/MaxWork/conda_envs/cellsam2

cd /home/max/MaxWork/CS2_CHOTA_Debug
export PYTHONPATH="$PWD:$PYTHONPATH"

BASE_MODEL=${BASE_MODEL:-CellSam2-2D-a0p15}
SMALL_MODEL=${SMALL_MODEL:-CellSam2-2D-small-a0p15}
CKPT=${CKPT:-20}
SEQUENCE=${SEQUENCE:-data/2D/test/CTC/10}   # 2048x2048, 64 frames -> 25 crops/frame
MAX_FRAMES=${MAX_FRAMES:-30}                # shorten each movie so the sweep finishes in reasonable time
N_LIST=${N_LIST:-"1 2 4 8 16 32 64"}
OUT=${OUT:-results/parallel_bench}
TRIM_FLAG=${TRIM_FLAG:-}                    # set TRIM_FLAG=--trim_memory for the trimmed-history run
SKIP_ENCODER=${SKIP_ENCODER:-0}             # 1 = skip the image-encoder batch test

mkdir -p "$OUT"
nvidia-smi

# 1) Full tracking pipeline, N positions in parallel (one process each)
for MODEL in "$BASE_MODEL" "$SMALL_MODEL"; do
  python benchmarks/parallel_positions_benchmark.py \
    --model_name "$MODEL" \
    --checkpoint_num "$CKPT" \
    --sequence "$SEQUENCE" \
    --max_frames "$MAX_FRAMES" \
    --n_list $N_LIST \
    --refine $TRIM_FLAG \
    --out_csv "$OUT/parallel_positions.csv" \
    --out_dir "$OUT/outputs" \
    2>&1 | tee "$OUT/parallel_positions_${MODEL}.log"
done

# 1b) Full tracking pipeline, N positions in parallel sharing ONE model (threads, one process)
for MODEL in "$BASE_MODEL" "$SMALL_MODEL"; do
  python benchmarks/shared_model_benchmark.py \
    --model_name "$MODEL" \
    --checkpoint_num "$CKPT" \
    --sequence "$SEQUENCE" \
    --max_frames "$MAX_FRAMES" \
    --n_list ${SHARED_N_LIST:-$N_LIST 128} \
    --refine $TRIM_FLAG \
    --out_csv "$OUT/shared_model.csv" \
    --out_dir "$OUT/outputs" \
    2>&1 | tee "$OUT/shared_model_${MODEL}.log"
done

# 2) Image encoder only: largest batch that fits + throughput
[ "$SKIP_ENCODER" = 1 ] || for MODEL in "$BASE_MODEL" "$SMALL_MODEL"; do
  for DTYPE in fp32 bf16; do
    python benchmarks/batched_encoder_benchmark.py \
      --model_name "$MODEL" \
      --checkpoint_num "$CKPT" \
      --sequence "$SEQUENCE" \
      --dtype "$DTYPE" \
      --out_csv "$OUT/batched_encoder.csv" \
      2>&1 | tee "$OUT/batched_encoder_${MODEL}_${DTYPE}.log"
  done
done

# 3) Summary table + plots
python benchmarks/plot_parallel_benchmark.py --results_dir "$OUT"
