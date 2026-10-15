#!/bin/bash
# Base vs small (bkg80), sequence 17 (most cells in the test set), memory history as-is vs trimmed.
# Everything runs sequentially: one model at a time, N copies of the same movie in parallel.
#
# Usage: bash benchmarks/run_seq17_asis_vs_trimmed.sh

cd /home/max/MaxWork/CS2_CHOTA_Debug
export PATH=/home/max/MaxWork/conda_envs/cellsam2/bin:$PATH
export PYTHONPATH="$PWD:$PYTHONPATH"

export BASE_MODEL=CellSam2-2D-bkg80
export SMALL_MODEL=CellSam2-2D-small-bkg80
export CKPT=20
export SEQUENCE=data/2D/test/CTC/17
export MAX_FRAMES=1000                      # i.e. all frames
VERIFY_OUT=results/parallel_bench_bkg80_seq17_verify

# 1) Trimming must not change the outputs
verify_ok=1
for MODEL in "$BASE_MODEL" "$SMALL_MODEL"; do
  mkdir -p "$VERIFY_OUT"
  python benchmarks/verify_memory_trim.py \
    --model_name "$MODEL" --checkpoint_num "$CKPT" \
    --sequences "$SEQUENCE" --out_dir "$VERIFY_OUT" \
    2>&1 | tee "$VERIFY_OUT/verify_${MODEL}.log"
  [ "${PIPESTATUS[0]}" = 0 ] || verify_ok=0
done

# 2) As-is
OUT=results/parallel_bench_bkg80_seq17_asis TRIM_FLAG= SKIP_ENCODER=0 \
N_LIST="1 2 4 8 16 24 32 40" SHARED_N_LIST="1 2 4 8 16 32 48 64" \
  bash benchmarks/run_parallel_benchmark.sh

# 3) Trimmed (only if outputs were identical)
if [ "$verify_ok" = 1 ]; then
  OUT=results/parallel_bench_bkg80_seq17_trimmed TRIM_FLAG=--trim_memory SKIP_ENCODER=1 \
  N_LIST="1 2 4 8 16 24 32 40" SHARED_N_LIST="1 2 4 8 16 32 48 64 96" \
    bash benchmarks/run_parallel_benchmark.sh
else
  echo "[run] trimmed outputs differ from as-is -> skipping trimmed sweeps"
fi
echo "[run] ALL DONE"
