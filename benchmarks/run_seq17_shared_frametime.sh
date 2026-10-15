#!/bin/bash
# Shared-model per-frame timing vs cell count, base vs small (bkg80), sequence 17, as-is code.
# Usage: bash benchmarks/run_seq17_shared_frametime.sh
cd /home/max/MaxWork/CS2_CHOTA_Debug
export PATH=/home/max/MaxWork/conda_envs/cellsam2/bin:$PATH
export PYTHONPATH="$PWD:$PYTHONPATH"
OUT=${OUT:-results/parallel_bench_bkg80_seq17_shared_frametime}
mkdir -p "$OUT"
for MODEL in CellSam2-2D-bkg80 CellSam2-2D-small-bkg80; do
  python benchmarks/shared_model_benchmark.py \
    --model_name "$MODEL" --checkpoint_num 20 \
    --sequence data/2D/test/CTC/17 \
    --n_list ${N_LIST:-1 2 4 8} --frame_metrics \
    --out_csv "$OUT/shared_model.csv" --out_dir "$OUT/outputs"
done
