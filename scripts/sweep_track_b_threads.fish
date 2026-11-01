# Track B follow-up sweep: thread count and layout at batch 64 (about 6-8 minutes)
set OUT results/track_b_sweep
set C --max-epochs 2 --target-acc 1.01 --bf16 --flush-denormal --out-dir $OUT
set -e OMP_NUM_THREADS
python accelerated_sysonly.py $C --threads 8 --tag bf16_nocl_t8
python accelerated_sysonly.py $C --channels-last --threads 6 --tag cl_bf16_t6
python accelerated_sysonly.py $C --channels-last --threads 4 --tag cl_bf16_t4
python accelerated_sysonly.py $C --threads 4 --tag bf16_nocl_t4
python summarize.py $OUT
