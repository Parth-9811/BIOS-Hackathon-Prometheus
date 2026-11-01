# Track B per-epoch sweep: 2 epochs each, compare the EPOCH-2 time (epoch 1 includes compile/warm-up).
# Run from the project root: fish scripts/sweep_track_b.fish   (about 20-25 minutes)
set OUT results/track_b_sweep
set C --max-epochs 2 --target-acc 1.01 --threads 8 --out-dir $OUT
set -e OMP_NUM_THREADS
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
python accelerated_sysonly.py $C --tag fp32
python accelerated_sysonly.py $C --channels-last --tag cl
python accelerated_sysonly.py $C --channels-last --bf16 --tag cl_bf16
python accelerated_sysonly.py $C --channels-last --bf16 --flush-denormal --tag cl_bf16_fd
python accelerated_sysonly.py $C --channels-last --bf16 --compile --tag cl_bf16_c
python accelerated_sysonly.py $C --channels-last --bf16 --compile --compile-mode max-autotune-no-cudagraphs --tag cl_bf16_cmax
# Memory allocator swap (often 5-15% at small batch). Install first: sudo pacman -S jemalloc  /  sudo apt install libjemalloc2
set JE (ldconfig -p | grep -m1 libjemalloc.so | awk '{print $NF}')
if test -n "$JE"
    env LD_PRELOAD=$JE python accelerated_sysonly.py $C --channels-last --bf16 --tag cl_bf16_jemalloc
else
    echo "jemalloc not installed, skipping allocator test"
end
python summarize.py $OUT
