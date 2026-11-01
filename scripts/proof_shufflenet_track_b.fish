# ShuffleNetV2 Track B proof set (recipe locked). Run from the project root (about 1.5-2 h).
set M --model shufflenet_v2_x1_0 --target-acc 0.72
set OUT results/shufflenet_v2_x1_0/track_b
set -e OMP_NUM_THREADS
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
test -d $OUT; and mv $OUT {$OUT}_old_(date +%H%M%S)
for s in 0 1 2
    python baseline.py $M --seed $s --max-epochs 40 --out-dir $OUT; and sleep 120
    python accelerated_sysonly.py $M --threads 8 --max-epochs 40 --seed $s --out-dir $OUT; and sleep 120
end
python summarize.py $OUT | tee $OUT/summary.md
python make_plots.py $OUT plots/shufflenet_v2_x1_0_track_b
