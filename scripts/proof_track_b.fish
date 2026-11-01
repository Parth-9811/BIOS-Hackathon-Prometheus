# Track B proof set (systems only, recipe locked). Edit B to the fastest flags from the sweep.
# Run from the project root: fish scripts/proof_track_b.fish   (about 1.5-2 hours)
set B --channels-last --bf16 --threads 8
set OUT results/track_b
set -e OMP_NUM_THREADS
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
test -d $OUT; and mv $OUT {$OUT}_old_(date +%H%M%S)
for s in 0 1 2
    python baseline.py --seed $s --target-acc 0.80 --out-dir $OUT; and sleep 120
    python accelerated_sysonly.py $B --max-epochs 40 --seed $s --target-acc 0.80 --out-dir $OUT; and sleep 120
end
python summarize.py $OUT | tee $OUT/summary.md
python make_plots.py $OUT plots/track_b
