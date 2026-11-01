# Track A proof set (systems + recipe). Run from the project root: fish scripts/proof_track_a.fish
set E 13
set A --channels-last --bf16 --schedule onecycle --threads 8 --batch-size 256 --base-lr 0.1
set OUT results/track_a
set -e OMP_NUM_THREADS
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
test -d $OUT; and mv $OUT {$OUT}_old_(date +%H%M%S)
for s in 0 1 2
    python baseline.py --seed $s --target-acc 0.80 --out-dir $OUT; and sleep 120
    python accelerated.py $A --max-epochs $E --seed $s --target-acc 0.80 --out-dir $OUT; and sleep 120
end
python summarize.py $OUT | tee $OUT/summary.md
python make_plots.py $OUT plots/track_a
