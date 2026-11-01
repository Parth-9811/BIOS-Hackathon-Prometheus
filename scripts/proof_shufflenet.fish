# ShuffleNetV2 Track A proof set. Usage from the project root: fish scripts/proof_shufflenet.fish 15
set E $argv[1]
test -z "$E"; and set E 15
set M --model shufflenet_v2_x1_0 --target-acc 0.72
set A --threads 8 --batch-size 256 --base-lr 0.1 --schedule onecycle
set OUT results/shufflenet_v2_x1_0/track_a
set -e OMP_NUM_THREADS
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
test -d $OUT; and mv $OUT {$OUT}_old_(date +%H%M%S)
for s in 0 1 2
    python baseline.py $M --seed $s --max-epochs 40 --out-dir $OUT; and sleep 120
    python accelerated.py $M $A --max-epochs $E --seed $s --out-dir $OUT; and sleep 120
end
python summarize.py $OUT | tee $OUT/summary.md
python make_plots.py $OUT plots/shufflenet_v2_x1_0
