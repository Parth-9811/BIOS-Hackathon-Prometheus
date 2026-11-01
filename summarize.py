"""Summarise one track's results into a markdown table.
Usage: python summarize.py results/track_a"""
import glob, json, os, statistics, sys

d = sys.argv[1] if len(sys.argv) > 1 else "results"
runs = [json.load(open(p)) | {"file": os.path.basename(p)} for p in sorted(glob.glob(os.path.join(d, "*.json")))]
base = [r["time_to_target_s"] for r in runs if r["name"] == "baseline" and r["reached"]]
base_e = [r["energy_to_target_j"] for r in runs if r["name"] == "baseline" and r["reached"] and r["energy_to_target_j"]]
bt = statistics.median(base) if base else None
be = statistics.median(base_e) if base_e else None
print(f"Baseline median: {bt and round(bt,1)} s over {len(base)} run(s), {be and round(be)} J over {len(base_e)} run(s)\n")
print("| run | epochs | s/epoch | reached | time (s) | speedup | energy (J) | energy saving | final acc |")
print("|---|---|---|---|---|---|---|---|---|")
for r in runs:
    h = r["history"]
    per = (h[-1]["train_s"] - h[0]["train_s"]) / (len(h) - 1) if len(h) > 1 else h[0]["train_s"]
    t, e = r["time_to_target_s"], r["energy_to_target_j"]
    sp = f"{bt / t:.2f}x" if t and bt else "-"
    es = f"{be / e:.2f}x" if e and be else "-"
    print(f"| {r['file'][:-5]} | {len(h)} | {per:.1f} | {'yes' if r['reached'] else 'no'} | "
          f"{t and round(t,1) or '-'} | {sp} | {e and round(e) or '-'} | {es} | {r['final_acc']:.4f} |")
