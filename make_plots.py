"""Pitch figures for one track. Usage: python make_plots.py results/track_a plots/track_a"""
import glob, json, os, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "results"
runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "*.json")))]
out = sys.argv[2] if len(sys.argv) > 2 else "plots"
other = sorted({r["name"] for r in runs} - {"baseline"})
assert len(other) == 1, f"expected baseline + one accelerated variant in {d}, got {other}"
acc = other[0]
groups = {"baseline": [r for r in runs if r["name"] == "baseline"], acc: [r for r in runs if r["name"] == acc]}
COL = {"baseline": "#eb6834", acc: "#2a78d6"}
a0 = groups[acc][0]
desc = ", ".join(x for x in ["bf16" if a0.get("bf16") else "fp32", f"batch {a0.get('batch_size')}",
                             "one-cycle" if a0.get("schedule") == "onecycle" else "same LR recipe",
                             "compiled" if a0.get("compile") else None] if x)
LABEL = {"baseline": "Baseline (fp32, batch 64)", acc: f"Accelerated ({desc})"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
target = runs[0]["target_acc"]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
os.makedirs(out, exist_ok=True)

# 1. Test accuracy vs wall-clock training time, every seed
fig, ax = plt.subplots(figsize=(9, 5.2))
for name, rs in groups.items():
    for i, r in enumerate(rs):
        xs = [0] + [h["train_s"] for h in r["history"]]
        ys = [10] + [100 * h["test_acc"] for h in r["history"]]
        ax.plot(xs, ys, color=COL[name], lw=2, alpha=0.9, marker="o", ms=4,
                label=LABEL[name] if i == 0 else None, solid_capstyle="round")
    reached = [r["time_to_target_s"] for r in rs if r["time_to_target_s"]]
    if not reached:
        continue
    med = statistics.median(reached)
    ax.axvline(med, color=COL[name], lw=1, ls=":", alpha=0.8)
    ax.annotate(f"median {med:.0f} s", (med, 52), xytext=(6, 0), textcoords="offset points",
                color=INK2, fontsize=10)
ax.axhline(100 * target, color=INK2, lw=1, ls="--")
ax.annotate(f"target {100 * target:.0f}%", (0, 100 * target), xytext=(4, 4), textcoords="offset points",
            color=INK2, fontsize=10)
ax.set_xlabel("Training wall-clock time (s), eval excluded")
ax.set_ylabel("CIFAR-10 test accuracy (%)")
ax.set_ylim(20, 85); ax.set_xlim(0, None)
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
NICE = {"resnet18": "ResNet-18", "shufflenet_v2_x1_0": "ShuffleNetV2", "squeezenet1_1": "SqueezeNet", "mobilenet_v3_small": "MobileNetV3-small"}
model = NICE.get(groups["baseline"][0].get("model", "resnet18"), groups["baseline"][0].get("model"))
ax.set_title(f"Same accuracy, {len(groups[acc])} seeds each: {model} on a laptop CPU",
             loc="left", color=INK, fontsize=13)
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
fig.tight_layout(); fig.savefig(os.path.join(out, "accuracy_vs_time.png"), dpi=200); plt.close(fig)

# 2. Time and energy to target: two panels, own axes, dots = seeds, bar = median
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, key, unit, scale, title in [(axes[0], "time_to_target_s", "s", 1, f"Time to {100 * target:.0f}%"),
                                    (axes[1], "energy_to_target_j", "kJ", 1e-3, f"CPU package energy to {100 * target:.0f}%")]:
    meds = {}
    for i, (name, rs) in enumerate(groups.items()):
        if not any(r[key] for r in rs):
            continue
        vals = [r[key] * scale for r in rs if r[key]]
        meds[name] = statistics.median(vals)
        ax.bar(i, meds[name], width=0.55, color=COL[name], alpha=0.85)
        ax.scatter([i] * len(vals), vals, color=INK, s=18, zorder=3)
        ax.annotate(f"{meds[name]:.0f} {unit}" if unit == "s" else f"{meds[name]:.1f} {unit}",
                    (i + 0.3, meds[name]), xytext=(6, 0), textcoords="offset points", ha="left", va="center", color=INK)
    ax.set_xticks([0, 1], ["Baseline", "Accelerated"]); ax.set_xlim(-0.5, 1.75)
    if len(meds) < 2:
        continue
    ratio = meds["baseline"] / meds[acc]
    ax.set_title(f"{title}: {ratio:.2f}x less", loc="left", color=INK, fontsize=12)
    ax.set_ylabel(unit); ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "Bars = median, dots = individual seeds", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(out, "time_energy.png"), dpi=200); plt.close(fig)
print(f"wrote {out}/accuracy_vs_time.png, {out}/time_energy.png")
