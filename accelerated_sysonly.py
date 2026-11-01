"""Track B: systems-only acceleration. The training recipe is LOCKED to the baseline's:
batch 64, SGD lr 0.01 (constant, no warmup, no schedule), momentum 0.9, weight decay 5e-4.
Only how the computation runs may change: data pipeline, threads, memory layout,
precision (bf16), compilation, fused optimizer, denormal flushing.

Usage: python accelerated_sysonly.py --channels-last --bf16 --target-acc 0.80 --out-dir results/track_b
"""
import sys

import accelerated

LOCKED = {"--batch-size": "64", "--base-lr": "0.01", "--schedule": "none", "--warmup-epochs": "0"}
FORBIDDEN = set(LOCKED) | {"--epochs-for-schedule"}

if __name__ == "__main__":
    user_args = sys.argv[1:]
    bad = [a for a in user_args if a.split("=")[0] in FORBIDDEN]
    if bad:
        sys.exit(f"Track B locks the training recipe; remove {bad}")
    if not any(a.split("=")[0] == "--name" for a in user_args):
        user_args += ["--name", "sysonly"]
    sys.argv = [sys.argv[0]] + user_args + [x for kv in LOCKED.items() for x in kv]
    accelerated.main()
