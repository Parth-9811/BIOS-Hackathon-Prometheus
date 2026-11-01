"""Shared benchmark harness for the BIOS hackathon (baseline vs accelerated).

Both baseline.py and accelerated.py import everything that defines *the task*
from here: dataset, augmentation, eval protocol, seeding, timing, energy.
Only the training step itself is allowed to differ between the two scripts.
"""
import glob
import json
import os
import platform
import random
import time

import numpy as np
import torch
import torch.nn as nn

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------------
# Environment: threads, seeds, provenance
# ----------------------------------------------------------------------------
def configure_threads(num_threads=None):
    """Pin intra-op threads explicitly so both runs use the same budget.
    Default: physical cores (logical / 2 if SMT is on)."""
    if num_threads is None:
        num_threads = int(os.environ.get("BENCH_THREADS", 0)) or max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(num_threads)
    return num_threads


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def env_info():
    cpu = platform.processor()
    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and not cpu:
                    cpu = line.split(":", 1)[1].strip()
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                if line.startswith("flags"):
                    flags = line
                    break
    except OSError:
        pass
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "avx512_bf16": "avx512_bf16" in flags,
        "amx_bf16": "amx_bf16" in flags,
        "mkldnn": torch.backends.mkldnn.is_available(),
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "KMP_AFFINITY", "GOMP_CPU_AFFINITY", "LD_PRELOAD")},
    }


# ----------------------------------------------------------------------------
# Data: real CIFAR-10 (same shape as your synthetic set, but learnable)
# ----------------------------------------------------------------------------
def _cache_path(root):
    return os.path.join(root, "cifar10_uint8.pt")


def _from_huggingface(root):
    """CIFAR-10 from the Hugging Face CDN (~120 MB, usually much faster than the
    Toronto server). Needs: pip install datasets pillow"""
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10", cache_dir=os.path.join(root, "hf"))
    def conv(split):
        imgs = np.stack([np.asarray(im.convert("RGB")) for im in split["img"]])
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous(), torch.tensor(split["label"])
    return (*conv(ds["train"]), *conv(ds["test"]))


def _from_torchvision(root):
    """Original Toronto source (slow). Also works if you already have
    cifar-10-batches-py/ in root, e.g. copied from a teammate or USB stick."""
    from torchvision.datasets import CIFAR10
    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    to_t = lambda d: torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous()
    return to_t(tr), torch.tensor(tr.targets), to_t(te), torch.tensor(te.targets)


def load_cifar10_tensors(root="./data", synthetic=False, source="hf"):
    """Returns uint8 NCHW tensors: (train_x, train_y, test_x, test_y).
    Downloads once, then caches a single ~180 MB tensor file in root.
    synthetic=True gives random data for smoke tests only (no accuracy signal)."""
    if synthetic:
        g = torch.Generator().manual_seed(0)
        return (torch.randint(0, 256, (5000, 3, 32, 32), dtype=torch.uint8, generator=g),
                torch.randint(0, 10, (5000,), generator=g),
                torch.randint(0, 256, (1000, 3, 32, 32), dtype=torch.uint8, generator=g),
                torch.randint(0, 10, (1000,), generator=g))
    cache = _cache_path(root)
    if os.path.exists(cache):
        return torch.load(cache)
    if source == "hf" and not os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
        data = _from_huggingface(root)
    else:
        data = _from_torchvision(root)
    assert data[0].shape == (50000, 3, 32, 32) and data[2].shape == (10000, 3, 32, 32)
    os.makedirs(root, exist_ok=True)
    torch.save(data, cache)
    return data


class _PILDataset(torch.utils.data.Dataset):
    """Per-image PIL + transforms path, identical in cost to torchvision's CIFAR10."""
    def __init__(self, x_uint8, y, transform):
        self.x = x_uint8.permute(0, 2, 3, 1).numpy()
        self.y = y.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.fromarray(self.x[i])), self.y[i]


def standard_train_loader(train_x, train_y, batch_size, num_workers, seed):
    """The 'typical' PIL + torchvision.transforms + DataLoader pipeline (baseline)."""
    import torchvision.transforms as T
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(_PILDataset(train_x, train_y, tf), batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, generator=g,
                                       persistent_workers=num_workers > 0, drop_last=False)


def normalize(x_uint8):
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1) * 255
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1) * 255
    return (x_uint8.float() - mean) / std


def fast_augment(x_uint8, gen):
    """Vectorised equivalent of RandomCrop(32, pad=4) + RandomHorizontalFlip.
    Same augmentation distribution as standard_train_loader, no PIL, no workers."""
    n = x_uint8.shape[0]
    padded = torch.nn.functional.pad(x_uint8, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), generator=gen)
    dy = torch.randint(0, 9, (n,), generator=gen)
    idx = torch.arange(32)
    rows = (dy[:, None] + idx[None, :])[:, None, :, None].expand(n, 3, 32, 40)
    out = padded.gather(2, rows)
    cols = (dx[:, None] + idx[None, :])[:, None, None, :].expand(n, 3, 32, 32)
    out = out.gather(3, cols)
    flip = torch.rand(n, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)
    return out


# ----------------------------------------------------------------------------
# Eval: identical for both scripts. Do not fork this.
# ----------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model, test_x, test_y, batch_size=1000, memory_format=torch.contiguous_format):
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, test_x.shape[0], batch_size):
        x = normalize(test_x[i:i + batch_size]).contiguous(memory_format=memory_format)
        correct += (model(x).argmax(1) == test_y[i:i + batch_size]).sum().item()
    model.train(was_training)
    return correct / test_x.shape[0]


# ----------------------------------------------------------------------------
# Energy: Intel/AMD RAPL via powercap (Linux). Needs read access to energy_uj
# (often root-only: `sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj`).
# ----------------------------------------------------------------------------
class RaplMeter:
    def __init__(self):
        self.zones = [z for z in glob.glob("/sys/class/powercap/intel-rapl:[0-9]")
                      if os.access(os.path.join(z, "energy_uj"), os.R_OK)]
        self.available = bool(self.zones)
        self.joules = 0.0
        self._last = None

    def _read(self):
        vals = []
        for z in self.zones:
            with open(os.path.join(z, "energy_uj")) as f:
                e = int(f.read())
            with open(os.path.join(z, "max_energy_range_uj")) as f:
                m = int(f.read())
            vals.append((e, m))
        return vals

    def start(self):
        if self.available:
            self._last = self._read()

    def stop(self):
        """Accumulates energy since start(); handles counter wrap-around."""
        if not self.available or self._last is None:
            return
        now = self._read()
        for (e0, m), (e1, _) in zip(self._last, now):
            self.joules += ((e1 - e0) % m) / 1e6
        self._last = None


# ----------------------------------------------------------------------------
# Runner: time-to-target-accuracy with the clock paused during eval
# ----------------------------------------------------------------------------
def run(name, train_one_epoch, model, test_x, test_y, max_epochs, target_acc, seed,
        eval_memory_format=torch.contiguous_format, out_dir="results", extra=None):
    """train_one_epoch(epoch) -> mean train loss. Must do *only* training work.
    Wall-clock and energy are measured around it; eval is excluded from both."""
    meter = RaplMeter()
    history, t_train, tta, eta = [], 0.0, None, None
    for epoch in range(max_epochs):
        meter.start()
        t0 = time.perf_counter()
        loss = train_one_epoch(epoch)
        t_train += time.perf_counter() - t0
        meter.stop()
        acc = evaluate(model, test_x, test_y, memory_format=eval_memory_format)
        history.append({"epoch": epoch + 1, "train_s": round(t_train, 2), "loss": loss, "test_acc": acc,
                        "joules": round(meter.joules, 1) if meter.available else None})
        print(f"[{name}] epoch {epoch + 1:3d}  t={t_train:8.1f}s  loss={loss:.4f}  acc={acc:.4f}", flush=True)
        if tta is None and acc >= target_acc:
            tta, eta = t_train, (meter.joules if meter.available else None)
            break
    result = {"name": name, "seed": seed, "target_acc": target_acc, "reached": tta is not None,
              "time_to_target_s": tta, "energy_to_target_j": eta,
              "final_acc": history[-1]["test_acc"], "history": history, "env": env_info(), **(extra or {})}
    os.makedirs(out_dir, exist_ok=True)
    tag = (extra or {}).get("tag")
    path = os.path.join(out_dir, f"{name}{'_' + tag if tag else ''}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{name}] time_to_target={tta} s  energy={eta} J  -> {path}")
    return result


def common_args(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-acc", type=float, default=0.80)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--source", choices=["hf", "torchvision"], default="hf",
                        help="hf = Hugging Face mirror (fast); torchvision = Toronto server or local cifar-10-batches-py")
    parser.add_argument("--synthetic", action="store_true", help="smoke test only")
    parser.add_argument("--model", default="resnet18", choices=MODELS)
    parser.add_argument("--out-dir", default="results", help="where result JSONs go, e.g. results/track_b")
    parser.add_argument("--tag", default="", help="suffix for the results file, e.g. t16 -> accelerated_t16_seed0.json")
    return parser


MODELS = ("resnet18", "shufflenet_v2_x1_0", "squeezenet1_1", "mobilenet_v3_small")


def make_model(name="resnet18"):
    """Stock torchvision architecture, 10 output classes. The architecture is fixed for every run: never edit it."""
    import torchvision.models as tvm
    if name not in MODELS:
        raise ValueError(f"unknown model {name}; choose from {MODELS}")
    return getattr(tvm, name)(num_classes=10)
