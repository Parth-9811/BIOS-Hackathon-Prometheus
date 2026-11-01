"""Track A: accelerated training (systems + training-recipe changes). Same model (H.make_model), same data and augmentation
(H.fast_augment == RandomCrop(32,4)+HFlip), same eval (H.evaluate).
Every lever is a flag so you can build the ablation table one row at a time."""
import argparse
import math

import torch
import torch.nn as nn

import bench_harness as H


def main():
    p = H.common_args(argparse.ArgumentParser())
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--base-lr", type=float, default=0.01, help="baseline LR at batch 64")
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--schedule", choices=["none", "cosine", "onecycle"], default="none")
    p.add_argument("--epochs-for-schedule", type=int, default=None, help="horizon for cosine/onecycle")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--bf16", action="store_true", help="only helps on CPUs with AVX512_BF16/AMX")
    p.add_argument("--compile-mode", default="default",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--flush-denormal", action="store_true", help="treat tiny floats as zero (CPU speed trick)")
    p.add_argument("--name", default="accelerated", help="result name prefix")
    p.add_argument("--profile", action="store_true", help="profile 30 training steps, print top ops, exit")
    a = p.parse_args()

    H.configure_threads(a.threads)
    if a.flush_denormal:
        torch.set_flush_denormal(True)
    H.set_seed(a.seed)
    train_x, train_y, test_x, test_y = H.load_cifar10_tensors(a.data, a.synthetic, a.source)
    mf = torch.channels_last if a.channels_last else torch.contiguous_format

    model = H.make_model(a.model).to(memory_format=mf)
    crit = nn.CrossEntropyLoss()
    lr = a.base_lr * a.batch_size / 64  # linear scaling rule
    try:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, fused=True)
    except (RuntimeError, TypeError):
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    steps_per_epoch = math.ceil(train_x.shape[0] / a.batch_size)
    horizon = (a.epochs_for_schedule or a.max_epochs) * steps_per_epoch
    warm = int(a.warmup_epochs * steps_per_epoch)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        if a.schedule == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, horizon - warm))))
        return 1.0

    if a.schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=horizon, pct_start=0.25)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def step_fn(x, y):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.bf16):
            loss = crit(model(x), y)
        loss.backward()
        opt.step()
        return loss.detach()

    if a.compile:
        # Compiling forward+backward of the model; the optimizer step stays eager (fused).
        # dynamic=False: the last batch of each epoch is smaller (50000 % B != 0). With the default,
        # Dynamo recompiles a slower dynamic-shape graph on that batch and uses it for every later step.
        # Static mode compiles one extra graph for the odd size and keeps both fast.
        model_c = torch.compile(model, dynamic=False, mode=a.compile_mode)
        def step_fn(x, y):  # noqa: F811
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.bf16):
                loss = crit(model_c(x), y)
            loss.backward()
            opt.step()
            return loss.detach()

    gen = torch.Generator().manual_seed(a.seed)
    model.train()

    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        def batch(i):
            idx = torch.randint(0, train_x.shape[0], (a.batch_size,), generator=gen)
            return H.normalize(H.fast_augment(train_x[idx], gen)).contiguous(memory_format=mf), train_y[idx]
        for i in range(10):  # warmup (and compile, if on)
            step_fn(*batch(i))
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for i in range(30):
                step_fn(*batch(i))
        table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        print(table)
        with open(f"profile{'_' + a.tag if a.tag else ''}.txt", "w") as f:
            f.write(table)
        return

    def train_one_epoch(epoch):
        perm = torch.randperm(train_x.shape[0], generator=gen)
        total = torch.zeros(())
        n = 0
        for i in range(0, perm.numel(), a.batch_size):
            idx = perm[i:i + a.batch_size]
            x = H.normalize(H.fast_augment(train_x[idx], gen)).contiguous(memory_format=mf)
            total += step_fn(x, train_y[idx])
            if sched.last_epoch < horizon - 1:  # OneCycleLR errors if stepped past its horizon
                sched.step()
            n += 1
        return (total / n).item()  # one sync per epoch, not per step

    H.run(a.name, train_one_epoch, model, test_x, test_y, a.max_epochs, a.target_acc, a.seed,
          eval_memory_format=mf, out_dir=a.out_dir, extra={k: v for k, v in vars(a).items() if k not in ("data",)})


if __name__ == "__main__":
    main()
