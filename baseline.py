"""Baseline (shared by both tracks): the 'already tuned' job, written the way most people would write it.
Freeze this file once you have agreed the recipe. Run: python baseline.py --seed 0"""
import argparse

import torch
import torch.nn as nn

import bench_harness as H


def main():
    p = H.common_args(argparse.ArgumentParser())
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    H.configure_threads(a.threads)
    H.set_seed(a.seed)
    train_x, train_y, test_x, test_y = H.load_cifar10_tensors(a.data, a.synthetic, a.source)
    loader = H.standard_train_loader(train_x, train_y, a.batch_size, a.workers, a.seed)

    model = H.make_model(a.model)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    model.train()

    def train_one_epoch(epoch):
        total, n = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        return total / n

    H.run("baseline", train_one_epoch, model, test_x, test_y, a.max_epochs, a.target_acc, a.seed,
          out_dir=a.out_dir, extra={"batch_size": a.batch_size, "lr": a.lr, "workers": a.workers, "tag": a.tag, "model": a.model})


if __name__ == "__main__":
    main()
