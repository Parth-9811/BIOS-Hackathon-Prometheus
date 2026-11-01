# Results

One JSON file per training run, named `<name>[_<tag>]_seed<N>.json`. Each file records the time and CPU energy to reach the accuracy target, test accuracy after every epoch, every command-line setting, and the environment (PyTorch version, CPU, thread count). Summarise any folder with `python summarize.py <folder>`.

| Folder | Contents |
|---|---|
| `track_a/` | ResNet-18, Track A proof runs: 3 baseline + 3 accelerated seeds, `summary.md` |
| `track_a/explore_205407/` | ResNet-18, Track A exploration runs used to choose batch size, threads, precision and epoch count |
| `track_a/profiles/` | PyTorch profiler output over 30 training steps, eager and compiled |
| `track_b/` | ResNet-18, Track B proof runs: 3 baseline + 3 systems-only seeds, `summary.md` |
| `track_b_sweep/` | ResNet-18, Track B 2-epoch timing sweep (11 configurations) |
| `shufflenet_v2_x1_0/track_a/` | ShuffleNetV2 1.0, Track A proof runs: 3 baseline + 3 accelerated seeds, `summary.md` |
| `shufflenet_v2_x1_0/tune/` | ShuffleNetV2 1.0, Track A runs used to choose precision and epoch count |
| `shufflenet_v2_x1_0/track_b_sweep/` | ShuffleNetV2 1.0, Track B 2-epoch timing check |
| `shufflenet_v2_x1_0/ceiling/` | ShuffleNetV2 1.0, 30-epoch baseline run, reference for the 72% target |
| `squeezenet1_1/ceiling/` | SqueezeNet 1.1, 30-epoch baseline run (did not learn: 10% accuracy), so not used |

Key fields in each file:

| Field | Meaning |
|---|---|
| `time_to_target_s` | Training time (evaluation excluded) until test accuracy first reached `target_acc`; `null` if never reached |
| `energy_to_target_j` | CPU package energy over the same window, from RAPL; `null` if RAPL was not readable |
| `reached` | Whether the target was reached within `max_epochs` |
| `history` | Per epoch: cumulative training time, loss, test accuracy, cumulative energy |
| `env` | PyTorch and Python versions, CPU model, thread count, relevant environment variables |
