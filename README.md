# torch-namo-optim

Production-ready PyTorch implementations of **NAMO** and **NAMO-D** optimizers (orthogonalized momentum).

These optimizers are designed for **matrix parameters** (2D tensors), e.g. linear/attention weights.
A common setup is:

- **NAMO / NAMO-D** for 2D weight matrices
- **AdamW** for everything else (biases, LayerNorm scales, embeddings, etc.)

## Install

```bash
poetry add torch-namo-optim
```

Or in editable mode:

```bash
git clone <your-repo>
cd torch-namo-optim
poetry install
```

## Quick start

```python
import torch
from torch_namo import NAMOD, split_params, RoutedOptimizer

model = ...
mat_params, other_params = split_params(model)

opt = RoutedOptimizer(
    matrix_params=mat_params,
    other_params=other_params,
    matrix_opt_cls=NAMOD,
    matrix_opt_kwargs=dict(
        lr=3e-4,
        mu1=0.95,
        mu2=0.99,
        eps=1e-8,
        c=1.0,
        weight_decay=0.01,
        orth_method="newton_schulz",
        ns_iters=5,
        stable_dtype=torch.float32,
    ),
    other_opt_cls=torch.optim.AdamW,
    other_opt_kwargs=dict(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
)

for batch in loader:
    opt.zero_grad(set_to_none=True)
    loss = ...
    loss.backward()
    opt.step()
```

## AMP (mixed precision)

`RoutedOptimizer` is a small wrapper, not a `torch.optim.Optimizer`.
For AMP, call `scaler.step` on each internal optimizer:

```python
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss = ...

    scaler.scale(loss).backward()
    scaler.step(opt.matrix_opt)
    scaler.step(opt.other_opt)
    scaler.update()
```

## Notes

- Orthogonalization supports:
  - `newton_schulz` (default): faster, good for large matrices
  - `svd`: exact but slower
- Internal computations use `stable_dtype` (default `float32`) for numerical stability.
- The implementation enforces `0 <= mu1 <= mu2 < 1`, matching the paper assumptions.

## Paper Consistency Status

Current repo status toward the article:

- Core NAMO and NAMO-D update equations are implemented.
- Tests include one-step consistency checks against the paper equations.
- Paper hyperparameters and command generation are included in `experiments/paper/`.

## Reproduce Commands

Generate paper-aligned command grids:

```bash
python -m torch_namo.repro.cli --mode all --model all --print-only
```

Write commands to `experiments/paper/generated/commands.txt`:

```bash
python -m torch_namo.repro.cli --mode all --model all
```

See `experiments/paper/README.md` for details and scope notes.

Execute runs directly against a trainer script/repo:

```bash
python -m torch_namo.repro.cli \
  --execute \
  --mode final \
  --model 124m \
  --workdir /path/to/trainer-repo \
  --train-entry "python train.py"
```

Useful execution controls:
- `--limit N` to run only the first `N` jobs
- `--start-index K` to skip the first `K` jobs
- `--continue-on-error` to keep running after failures
- `--max-steps-override S` to force short smoke runs

Generate paper artifacts (tables + loss curves) from trainer logs:

```bash
python -m torch_namo.repro.artifacts \
  --inputs "/path/to/logs/**/*.csv" "/path/to/logs/**/*.jsonl" \
  --out-dir experiments/paper/artifacts
```

## License

MIT.

## Release

See `RELEASING.md` for the version bump, changelog, and tag-based publish flow.
