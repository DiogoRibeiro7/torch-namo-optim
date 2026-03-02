# Paper Reproduction

This folder contains outputs and docs for reproducing the experiments described in
`paper/2602.17080v2.pdf`.

## Implementation Location

The reproducibility implementation is package-style and lives in:

- `src/torch_namo/repro/reproduction_spec.py`
- `src/torch_namo/repro/cli.py`

## What Is Explicitly Encoded

- Shared settings:
  - `context_length=1024`
  - effective batch size `480` sequences
  - warmup `2000` steps
  - NAMO/NAMO-D: `mu1=0.95`, `mu2=0.99`, `weight_decay=0.01`
- GPT-2 (124M): micro-batch `60`, grad accumulation `8`, final run `50K` steps
- GPT-2 (355M): micro-batch `40`, grad accumulation `12`, final run `10K` steps
- 355M sweep grids:
  - AdamW/Muon LR: `{0.0006, 0.0009, 0.0013, 0.0018, 0.0025}`
  - NAMO/NAMO-D LR: `{0.005, 0.007, 0.009, 0.012, 0.015}`
  - NAMO-D c: `{0.12, 0.40, 0.75, 0.90}`
- Table 1 best hyperparameters for 124M and 355M final runs.

## Note on 124M LR Sweep

The paper text states a 124M LR sweep was performed, but it does not list the exact LR grid in
plain text. This repo encodes Table 1 best values for 124M final runs and explicit 355M sweep grids.

## Usage

Print commands without writing files:

`python -m torch_namo.repro.cli --mode all --model all --print-only`

Write final-run commands to a file:

`python -m torch_namo.repro.cli --mode final --model all`

Generate only explicit paper sweeps for 355M:

`python -m torch_namo.repro.cli --mode sweep --model 355m`

If your trainer entrypoint is not `python train.py`, set:

`python -m torch_namo.repro.cli --train-entry "python path/to/your_train.py"`
