from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from torch_namo import NAMO, NAMOD, RoutedOptimizer, split_params


def _build_model(model_size: str) -> torch.nn.Module:
    # Small surrogate models to validate optimizer wiring/log schema end-to-end.
    hidden = 128 if model_size == "124m" else 256
    return torch.nn.Sequential(
        torch.nn.Linear(64, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, 64),
    )


def _build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer | RoutedOptimizer:
    matrix_params, other_params = split_params(model)

    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.optimizer == "muon":
        # Placeholder route for integration tests; use NAMO with no second-moment smoothing.
        return RoutedOptimizer(
            matrix_params=matrix_params,
            other_params=other_params,
            matrix_opt_cls=NAMO,
            matrix_opt_kwargs=dict(
                lr=args.learning_rate,
                mu1=0.95,
                mu2=0.0,
                eps=1e-8,
                weight_decay=args.weight_decay,
                orth_method="newton_schulz",
                ns_iters=3,
            ),
            other_opt_cls=torch.optim.AdamW,
            other_opt_kwargs=dict(lr=args.learning_rate, weight_decay=args.weight_decay),
        )

    if args.optimizer == "namo":
        return RoutedOptimizer(
            matrix_params=matrix_params,
            other_params=other_params,
            matrix_opt_cls=NAMO,
            matrix_opt_kwargs=dict(
                lr=args.learning_rate,
                mu1=args.mu1,
                mu2=args.mu2,
                eps=1e-8,
                weight_decay=args.weight_decay,
                orth_method="newton_schulz",
                ns_iters=3,
            ),
            other_opt_cls=torch.optim.AdamW,
            other_opt_kwargs=dict(lr=args.learning_rate, weight_decay=args.weight_decay),
        )

    return RoutedOptimizer(
        matrix_params=matrix_params,
        other_params=other_params,
        matrix_opt_cls=NAMOD,
        matrix_opt_kwargs=dict(
            lr=args.learning_rate,
            mu1=args.mu1,
            mu2=args.mu2,
            eps=1e-8,
            c=args.namo_d_c,
            weight_decay=args.weight_decay,
            orth_method="newton_schulz",
            ns_iters=3,
        ),
        other_opt_cls=torch.optim.AdamW,
        other_opt_kwargs=dict(lr=args.learning_rate, weight_decay=args.weight_decay),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Example trainer for torch_namo repro integration.")
    p.add_argument("--model-size", choices=["124m", "355m"], required=True)
    p.add_argument("--optimizer", choices=["adamw", "muon", "namo", "namod"], required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--micro-batch-size", type=int, default=60)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--mu1", type=float, default=0.95)
    p.add_argument("--mu2", type=float, default=0.99)
    p.add_argument("--namo-d-c", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--log-dir", type=Path, default=Path("experiments/paper/logs"))
    return p.parse_args()


def _optimizer_zero_grad(opt: torch.optim.Optimizer | RoutedOptimizer) -> None:
    if isinstance(opt, RoutedOptimizer):
        opt.zero_grad(set_to_none=True)
    else:
        opt.zero_grad(set_to_none=True)


def _optimizer_step(opt: torch.optim.Optimizer | RoutedOptimizer) -> None:
    if isinstance(opt, RoutedOptimizer):
        opt.step()
    else:
        opt.step()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    model = _build_model(args.model_size)
    opt = _build_optimizer(model, args)
    loss_fn = torch.nn.MSELoss()

    x_train = torch.randn(args.micro_batch_size, 64)
    y_train = torch.randn(args.micro_batch_size, 64)
    x_val = torch.randn(args.micro_batch_size, 64)
    y_val = torch.randn(args.micro_batch_size, 64)

    log_path = args.log_dir / f"{args.run_name}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run_name", "model_size", "optimizer", "split", "step", "loss"],
        )
        w.writeheader()

        for step in range(1, args.max_steps + 1):
            _optimizer_zero_grad(opt)
            pred = model(x_train)
            train_loss = loss_fn(pred, y_train)
            train_loss.backward()
            _optimizer_step(opt)

            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val)

            w.writerow(
                {
                    "run_name": args.run_name,
                    "model_size": args.model_size,
                    "optimizer": args.optimizer,
                    "split": "train",
                    "step": step,
                    "loss": float(train_loss.item()),
                }
            )
            w.writerow(
                {
                    "run_name": args.run_name,
                    "model_size": args.model_size,
                    "optimizer": args.optimizer,
                    "split": "val",
                    "step": step,
                    "loss": float(val_loss.item()),
                }
            )

    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
