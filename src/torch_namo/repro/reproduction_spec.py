from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelSize = Literal["124m", "355m"]
OptimizerName = Literal["adamw", "muon", "namo", "namod"]
RunStage = Literal["final", "sweep"]


@dataclass(frozen=True)
class RunSpec:
    stage: RunStage
    model: ModelSize
    optimizer: OptimizerName
    lr: float
    steps: int
    c: float | None = None


SHARED: dict[str, float | int] = {
    "context_length": 1024,
    "warmup_steps": 2000,
    "mu1": 0.95,
    "mu2": 0.99,
    "weight_decay": 0.01,
}

MODEL_CONFIG: dict[ModelSize, dict[str, int]] = {
    "124m": {
        "micro_batch_size": 60,
        "grad_accum_steps": 8,
        "sweep_steps": 10_000,
        "final_steps": 50_000,
    },
    "355m": {
        "micro_batch_size": 40,
        "grad_accum_steps": 12,
        "sweep_steps": 10_000,
        "final_steps": 10_000,
    },
}

BEST_TABLE_1: dict[ModelSize, dict[OptimizerName, dict[str, float]]] = {
    "124m": {
        "adamw": {"lr": 0.0013},
        "muon": {"lr": 0.0013},
        "namo": {"lr": 0.012},
        "namod": {"lr": 0.009, "c": 0.1},
    },
    "355m": {
        "adamw": {"lr": 0.0009},
        "muon": {"lr": 0.0009},
        "namo": {"lr": 0.007},
        "namod": {"lr": 0.009, "c": 0.9},
    },
}

SWEEP_355M_ADAMW_MUON_LR = [0.0006, 0.0009, 0.0013, 0.0018, 0.0025]
SWEEP_355M_NAMO_NAMOD_LR = [0.005, 0.007, 0.009, 0.012, 0.015]
SWEEP_355M_NAMOD_C = [0.12, 0.4, 0.75, 0.9]


def build_runs(*, mode: Literal["final", "sweep", "all"], models: set[ModelSize]) -> list[RunSpec]:
    runs: list[RunSpec] = []

    if mode in {"final", "all"}:
        for model in sorted(models):
            for optimizer, cfg in BEST_TABLE_1[model].items():
                runs.append(
                    RunSpec(
                        stage="final",
                        model=model,
                        optimizer=optimizer,  # type: ignore[arg-type]
                        lr=cfg["lr"],
                        c=cfg.get("c"),
                        steps=MODEL_CONFIG[model]["final_steps"],
                    )
                )

    if mode in {"sweep", "all"} and "355m" in models:
        for optimizer in ("adamw", "muon"):
            for lr in SWEEP_355M_ADAMW_MUON_LR:
                runs.append(
                    RunSpec(
                        stage="sweep",
                        model="355m",
                        optimizer=optimizer,
                        lr=lr,
                        steps=MODEL_CONFIG["355m"]["sweep_steps"],
                    )
                )

        for lr in SWEEP_355M_NAMO_NAMOD_LR:
            runs.append(
                RunSpec(
                    stage="sweep",
                    model="355m",
                    optimizer="namo",
                    lr=lr,
                    steps=MODEL_CONFIG["355m"]["sweep_steps"],
                )
            )
            for c in SWEEP_355M_NAMOD_C:
                runs.append(
                    RunSpec(
                        stage="sweep",
                        model="355m",
                        optimizer="namod",
                        lr=lr,
                        c=c,
                        steps=MODEL_CONFIG["355m"]["sweep_steps"],
                    )
                )

    return runs


def run_name(run: RunSpec) -> str:
    return (
        f"{run.stage}-{run.model}-{run.optimizer}-lr{run.lr}"
        if run.c is None
        else f"{run.stage}-{run.model}-{run.optimizer}-lr{run.lr}-c{run.c}"
    )


def run_args(run: RunSpec) -> list[str]:
    model_cfg = MODEL_CONFIG[run.model]
    args = [
        "--model-size",
        str(run.model),
        "--optimizer",
        str(run.optimizer),
        "--learning-rate",
        str(run.lr),
        "--weight-decay",
        str(SHARED["weight_decay"]),
        "--context-length",
        str(SHARED["context_length"]),
        "--micro-batch-size",
        str(model_cfg["micro_batch_size"]),
        "--grad-accum-steps",
        str(model_cfg["grad_accum_steps"]),
        "--warmup-steps",
        str(SHARED["warmup_steps"]),
        "--max-steps",
        str(run.steps),
        "--run-name",
        run_name(run),
    ]

    if run.optimizer in {"namo", "namod"}:
        args.extend(["--mu1", str(SHARED["mu1"]), "--mu2", str(SHARED["mu2"])])
    if run.optimizer == "namod" and run.c is not None:
        args.extend(["--namo-d-c", str(run.c)])

    return args


def format_command(run: RunSpec, *, train_entry: str = "python train.py") -> str:
    return " ".join([train_entry, *run_args(run)])
