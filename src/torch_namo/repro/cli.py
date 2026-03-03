from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from .reproduction_spec import build_runs, format_command, run_args


def _split_train_entry(train_entry: str) -> list[str]:
    return shlex.split(train_entry, posix=True)


def _execute_runs(
    *,
    commands: list[list[str]],
    workdir: Path,
    continue_on_error: bool,
) -> int:
    failures: list[int] = []
    total = len(commands)

    for idx, cmd in enumerate(commands, start=1):
        print(f"[{idx}/{total}] Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=workdir, check=False)
        if proc.returncode != 0:
            failures.append(idx)
            print(f"[{idx}/{total}] Failed with exit code {proc.returncode}")
            if not continue_on_error:
                return proc.returncode

    if failures:
        print(f"Completed with failures in runs: {failures}")
        return 1
    print("All runs completed successfully.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper reproduction commands.")
    parser.add_argument("--mode", choices=["final", "sweep", "all"], default="all")
    parser.add_argument("--model", choices=["124m", "355m", "all"], default="all")
    parser.add_argument("--train-entry", default="python train.py")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/paper/generated/commands.txt"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    models = {"124m", "355m"} if args.model == "all" else {args.model}
    runs = build_runs(mode=args.mode, models=models)
    selected = runs[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    commands = [
        format_command(
            run,
            train_entry=args.train_entry,
            max_steps_override=args.max_steps_override,
        )
        for run in selected
    ]

    if args.print_only:
        for cmd in commands:
            print(cmd)
        print(f"\n# total_commands={len(commands)}")
        return

    if args.execute:
        train_entry_tokens = _split_train_entry(args.train_entry)
        exec_commands = [
            train_entry_tokens + run_args(run, max_steps_override=args.max_steps_override)
            for run in selected
        ]
        exit_code = _execute_runs(
            commands=exec_commands,
            workdir=args.workdir,
            continue_on_error=args.continue_on_error,
        )
        raise SystemExit(exit_code)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
