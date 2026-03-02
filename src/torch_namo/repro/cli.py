from __future__ import annotations

import argparse
from pathlib import Path

from .reproduction_spec import build_runs, format_command


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
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    models = {"124m", "355m"} if args.model == "all" else {args.model}
    runs = build_runs(mode=args.mode, models=models)
    commands = [format_command(run, train_entry=args.train_entry) for run in runs]

    if args.print_only:
        for cmd in commands:
            print(cmd)
        print(f"\n# total_commands={len(commands)}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {args.out}")


if __name__ == "__main__":
    main()
