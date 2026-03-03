from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class LossRow:
    run_name: str
    model_size: str
    optimizer: str
    split: str
    step: int
    loss: float


def _coalesce(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _normalize_record(rec: dict[str, Any]) -> LossRow | None:
    run_name = _coalesce(rec, "run_name", "run", default="")
    model_size = str(_coalesce(rec, "model_size", "model", default="")).lower()
    optimizer = str(_coalesce(rec, "optimizer", "opt", default="")).lower()
    split = str(_coalesce(rec, "split", default="")).lower()
    step_raw = _coalesce(rec, "step", "global_step", default=None)
    loss_raw = _coalesce(rec, "loss", "value", default=None)

    if not run_name or not model_size or not optimizer or split not in {"train", "val"}:
        return None
    if step_raw is None or loss_raw is None:
        return None

    try:
        step = int(float(step_raw))
        loss = float(loss_raw)
    except (TypeError, ValueError):
        return None

    return LossRow(
        run_name=str(run_name),
        model_size=model_size,
        optimizer=optimizer,
        split=split,
        step=step,
        loss=loss,
    )


def load_logs(paths: Iterable[Path]) -> list[LossRow]:
    rows: list[LossRow] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                for rec in csv.DictReader(f):
                    row = _normalize_record(rec)
                    if row is not None:
                        rows.append(row)
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        row = _normalize_record(rec)
                        if row is not None:
                            rows.append(row)
    return rows


def _final_metrics(rows: list[LossRow]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], LossRow] = {}
    for row in rows:
        key = (row.run_name, row.model_size, row.optimizer, row.split)
        prev = latest.get(key)
        if prev is None or row.step > prev.step:
            latest[key] = row

    per_run: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (_, model_size, optimizer, split), row in latest.items():
        key = (row.run_name, model_size, optimizer)
        item = per_run.setdefault(
            key,
            {
                "run_name": row.run_name,
                "model_size": model_size,
                "optimizer": optimizer,
                "train_step": None,
                "train_loss": None,
                "val_step": None,
                "val_loss": None,
            },
        )
        if split == "train":
            item["train_step"] = row.step
            item["train_loss"] = row.loss
        else:
            item["val_step"] = row.step
            item["val_loss"] = row.loss

    return sorted(per_run.values(), key=lambda x: (x["model_size"], x["optimizer"], x["run_name"]))


def _best_by_optimizer(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in final_rows:
        grouped.setdefault((row["model_size"], row["optimizer"]), []).append(row)

    best: list[dict[str, Any]] = []
    for key, items in grouped.items():
        with_val = [x for x in items if x["val_loss"] is not None]
        if with_val:
            pick = min(with_val, key=lambda x: float(x["val_loss"]))
        else:
            pick = min(items, key=lambda x: float(x["train_loss"]) if x["train_loss"] is not None else float("inf"))
        best.append(dict(pick))
    return sorted(best, key=lambda x: (x["model_size"], x["optimizer"]))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llllrr}",
        "\\toprule",
        "Model & Optimizer & Run & Train step & Train loss & Val loss \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model_size']} & {row['optimizer']} & {row['run_name']} & "
            f"{row['train_step'] if row['train_step'] is not None else '-'} & "
            f"{row['train_loss'] if row['train_loss'] is not None else '-'} & "
            f"{row['val_loss'] if row['val_loss'] is not None else '-'} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_curves(rows: list[LossRow], best_rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure generation")
        return

    best_names = {r["run_name"] for r in best_rows}
    selected = [r for r in rows if r.run_name in best_names]
    if not selected:
        return

    by_model_split: dict[tuple[str, str], list[LossRow]] = {}
    for row in selected:
        by_model_split.setdefault((row.model_size, row.split), []).append(row)

    for (model_size, split), subset in by_model_split.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        by_opt: dict[str, list[LossRow]] = {}
        for row in subset:
            by_opt.setdefault(row.optimizer, []).append(row)
        for optimizer, vals in sorted(by_opt.items()):
            vals = sorted(vals, key=lambda x: x.step)
            ax.plot([x.step for x in vals], [x.loss for x in vals], label=optimizer)
        ax.set_title(f"{model_size} {split} loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(alpha=0.3)
        out_path = out_dir / f"loss_curve_{model_size}_{split}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def generate_artifacts(inputs: list[Path], out_dir: Path) -> None:
    rows = load_logs(inputs)
    if not rows:
        raise ValueError("No valid rows found in input logs.")

    rows_sorted = sorted(rows, key=lambda x: (x.model_size, x.optimizer, x.run_name, x.split, x.step))
    curves_csv = out_dir / "tables" / "loss_curves.csv"
    _write_csv(
        curves_csv,
        [
            {
                "run_name": r.run_name,
                "model_size": r.model_size,
                "optimizer": r.optimizer,
                "split": r.split,
                "step": r.step,
                "loss": r.loss,
            }
            for r in rows_sorted
        ],
        ["run_name", "model_size", "optimizer", "split", "step", "loss"],
    )

    final_rows = _final_metrics(rows_sorted)
    final_csv = out_dir / "tables" / "final_metrics.csv"
    _write_csv(
        final_csv,
        final_rows,
        ["run_name", "model_size", "optimizer", "train_step", "train_loss", "val_step", "val_loss"],
    )

    best_rows = _best_by_optimizer(final_rows)
    best_csv = out_dir / "tables" / "final_metrics_best.csv"
    _write_csv(
        best_csv,
        best_rows,
        ["run_name", "model_size", "optimizer", "train_step", "train_loss", "val_step", "val_loss"],
    )
    _write_latex_table(out_dir / "tables" / "final_metrics_best.tex", best_rows)
    _plot_curves(rows_sorted, best_rows, out_dir / "figures")

    print(f"Wrote: {curves_csv}")
    print(f"Wrote: {final_csv}")
    print(f"Wrote: {best_csv}")
    print(f"Wrote: {out_dir / 'tables' / 'final_metrics_best.tex'}")
    print(f"Wrote figures under: {out_dir / 'figures'}")


def _expand_inputs(inputs: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for pattern in inputs:
        p = Path(pattern)
        if any(ch in pattern for ch in ["*", "?", "["]):
            expanded.extend(sorted(Path().glob(pattern)))
        elif p.is_dir():
            expanded.extend(sorted(p.rglob("*.csv")))
            expanded.extend(sorted(p.rglob("*.jsonl")))
            expanded.extend(sorted(p.rglob("*.ndjson")))
        elif p.exists():
            expanded.append(p)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper tables/figures from trainer logs (.csv/.jsonl)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files/dirs/globs containing training logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/paper/artifacts"),
    )
    args = parser.parse_args()

    paths = _expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No input files matched.")
    generate_artifacts(paths, args.out_dir)


if __name__ == "__main__":
    main()
