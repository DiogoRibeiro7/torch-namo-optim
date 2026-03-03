from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from torch_namo.repro.artifacts import generate_artifacts


def test_generate_artifacts_from_csv_and_jsonl(tmp_path: Path):
    csv_path = tmp_path / "run_a.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run_name", "model_size", "optimizer", "split", "step", "loss"],
        )
        w.writeheader()
        w.writerows(
            [
                {
                    "run_name": "final-124m-adamw-lr0.0013",
                    "model_size": "124m",
                    "optimizer": "adamw",
                    "split": "train",
                    "step": 1,
                    "loss": 4.0,
                },
                {
                    "run_name": "final-124m-adamw-lr0.0013",
                    "model_size": "124m",
                    "optimizer": "adamw",
                    "split": "val",
                    "step": 1,
                    "loss": 4.2,
                },
                {
                    "run_name": "final-124m-adamw-lr0.0013",
                    "model_size": "124m",
                    "optimizer": "adamw",
                    "split": "train",
                    "step": 2,
                    "loss": 3.8,
                },
            ]
        )

    jsonl_path = tmp_path / "run_b.jsonl"
    lines = [
        {
            "run_name": "final-124m-namo-lr0.012",
            "model_size": "124m",
            "optimizer": "namo",
            "split": "train",
            "step": 1,
            "loss": 3.5,
        },
        {
            "run_name": "final-124m-namo-lr0.012",
            "model_size": "124m",
            "optimizer": "namo",
            "split": "val",
            "step": 1,
            "loss": 3.7,
        },
        {
            "run_name": "final-124m-namo-lr0.012",
            "model_size": "124m",
            "optimizer": "namo",
            "split": "val",
            "step": 2,
            "loss": 3.6,
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    out_dir = tmp_path / "artifacts"
    generate_artifacts([csv_path, jsonl_path], out_dir)

    assert (out_dir / "tables" / "loss_curves.csv").exists()
    assert (out_dir / "tables" / "final_metrics.csv").exists()
    assert (out_dir / "tables" / "final_metrics_best.csv").exists()
    assert (out_dir / "tables" / "final_metrics_best.tex").exists()

    content = (out_dir / "tables" / "final_metrics.csv").read_text(encoding="utf-8")
    assert "final-124m-adamw-lr0.0013" in content
    assert "final-124m-namo-lr0.012" in content


def test_artifacts_cli_runs(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = tmp_path / "single.csv"
    input_csv.write_text(
        "run_name,model_size,optimizer,split,step,loss\n"
        "final-355m-muon-lr0.0009,355m,muon,train,1,3.0\n"
        "final-355m-muon-lr0.0009,355m,muon,val,1,3.2\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch_namo.repro.artifacts",
            "--inputs",
            str(input_csv),
            "--out-dir",
            str(out_dir),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "tables" / "final_metrics_best.csv").exists()
