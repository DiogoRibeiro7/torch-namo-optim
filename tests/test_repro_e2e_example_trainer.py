from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_repro_cli_executes_example_trainer_and_writes_logs(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = tmp_path / "logs"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch_namo.repro.cli",
            "--execute",
            "--mode",
            "final",
            "--model",
            "124m",
            "--limit",
            "1",
            "--max-steps-override",
            "3",
            "--train-entry",
            f'"{sys.executable}" "experiments/paper/example_trainer.py" --log-dir "{log_dir}"',
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    logs = list(log_dir.glob("*.csv"))
    assert logs
    content = logs[0].read_text(encoding="utf-8")
    assert "run_name,model_size,optimizer,split,step,loss" in content
