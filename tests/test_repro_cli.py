from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _write_dummy_trainer(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "import sys",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--marker-file', required=True)",
                "parser.add_argument('--fail', action='store_true')",
                "args, rest = parser.parse_known_args()",
                "Path(args.marker_file).write_text('\\n'.join(rest), encoding='utf-8')",
                "if args.fail:",
                "    sys.exit(3)",
                "sys.exit(0)",
            ]
        ),
        encoding="utf-8",
    )


def _run_cli(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    cmd = [sys.executable, "-m", "torch_namo.repro.cli", *args]
    return subprocess.run(cmd, cwd=repo_root, env=env, text=True, capture_output=True, check=False)


def test_repro_cli_execute_runs_with_dummy_trainer(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    trainer = tmp_path / "dummy_train.py"
    marker = tmp_path / "marker.txt"
    _write_dummy_trainer(trainer)

    train_entry = f'"{sys.executable}" "{trainer}" --marker-file "{marker}"'
    proc = _run_cli(
        repo_root,
        "--execute",
        "--mode",
        "final",
        "--model",
        "124m",
        "--limit",
        "1",
        "--train-entry",
        train_entry,
    )

    assert proc.returncode == 0, proc.stderr
    assert marker.exists()
    payload = marker.read_text(encoding="utf-8")
    assert "--optimizer" in payload
    assert "--run-name" in payload


def test_repro_cli_execute_propagates_trainer_failure(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    trainer = tmp_path / "dummy_train.py"
    marker = tmp_path / "marker.txt"
    _write_dummy_trainer(trainer)

    train_entry = f'"{sys.executable}" "{trainer}" --marker-file "{marker}" --fail'
    proc = _run_cli(
        repo_root,
        "--execute",
        "--mode",
        "final",
        "--model",
        "124m",
        "--limit",
        "1",
        "--train-entry",
        train_entry,
    )

    assert proc.returncode == 3
