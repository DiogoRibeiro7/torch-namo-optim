from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def update_pyproject_version(new_version: str) -> None:
    text = _read(PYPROJECT)
    updated, n = re.subn(
        r'(?m)^version\s*=\s*"[0-9A-Za-z.\-+]+"\s*$',
        f'version = "{new_version}"',
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not update version in pyproject.toml")
    _write(PYPROJECT, updated)


def update_changelog(new_version: str) -> None:
    text = _read(CHANGELOG)
    today = date.today().isoformat()
    marker = "## [Unreleased]"
    if marker not in text:
        raise RuntimeError("CHANGELOG.md is missing '## [Unreleased]' section")

    insertion = f"{marker}\n\n## [{new_version}] - {today}"
    updated = text.replace(marker, insertion, 1)
    _write(CHANGELOG, updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump package version and roll changelog.")
    parser.add_argument("version", help="New version (for example: 0.1.1)")
    args = parser.parse_args()

    update_pyproject_version(args.version)
    update_changelog(args.version)
    print(f"Updated pyproject.toml and CHANGELOG.md to version {args.version}")


if __name__ == "__main__":
    main()
