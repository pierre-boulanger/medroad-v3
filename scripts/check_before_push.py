#!/usr/bin/env python3
"""
Pre-push safety check.

Run before the first push and before any release. It refuses nothing and
changes nothing; it reports what a reviewer, or a data use agreement auditor,
would find in the repository.

Two classes of problem matter here. Credentials leak quietly and are painful to
revoke once public. Credentialed clinical data must not be redistributed at
all, and a single committed CSV is a data use agreement breach rather than an
inconvenience.

    python scripts/check_before_push.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files that must never be tracked, whatever .gitignore says.
FORBIDDEN_NAMES = {".env"}
FORBIDDEN_SUFFIXES = {".csv", ".gz", ".parquet", ".pt", ".pkl", ".joblib", ".ubj"}
ALLOWED_PATHS = {"tests/fixtures"}

# Credential shapes. Deliberately narrow: broad patterns produce noise that
# gets ignored, which defeats the purpose.
SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "Anthropic API key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), "GitHub personal access token"),
    (re.compile(r"(?i)\b(client_secret|api[_-]?key|password)\s*[=:]\s*"
                r"['\"]?(?!<|\$|\{|change-me|YOUR_|Admin1234!|root\b|openemr\b)"
                r"[A-Za-z0-9/+_\-]{12,}"), "hardcoded credential"),
]

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             ".pytest_cache", ".ruff_cache", "results", "models_saved"}


def _read(path: Path) -> str:
    """
    Read text as UTF-8 regardless of platform.

    Python on Windows defaults to the system code page, usually cp1252, which
    raises UnicodeDecodeError on any file containing a dash, arrow or accented
    character. Every text read in this project states its encoding for that
    reason.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def tracked_files() -> list[Path] | None:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [ROOT / line for line in out.stdout.splitlines() if line]


def walk_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def main() -> int:
    files = tracked_files()
    mode = "tracked by git" if files is not None else "present on disk"
    if files is None:
        print("note: not a git repository yet, checking files on disk\n")
        files = walk_files()

    problems: list[str] = []

    # 1. forbidden file types
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        if any(rel.startswith(a) for a in ALLOWED_PATHS):
            continue
        if f.name in FORBIDDEN_NAMES:
            problems.append(f"{rel}: credentials file")
        elif f.suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"{rel}: data or model artefact ({f.suffix})")

    # 2. secrets in text
    for f in files:
        if f.suffix in FORBIDDEN_SUFFIXES or not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = f.relative_to(ROOT).as_posix()
        if rel.endswith("check_before_push.py"):
            continue            # this file contains the patterns themselves
        for pat, label in SECRET_PATTERNS:
            for m in pat.finditer(text):
                line = text[:m.start()].count("\n") + 1
                problems.append(f"{rel}:{line}: possible {label}")

    # 3. placeholders that should be filled before publication
    reminders = []
    cff = ROOT / "CITATION.cff"
    if cff.exists() and "0000-0000-0000-0000" in _read(cff):
        reminders.append("CITATION.cff: ORCID placeholder")
    for name in ("pyproject.toml", "README.md", "CITATION.cff"):
        p = ROOT / name
        if p.exists() and "pboulanger/medroad-v3" in _read(p):
            reminders.append(f"{name}: check the repository URL is correct")
    lic = ROOT / "LICENSE"
    if lic.exists() and "MIT License" in _read(lic):
        reminders.append("LICENSE: MIT — confirm it is compatible with "
                         "institutional IP policy")

    print(f"checked {len(files)} files ({mode})\n")

    if problems:
        print("BLOCKING — do not push until these are resolved:")
        for p in sorted(set(problems)):
            print(f"  {p}")
        print()
    else:
        print("no credentials or clinical data found\n")

    if reminders:
        print("Before publishing, confirm:")
        for r in sorted(set(reminders)):
            print(f"  {r}")
        print()

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
