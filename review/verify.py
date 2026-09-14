"""Validate the current version. Historical audit outputs are in verification_results.json."""

from pathlib import Path
import os
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=root, env=env, check=True
    )
    subprocess.run(
        [sys.executable, "main.py", "--n-files", "18", "--seed", "42"],
        cwd=root,
        env=env,
        check=True,
    )
