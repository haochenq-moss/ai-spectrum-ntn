"""Compile agent workflows with the public Nemo CLI."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def compile_workflow(workflow_file: str, output_file: str, optimize: str = "aggressive") -> str:
    """Validate and lower a Nemo workflow, returning the YAML IR artifact path.

    Nemo's public compiler currently exposes IR via ``--dump-ir``; optimization
    is retained as caller metadata until the compiler exposes optimization flags.
    """
    del optimize
    nemo = shutil.which("nemo") or str(Path.home() / ".local" / "bin" / "nemo")
    if not Path(nemo).is_file():
        nemo = None
    if nemo is None:
        raise RuntimeError("Nemo compiler is unavailable. Install the `nemo` CLI and add it to PATH before compiling workflows.")
    workflow_path, output_path = Path(workflow_file), Path(output_file)
    if not workflow_path.is_file():
        raise FileNotFoundError(f"Workflow file does not exist: {workflow_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([nemo, "compile", str(workflow_path), "--dump-ir"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    output_path.write_text(result.stdout)
    return str(output_path)