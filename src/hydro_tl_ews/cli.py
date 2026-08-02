"""Console entry point for ``hydro-tl-ews`` (installed via pyproject.toml)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Load ``scripts/run_experiment.py`` and call its ``main``."""
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "run_experiment.py"
    if not script.is_file():
        raise SystemExit(
            f"Cannot find {script}. Run from a source checkout of hydro-tl-ews."
        )
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("hydro_tl_ews_run_experiment", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
