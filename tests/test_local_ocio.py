#!/usr/bin/env python3
"""Verify the repository-local OCIO module/config/LUT layout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OCIO_ROOT = ROOT / "ocioutils"


def main() -> None:
    required = (
        OCIO_ROOT / "color_conversion.py",
        OCIO_ROOT / "config.ocio",
        OCIO_ROOT / "luts" / "AgX_Base_sRGB.cube",
        OCIO_ROOT / "filmic" / "filmic_desat_33.cube",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing local OCIO resources: {missing}")

    code = (
        "import numpy as np; "
        "from ocioutils import agx; "
        "x=np.ones((1,1,3), dtype=np.float32); "
        "y=agx.apply(x, dst_space='AgX Base sRGB'); "
        "assert y.shape == x.shape; "
        "assert str(agx.config_path).endswith('ocioutils/config.ocio'); "
        "print(agx.config_path)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Execute outside the checkout to prove that neither config nor LUT lookup
    # relies on the caller's current working directory.
    subprocess.run([sys.executable, "-c", code], cwd="/tmp", env=env, check=True)
    print("local OCIO test passed")


if __name__ == "__main__":
    main()
