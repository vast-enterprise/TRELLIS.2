#!/usr/bin/env bash
# Set up the dependencies required by the Blender GLB -> O-Voxel test path.
#
# Run from the repository root in a Debian/Ubuntu CUDA Pod:
#   bash set_ovoxel.sh
#
# This script intentionally uses the same commands that were used for the
# successful Pod setup: install Eigen, Blender, Pillow and OpenColorIO, then
# build the extension with Torch's CUDA extension builder.
#
# Optional environment variables:
#   CUDA_HOME=/usr/local/cuda   CUDA toolkit used by torch's extension builder
#   MAX_JOBS=2                 parallel extension compiler jobs
#   PYTHON_BIN=python           Python interpreter paired with Torch
#   SKIP_APT=1                 do not install Debian packages
#   SKIP_BUILD=1               do not rebuild o-voxel

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVOXEL_ROOT="$REPO_ROOT/o-voxel"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MAX_JOBS="${MAX_JOBS:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$OVOXEL_ROOT/setup.py" ]]; then
    echo "error: expected o-voxel/setup.py below $REPO_ROOT" >&2
    exit 1
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "error: nvcc not found at $CUDA_HOME/bin/nvcc; set CUDA_HOME explicitly" >&2
    exit 1
fi

if [[ "${SKIP_APT:-0}" != "1" ]]; then
    if [[ "$(id -u)" -ne 0 ]]; then
        echo "error: package setup needs root. Re-run as root or use SKIP_APT=1 after installing:" >&2
        echo "       libeigen3-dev blender python3-pil python3-pyopencolorio" >&2
        exit 1
    fi
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        libeigen3-dev \
        blender \
        python3-pil \
        python3-pyopencolorio
fi

test -f /usr/include/eigen3/Eigen/Dense || {
    echo "error: Eigen headers missing (/usr/include/eigen3/Eigen/Dense)" >&2
    exit 1
}
command -v blender >/dev/null || {
    echo "error: Blender is missing" >&2
    exit 1
}

# Blender's Python uses the system site-packages in the tested Pod.  Verify
# Pillow there as well; the package is needed by dump_pbr.py.
blender -b --python-expr 'import PIL; print("Blender Pillow:", PIL.__file__)' >/dev/null

"$PYTHON_BIN" - <<'PY'
import torch
try:
    import PyOpenColorIO as ocio
except ImportError as e:
    raise SystemExit("error: PyOpenColorIO is missing") from e
try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit("error: Open3D is missing (required for PLY visualization)") from e
print(f"Python/Torch: {torch.__version__}, CUDA={torch.version.cuda}, GPU={torch.cuda.is_available()}")
print(f"PyOpenColorIO: {ocio.__version__}")
print(f"Open3D: {o3d.__version__}")
PY
blender --version | head -n 1

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    (
        cd "$OVOXEL_ROOT"
        export CUDA_HOME MAX_JOBS
        "$PYTHON_BIN" setup.py build_ext --inplace
    )
fi

PYTHONPATH="$OVOXEL_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import o_voxel
print("o_voxel extension import: OK")
PY

echo "Environment ready. Example:"
echo "  PYTHONPATH=o-voxel:. python tests/test_glb_to_vxz.py --glb testmesh/<model>.glb --output-dir /tmp/trellis2_glb_test --blender blender --resolution 64 --color-space agx"
