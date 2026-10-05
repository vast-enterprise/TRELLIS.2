#!/usr/bin/env python3
"""Export an existing VXZM sample as a colored/normal PLY point cloud.

Examples:
    python visualize_vxzm.py sample.vxzm --output sample.ply
    python visualize_vxzm.py sample.vxzm --output preview.ply --max-points 500000

The command reads VXZM directly; it does not download a GLB or rerun Blender
voxelization.  Duplicate XYZ records are retained, and ``--normal-offset``
only separates coincident records in the visualization.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
# Keep this script usable from a source checkout where o-voxel has not been
# installed as a site package yet.
sys.path.insert(0, str(ROOT / "o-voxel"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Input .vxzm file")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output .ply point cloud")
    parser.add_argument("--normal-offset", type=float, default=0.08,
                        help="Offset along decoded normal in fine-voxel units (default: 0.08)")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Deterministically retain at most this many records")
    parser.add_argument("--multi-sample-only", action="store_true",
                        help="Keep only voxels containing at least --min-samples records")
    parser.add_argument("--min-samples", type=int, default=2,
                        help="Multiplicity threshold for --multi-sample-only (default: 2)")
    parser.add_argument("--aabb", type=float, nargs=6,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                        default=(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5),
                        help="World-space bounds, default [-.5,.5]^3")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.input.suffix.lower() != ".vxzm" or not args.input.is_file():
        raise ValueError(f"input must be an existing .vxzm file: {args.input}")
    if args.normal_offset < 0:
        raise ValueError("--normal-offset must be non-negative")
    if args.max_points is not None and args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    if args.min_samples < 2:
        raise ValueError("--min-samples must be at least 2")
    bounds = (args.aabb[:3], args.aabb[3:])
    if any(hi <= lo for lo, hi in zip(bounds[0], bounds[1])):
        raise ValueError("--aabb max values must be greater than min values")

    import o_voxel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    coord, attr = o_voxel.io.vxzm_to_ply(
        args.input, args.output, aabb=bounds,
        normal_offset=args.normal_offset, max_points=args.max_points,
        multi_sample_only=args.multi_sample_only, min_samples=args.min_samples,
    )
    selection = " multi-sample" if args.multi_sample_only else ""
    print(f"Wrote {args.output} ({len(coord)}{selection} records, RGB + normals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
