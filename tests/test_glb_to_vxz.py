#!/usr/bin/env python3
"""End-to-end GLB -> Blender PBR dump -> VXZ -> PLY smoke test.

This script is intentionally usable outside the test runner.  It invokes a
headless Blender process to produce the same pickle consumed by the normal
PBR voxelization pipeline, then writes the voxel coordinates and base colors
to PLY files for quick visual inspection.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVOXEL_ROOT = ROOT / "o-voxel"
if str(OVOXEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OVOXEL_ROOT))


def run_blender_dump(blender: str, glb: Path, dump_path: Path) -> None:
    script = ROOT / "data_toolkit" / "blender_script" / "dump_pbr.py"
    cmd = [
        blender,
        "-b",
        "-P",
        str(script),
        "--",
        "--object",
        str(glb),
        "--output_path",
        str(dump_path),
    ]
    subprocess.run(cmd, check=True)
    if not dump_path.is_file():
        error = dump_path.with_name(dump_path.name + "_error.txt")
        detail = error.read_text() if error.exists() else "no dump error file"
        raise RuntimeError(f"Blender did not produce {dump_path}: {detail}")


def _lexicographic_order(coord):
    """Return a deterministic xyz sort order without assuming VXZ ordering."""
    import numpy as np
    import torch

    xyz = coord.cpu().numpy()
    # np.lexsort uses the final key as the primary key.
    return torch.from_numpy(np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))).long()


def validate_vxz_roundtrip(vxz_path: Path, coord, attr) -> tuple:
    """Verify header, voxel coordinates, and every uint8 attribute losslessly round-trip."""
    import torch
    import o_voxel

    info = o_voxel.io.read_vxz_info(str(vxz_path))
    if info["num_voxel"] != coord.shape[0]:
        raise RuntimeError(f"VXZ header reports {info['num_voxel']} voxels, expected {coord.shape[0]}")
    expected_layout = [[name, value.shape[1]] for name, value in attr.items()]
    if info["attr"] != expected_layout:
        raise RuntimeError(f"VXZ attribute layout changed: {info['attr']} vs {expected_layout}")

    read_coord, read_attr = o_voxel.io.read_vxz(str(vxz_path))
    if read_coord.shape != coord.shape:
        raise RuntimeError(f"VXZ coordinate shape changed: {read_coord.shape} vs {coord.shape}")
    if set(read_attr) != set(attr):
        raise RuntimeError(f"VXZ attribute keys changed: {sorted(read_attr)} vs {sorted(attr)}")
    source_order = _lexicographic_order(coord)
    read_order = _lexicographic_order(read_coord)
    if not torch.equal(coord[source_order].cpu(), read_coord[read_order].cpu()):
        raise RuntimeError("VXZ coordinate values changed during round-trip")
    for name, value in attr.items():
        if value.shape != read_attr[name].shape:
            raise RuntimeError(f"VXZ attribute shape changed for {name}: {read_attr[name].shape} vs {value.shape}")
        if not torch.equal(value[source_order].cpu(), read_attr[name][read_order].cpu()):
            raise RuntimeError(f"VXZ attribute values changed for {name}")
    return read_coord, read_attr


def write_open3d_ply(path: Path, coord, color=None) -> None:
    """Write a PLY with Open3D's standard XYZ and RGB vertex properties."""
    import numpy as np
    import open3d as o3d

    points = np.asarray(coord.cpu().numpy(), dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        rgb = np.asarray(color.cpu().numpy(), dtype=np.float64) / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False,
                                    compressed=False, print_progress=False):
        raise RuntimeError(f"Open3D failed to write {path}")


def validate_color_ply(path: Path, expected_coord, expected_color) -> None:
    """Verify Open3D can read the RGB PLY and that colors round-trip."""
    import torch
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = torch.from_numpy(__import__('numpy').asarray(cloud.points))
    colors = torch.from_numpy(__import__('numpy').asarray(cloud.colors))
    expected_count = len(expected_coord)
    if len(points) != expected_count or len(colors) != expected_count:
        raise RuntimeError(f"Color PLY has {len(points)} points/{len(colors)} colors, expected {expected_count}")
    # Open3D stores colors as float RGB in [0, 1] and quantizes uchar PLY on
    # write.  Compare after converting back to uint8, allowing no more than
    # one least-significant bit of file-format rounding.
    decoded = torch.clamp(torch.round(colors * 255.0), 0, 255).to(torch.uint8)
    expected = expected_color.cpu().to(torch.uint8)
    order = _lexicographic_order(points)
    expected_order = _lexicographic_order(expected_coord)
    if not torch.equal(decoded[order], expected[expected_order]):
        max_err = (decoded[order].to(torch.int16) - expected[expected_order].to(torch.int16)).abs().max().item()
        raise RuntimeError(f"Color PLY RGB values differ from VXZ by up to {max_err} levels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True, help="Input GLB model")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--blender", default="blender", help="Blender executable")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument("--color-space", choices=("linear", "srgb", "agx"), default="agx")
    parser.add_argument("--add-emission", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    glb = args.glb.expanduser().resolve()
    if glb.suffix.lower() != ".glb":
        raise ValueError(f"--glb must point to a GLB file, got {glb}")
    if not glb.is_file():
        raise FileNotFoundError(glb)
    if args.resolution <= 1:
        raise ValueError("--resolution must be greater than 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the intermediate pickle next to the outputs for reproducibility.
    dump_path = args.output_dir / (glb.stem + ".pbr.pkl")
    vxz_path = args.output_dir / (glb.stem + ".vxz")
    # The primary ``.ply`` is deliberately the colored point cloud.  Many
    # viewers ignore arbitrary names such as ``color_0`` and therefore show a
    # previously generated coordinate-only PLY as grey.  Keep a separate
    # coordinate-only file as ``_voxel.ply`` and a descriptive color alias.
    voxel_ply = args.output_dir / (glb.stem + "_voxel.ply")
    color_ply = args.output_dir / (glb.stem + "_color.ply")
    primary_ply = args.output_dir / (glb.stem + ".ply")

    run_blender_dump(args.blender, glb, dump_path)
    with dump_path.open("rb") as f:
        dump = pickle.load(f)

    import o_voxel

    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump,
        grid_size=args.resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        color_space=args.color_space,
        add_emission=args.add_emission,
        verbose=True,
    )
    if coord.ndim != 2 or coord.shape[1] != 3 or coord.shape[0] == 0:
        raise RuntimeError(f"Voxelizer returned invalid coordinates: {tuple(coord.shape)}")
    if "base_color" not in attr:
        raise RuntimeError("Voxelizer output has no base_color attribute")
    if attr["base_color"].shape[0] != coord.shape[0]:
        raise RuntimeError("Coordinate and base_color counts differ")

    o_voxel.io.write_vxz(str(vxz_path), coord.int().cpu(), attr)
    read_coord, read_attr = validate_vxz_roundtrip(vxz_path, coord.int().cpu(), attr)

    # Open3D emits conventional red/green/blue properties for the color PLY,
    # recognized by MeshLab and CloudCompare.  It also lets this visualizer
    # operate without the optional ``plyfile`` package used by o_voxel's
    # generic PLY reader/writer.
    write_open3d_ply(voxel_ply, read_coord)
    write_open3d_ply(color_ply, read_coord, read_attr["base_color"])
    # Make ``<model>.ply`` the convenient colored output users normally open.
    write_open3d_ply(primary_ply, read_coord, read_attr["base_color"])
    for path in (vxz_path, voxel_ply, color_ply, primary_ply):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    validate_color_ply(color_ply, read_coord, read_attr["base_color"])
    validate_color_ply(primary_ply, read_coord, read_attr["base_color"])
    print(f"Wrote {vxz_path}")
    print(f"Wrote {voxel_ply}")
    print(f"Wrote {color_ply}")
    print(f"Wrote {primary_ply} (colored)")
    print(f"Validated {coord.shape[0]} voxels ({args.color_space}, add_emission={args.add_emission})")


if __name__ == "__main__":
    main()
