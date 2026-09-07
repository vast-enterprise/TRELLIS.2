#!/usr/bin/env python3
"""End-to-end GLB -> Blender dump -> VXZM -> colored/normal PLY."""
from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "o-voxel"))
from test_glb_to_vxz import run_blender_dump


def _canonical_order(coord, attr):
    """Sort records by coordinates and all attributes, preserving pairing."""
    columns = [coord.numpy()[:, index] for index in range(3)]
    for name in sorted(attr):
        value = attr[name].numpy()
        columns.extend(value[:, index] for index in range(value.shape[1]))
    # np.lexsort treats the final key as primary.
    return np.lexsort(tuple(reversed(columns)))


def validate_roundtrip(coord, attr, read_coord, read_attr):
    if set(attr) != set(read_attr):
        raise RuntimeError(
            f"VXZM attributes changed: {sorted(attr)} vs {sorted(read_attr)}"
        )
    source_order = _canonical_order(coord.cpu(), {k: v.cpu() for k, v in attr.items()})
    read_order = _canonical_order(read_coord.cpu(), {k: v.cpu() for k, v in read_attr.items()})
    if not np.array_equal(coord.cpu().numpy()[source_order],
                          read_coord.cpu().numpy()[read_order]):
        raise RuntimeError("VXZM coordinates changed during round-trip")
    for name, value in attr.items():
        if not np.array_equal(value.cpu().numpy()[source_order],
                              read_attr[name].cpu().numpy()[read_order]):
            raise RuntimeError(f"VXZM attribute {name} changed during round-trip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--region-resolution", type=int, default=256)
    parser.add_argument("--cluster-angle-degrees", type=float, default=15.0)
    parser.add_argument("--color-space", choices=("linear", "srgb", "agx"), default="agx")
    parser.add_argument("--add-emission", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal-offset", type=float, default=0.08,
                        help="PLY-only separation in units of one fine voxel")
    parser.add_argument("--max-records-per-voxel", type=int, default=0)
    parser.add_argument("--max-total-records", type=int, default=0)
    args = parser.parse_args()
    if args.glb.suffix.lower() != ".glb" or not args.glb.is_file():
        raise ValueError(f"--glb must point to an existing GLB file: {args.glb}")
    if args.resolution <= 1:
        raise ValueError("--resolution must be greater than 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_path = args.output_dir / f"{args.glb.stem}.pbr.pkl"
    vxzm_path = args.output_dir / f"{args.glb.stem}.vxzm"
    ply_path = args.output_dir / f"{args.glb.stem}_vxzm.ply"
    run_blender_dump(args.blender, args.glb.resolve(), dump_path)
    with dump_path.open("rb") as f:
        dump = pickle.load(f)

    import o_voxel
    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr_multi(
        dump, grid_size=args.resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        color_space=args.color_space, add_emission=args.add_emission,
        cluster_angle_degrees=args.cluster_angle_degrees,
        max_records_per_voxel=args.max_records_per_voxel,
        max_total_records=args.max_total_records, verbose=True,
    )
    o_voxel.io.write_vxzm(vxzm_path, coord, attr, grid_size=args.resolution,
                           region_resolution=args.region_resolution,
                           metadata={
                               "cluster_angle_degrees": args.cluster_angle_degrees,
                               "color_space": args.color_space,
                               "add_emission": args.add_emission,
                               "max_records_per_voxel": args.max_records_per_voxel,
                               "max_total_records": args.max_total_records,
                           })
    read_coord, read_attr = o_voxel.io.read_vxzm(vxzm_path)
    info = o_voxel.io.read_vxzm_info(vxzm_path)
    if len(read_coord) != len(coord) or info["num_records"] != len(coord):
        raise RuntimeError("VXZM record count changed during round-trip")
    validate_roundtrip(coord, attr, read_coord, read_attr)
    # Exercise file-like input as well as the ordinary path-based readers.
    o_voxel.io.vxzm_to_ply(io.BytesIO(vxzm_path.read_bytes()), ply_path,
                           normal_offset=args.normal_offset)
    import open3d as o3d
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if not (len(cloud.points) == len(cloud.colors) == len(cloud.normals) == len(coord)):
        raise RuntimeError("VXZM PLY point/color/normal counts differ")
    ply_points = np.asarray(cloud.points)
    ply_normals = np.asarray(cloud.normals)
    ply_color = np.clip(np.rint(np.asarray(cloud.colors) * 255.0), 0, 255).astype(np.uint8)
    expected_color = read_attr["base_color"].numpy()
    if not np.array_equal(ply_color, expected_color):
        max_error = np.abs(ply_color.astype(np.int16) - expected_color.astype(np.int16)).max()
        raise RuntimeError(f"VXZM PLY RGB differs from stored base_color by {max_error} levels")
    expected_normals = read_attr["normal"].numpy().astype(np.float64) / 255.0 * 2.0 - 1.0
    normal_lengths = np.linalg.norm(expected_normals, axis=1, keepdims=True)
    expected_normals = np.divide(
        expected_normals, normal_lengths, out=np.zeros_like(expected_normals),
        where=normal_lengths > 1e-12,
    )
    if not np.allclose(ply_normals, expected_normals, atol=1e-6):
        raise RuntimeError("VXZM PLY normals differ from stored quantized normals")
    grid = np.asarray(info["grid_size"], dtype=np.float64)
    expected_points = (read_coord.numpy().astype(np.float64) + 0.5) / grid - 0.5
    expected_points += expected_normals * args.normal_offset / grid
    if not np.allclose(ply_points, expected_points, atol=1e-6):
        raise RuntimeError("VXZM PLY positions do not match voxel centers/normal offset")
    print(f"Wrote {vxzm_path}: {info['num_records']} records, "
          f"{info['num_unique_voxels']} unique voxels, {info['num_regions']} regions")
    print(f"Wrote {ply_path} with RGB and normals")


if __name__ == "__main__":
    main()
