#!/usr/bin/env python3
"""Create an intersected geometry/PBR voxel pair from one BOS GLB.

This is the geometry-aware counterpart of :mod:`create_vxz`.  The GLB is
dumped once with Blender, then the dump is used for both conversions:

* ``texture/<uuid>.vxz`` (or ``.vxzm``) contains the PBR surface attributes;
* ``geometry/<uuid>.vxz`` contains flexible dual-grid ``vertices`` and
  ``intersected`` attributes.

Before either file is written, coordinates are compared exactly.  Only the
intersection of the PBR and dual-grid voxel coordinates is retained.  This
means that PBR and geometry files contain exactly the same voxel-coordinate
set; all non-common voxels are discarded.

For multi-surface VXZM output, every PBR record at a common coordinate is
retained while the geometry side still contains one dual-grid row per voxel.

The default BOS upload layout is::

    <upload-folder>/texture/<uuid[:2]>/<uuid>.<format>
    <upload-folder>/geometry/<uuid[:2]>/<uuid>.vxz

The input GLB layout is the same as ``create_vxz.py`` (``--bos-folder`` plus
``<uuid[:2]>/<uuid>.glb``).
"""

from __future__ import annotations

import argparse
import os
import posixpath
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    import torch

# Reuse the BOS, Blender and validation helpers, while keeping conversion and
# output handling here so the two generated files can be intersected first.
import create_vxz as _pbr

# Public module-level aliases make orchestration easy to mock in the same way
# as ``tests/test_create_vxz.py`` and preserve the helper names users already
# know from create_vxz.py.
_make_bos_client = _pbr._make_bos_client
download_glb = _pbr.download_glb
run_blender_dump = _pbr.run_blender_dump
_load_dump = _pbr._load_dump
upload_file = _pbr.upload_file
write_color_ply = _pbr.write_color_ply


ROOT = Path(__file__).resolve().parent
OVOXEL_ROOT = ROOT / "o-voxel"
if str(OVOXEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OVOXEL_ROOT))

DEFAULT_OUTPUT_DIR = Path("./geo_pbr_outputs")
DEFAULT_UPLOAD_FOLDER = "sample_vxz"


def _grid_size(value: int | Tuple[int, int, int] | list[int]) -> int | list[int]:
    """Validate and normalize a scalar or three-dimensional grid size."""
    if isinstance(value, int):
        if value <= 1:
            raise ValueError("grid resolution must be greater than 1")
        return value
    values = [int(x) for x in value]
    if len(values) != 3 or any(x <= 1 for x in values):
        raise ValueError("grid resolution must contain three values greater than 1")
    return values


def _mesh_from_dump(dump: Dict[str, Any]) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Build one indexed mesh from the objects in a Blender PBR dump.

    Both this geometry path and the PBR path use the same idempotent strict
    interior normalization.  This avoids the voxelizer's positive-boundary
    issue without allowing their coordinate systems to diverge.
    """
    import numpy as np
    import torch

    _pbr.normalize_dump_geometry(dump)

    vertices = []
    faces = []
    start = 0
    for index, obj in enumerate(dump.get("objects", [])):
        obj_vertices = np.asarray(obj.get("vertices"))
        obj_faces = np.asarray(obj.get("faces"))
        if obj_vertices.size == 0 or obj_faces.size == 0:
            continue
        if obj_vertices.ndim != 2 or obj_vertices.shape[1] != 3:
            raise ValueError(f"object {index} has invalid vertices shape {obj_vertices.shape}")
        if obj_faces.ndim != 2 or obj_faces.shape[1] != 3:
            raise ValueError(f"object {index} has invalid faces shape {obj_faces.shape}")
        if not np.isfinite(obj_vertices).all():
            raise ValueError(f"object {index} contains non-finite vertices")
        if np.any(obj_faces < 0) or np.any(obj_faces >= len(obj_vertices)):
            raise ValueError(f"object {index} contains out-of-range face indices")
        vertices.append(obj_vertices.astype(np.float32, copy=False))
        faces.append(obj_faces.astype(np.int64, copy=False) + start)
        start += len(obj_vertices)
    if not vertices:
        raise ValueError("PBR dump contains no non-empty mesh objects")
    all_vertices = np.concatenate(vertices, axis=0)
    all_faces = np.concatenate(faces, axis=0)
    if np.any(all_vertices < -0.50001) or np.any(all_vertices > 0.50001):
        raise ValueError(
            "Blender dump vertices are outside [-0.5, 0.5]; regenerate it with dump_pbr.py"
        )
    return torch.from_numpy(all_vertices).float(), torch.from_numpy(all_faces).long()


def convert_geometry(
    dump: Dict[str, Any],
    resolution: int | list[int],
    *,
    face_weight: float = 1.0,
    boundary_weight: float = 0.2,
    regularization_weight: float = 1e-2,
    timing: bool = False,
) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
    """Voxelize a dump into the repository's flexible dual-grid VXZ schema."""
    import o_voxel
    import torch

    vertices, faces = _mesh_from_dump(dump)
    grid = _grid_size(resolution)
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices,
        faces=faces,
        grid_size=grid,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=float(face_weight),
        boundary_weight=float(boundary_weight),
        regularization_weight=float(regularization_weight),
        timing=bool(timing),
    )
    if voxel_indices.ndim != 2 or voxel_indices.shape[1] != 3:
        raise RuntimeError(f"dual-grid converter returned invalid coordinates {voxel_indices.shape}")
    grid_tensor = torch.as_tensor(grid, dtype=voxel_indices.dtype, device=voxel_indices.device)
    if torch.any(voxel_indices < 0) or torch.any(voxel_indices >= grid_tensor):
        raise RuntimeError("dual-grid converter returned coordinates outside the requested grid")
    if dual_vertices.ndim != 2 or dual_vertices.shape[0] != voxel_indices.shape[0] or dual_vertices.shape[1] != 3:
        raise RuntimeError("dual-grid converter returned incompatible dual vertices")
    if not torch.isfinite(dual_vertices).all():
        raise RuntimeError("dual-grid converter returned non-finite dual vertices")
    if intersected.ndim != 2 or intersected.shape[0] != voxel_indices.shape[0] or intersected.shape[1] < 3:
        raise RuntimeError("dual-grid converter returned invalid intersection flags")

    # This is the exact quantization/readback convention used by
    # data_toolkit/dual_grid.py and FlexiDualGridDataset.
    scale = torch.as_tensor(grid, dtype=dual_vertices.dtype, device=dual_vertices.device)
    local_vertices = dual_vertices * scale - voxel_indices.to(dual_vertices)
    if torch.any(local_vertices < -1e-3) or torch.any(local_vertices > 1 + 1e-3):
        raise RuntimeError("dual-grid vertices lie outside their voxel cells")
    dual_vertices = torch.clamp(local_vertices, 0, 1)
    # Match data_toolkit/dual_grid.py exactly: conversion to uint8 truncates
    # fractional values (rather than rounding), which keeps readback and
    # training-time dual-grid values bit-for-bit compatible.
    dual_vertices = torch.clamp(dual_vertices * 255, 0, 255).to(torch.uint8)
    flags = intersected[:, :3].to(torch.uint8)
    flags = (flags[:, 0:1] + 2 * flags[:, 1:2] + 4 * flags[:, 2:3]).to(torch.uint8)
    return voxel_indices.to(torch.int32).cpu(), {
        "vertices": dual_vertices.cpu(),
        "intersected": flags.cpu(),
    }


def intersect_voxels(
    pbr_coord: "torch.Tensor",
    pbr_attr: Dict[str, "torch.Tensor"],
    geometry_coord: "torch.Tensor",
    geometry_attr: Dict[str, "torch.Tensor"],
) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"], "torch.Tensor", Dict[str, "torch.Tensor"], int]:
    """Filter both representations to the exact intersection of coordinates.

    The mask is coordinate-based rather than a one-to-one join. Attribute rows
    retain their original order, which keeps the output deterministic.
    """
    import numpy as np
    import torch

    def validate(name: str, coord: "torch.Tensor", attrs: Dict[str, "torch.Tensor"]) -> None:
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"{name} coordinates must have shape [N, 3], got {tuple(coord.shape)}")
        for key, value in attrs.items():
            if value.ndim == 0 or value.shape[0] != coord.shape[0]:
                raise ValueError(f"{name} attribute {key!r} has {tuple(value.shape)} for {coord.shape[0]} coordinates")

    validate("PBR", pbr_coord, pbr_attr)
    validate("geometry", geometry_coord, geometry_attr)
    # The two CPU voxelizers normally emit the same unique coordinate rows in
    # the same order.  Avoid the conversion, three sort-based membership
    # operations, and copies of every PBR attribute in that common case.
    if (
        pbr_coord.device == geometry_coord.device
        and pbr_coord.shape == geometry_coord.shape
        and torch.equal(pbr_coord, geometry_coord)
    ):
        return pbr_coord, pbr_attr, geometry_coord, geometry_attr, int(len(geometry_coord))

    # A 1024^3 conversion can contain millions of rows. Python tuple sets
    # consume hundreds of bytes per coordinate, so encode XYZ losslessly into
    # one uint64.  Joint minima and extents make the mapping valid even for
    # signed coordinates, although production voxel coordinates are
    # non-negative.
    pbr_xyz = pbr_coord.detach().cpu().numpy().astype(np.int64, copy=False)
    geometry_xyz = geometry_coord.detach().cpu().numpy().astype(np.int64, copy=False)
    if len(pbr_xyz) == 0 or len(geometry_xyz) == 0:
        pbr_mask_np = np.zeros(len(pbr_xyz), dtype=bool)
        geometry_mask_np = np.zeros(len(geometry_xyz), dtype=bool)
        common_count = 0
    else:
        lower = np.minimum(pbr_xyz.min(axis=0), geometry_xyz.min(axis=0))
        upper = np.maximum(pbr_xyz.max(axis=0), geometry_xyz.max(axis=0))
        extent = upper.astype(object) - lower.astype(object) + 1
        uint64_max = int(np.iinfo(np.uint64).max)
        if any(int(value) > uint64_max for value in extent):
            raise ValueError("voxel coordinate extent is too large for exact uint64 intersection")
        key_space = int(extent[0]) * int(extent[1]) * int(extent[2])
        if key_space > uint64_max:
            raise ValueError("voxel coordinate range is too large for exact uint64 intersection")

        def keys(xyz):
            shifted = xyz.astype(np.uint64) - lower.astype(np.uint64)
            return ((shifted[:, 0] * np.uint64(extent[1]) + shifted[:, 1]) *
                    np.uint64(extent[2]) + shifted[:, 2])

        pbr_keys = keys(pbr_xyz)
        geometry_keys = keys(geometry_xyz)
        # Geometry rows must be unique.  Sort them once and reuse the sorted
        # keys to join every PBR row by binary search.  The old path performed
        # one unique and two independent sort-based isin operations.
        geometry_order = np.argsort(geometry_keys, kind="stable")
        sorted_geometry_keys = geometry_keys[geometry_order]
        if np.any(sorted_geometry_keys[1:] == sorted_geometry_keys[:-1]):
            raise ValueError("geometry coordinates must be unique")
        positions = np.searchsorted(sorted_geometry_keys, pbr_keys)
        pbr_mask_np = positions < len(sorted_geometry_keys)
        candidate_positions = positions[pbr_mask_np]
        pbr_mask_np[pbr_mask_np] = (
            sorted_geometry_keys[candidate_positions] == pbr_keys[pbr_mask_np]
        )
        geometry_mask_np = np.zeros(len(geometry_keys), dtype=bool)
        geometry_mask_np[geometry_order[positions[pbr_mask_np]]] = True
        common_count = int(np.count_nonzero(geometry_mask_np))

    pbr_mask = torch.from_numpy(pbr_mask_np).to(device=pbr_coord.device)
    geometry_mask = torch.from_numpy(geometry_mask_np).to(device=geometry_coord.device)

    def apply(coord: "torch.Tensor", attrs: Dict[str, "torch.Tensor"], mask: "torch.Tensor"):
        return coord[mask], {
            key: value[mask.to(device=value.device)] for key, value in attrs.items()
        }

    pbr_coord, pbr_attr = apply(pbr_coord, pbr_attr, pbr_mask)
    geometry_coord, geometry_attr = apply(geometry_coord, geometry_attr, geometry_mask)
    return pbr_coord, pbr_attr, geometry_coord, geometry_attr, common_count


def _convert_pbr_attributes(dump: Dict[str, Any], args: argparse.Namespace):
    """Run the same PBR voxelizer and options as ``create_vxz.py``."""
    import o_voxel

    # convert_geometry() applies this same in-place, idempotent transform.
    # Whichever conversion runs first therefore establishes one shared frame.
    _pbr.normalize_dump_geometry(dump)

    common = {
        "grid_size": args.resolution,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "mip_level_offset": args.mip_level_offset,
        "verbose": args.verbose,
        "timing": args.timing,
        "color_space": args.color_space,
        "add_emission": args.add_emission,
    }
    if args.output_format == "vxz":
        return o_voxel.convert.blender_dump_to_volumetric_attr(dump, **common)
    return o_voxel.convert.blender_dump_to_volumetric_attr_multi(
        dump,
        cluster_angle_degrees=args.cluster_angle_degrees,
        max_records_per_voxel=args.max_records_per_voxel,
        max_total_records=args.max_total_records,
        **common,
    )


def _write_pbr(path: Path, coord: "torch.Tensor", attr: Dict[str, "torch.Tensor"], args: argparse.Namespace) -> None:
    import o_voxel

    compression = args.compression
    if compression is None:
        compression = "zstd"
    if args.visualize:
        normal_offset = args.normal_offset if args.output_format == "vxzm" else 0.0
        write_color_ply(path, coord, attr, args.resolution, normal_offset)
        return
    if args.output_format == "vxz":
        attrs = {name: value for name, value in attr.items() if name not in ("normal", "emissive")}
        o_voxel.io.write_vxz(str(path), coord.int().cpu(), attrs,
                             compression=compression, compression_level=args.compression_level)
    else:
        o_voxel.io.write_vxzm(
            str(path), coord, attr,
            grid_size=args.resolution,
            region_resolution=args.region_resolution,
            compression=compression,
            compression_level=args.compression_level,
            metadata=_pbr._write_metadata(args),
        )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"PBR writer did not produce {path}")


def _write_geometry(
    path: Path,
    coord: "torch.Tensor",
    attr: Dict[str, "torch.Tensor"],
    args: argparse.Namespace,
) -> None:
    import o_voxel

    compression = args.geometry_compression or args.compression or "zstd"
    path.parent.mkdir(parents=True, exist_ok=True)
    o_voxel.io.write_vxz(
        str(path), coord.int().cpu(), attr, compression=compression,
        compression_level=(args.geometry_compression_level
                           if args.geometry_compression_level is not None
                           else args.compression_level),
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"geometry writer did not produce {path}")


def _default_upload_key(folder: str, uuid: str, path: Path) -> str:
    if folder.strip("/"):
        return posixpath.join(folder.strip("/"), uuid[:2], path.name)
    return posixpath.join(uuid[:2], path.name)


def _default_geometry_upload_key(folder: str, uuid: str) -> str:
    """Use the canonical geometry basename, independent of local cache names."""
    name = f"{uuid}.vxz"
    return posixpath.join(folder.strip("/"), uuid[:2], name) if folder.strip("/") else posixpath.join(uuid[:2], name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uuid", nargs="?", type=_pbr._validate_uuid)
    parser.add_argument("--uuid", dest="uuid_option", type=_pbr._validate_uuid)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Local PBR output path")
    parser.add_argument("--geometry-output", type=Path, default=None, help="Local dual-grid VXZ path")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--geometry-resolution", type=int, default=None,
                        help="Dual-grid resolution (default: --resolution)")
    parser.add_argument(
        "--output-format", "--format", dest="output_format",
        choices=("vxz", "vxzm"), default="vxz",
        help="PBR container format (geometry is always VXZ)",
    )
    parser.add_argument("--region-resolution", type=int, default=256)
    parser.add_argument("--cluster-angle-degrees", type=float, default=15.0)
    parser.add_argument("--max-records-per-voxel", type=_pbr._non_negative_int, default=0)
    parser.add_argument("--max-total-records", type=_pbr._non_negative_int, default=0)
    parser.add_argument("--mip-level-offset", type=float, default=0.0)
    parser.add_argument("--color-space", choices=("linear", "srgb", "agx"), default="agx")
    emission = parser.add_mutually_exclusive_group()
    emission.add_argument("--add-emission", dest="add_emission", action="store_true")
    emission.add_argument("--no-add-emission", dest="add_emission", action="store_false")
    parser.set_defaults(add_emission=True)
    parser.add_argument("--compression", choices=("none", "deflate", "lzma", "zstd"), default=None,
                        help="PBR compression (default: zstd level 3)")
    parser.add_argument("--geometry-compression", choices=("none", "deflate", "lzma", "zstd"), default=None,
                        help="Geometry compression (default: zstd level 3)")
    parser.add_argument("--compression-level", type=int, default=None)
    parser.add_argument("--geometry-compression-level", type=int, default=None)
    parser.add_argument("--normal-offset", type=float, default=0.08,
                        help="PLY-only normal separation in fine-voxel units")
    parser.add_argument("--visualize", action="store_true",
                        help="Write the filtered PBR as PLY")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--bos-endpoint", default=os.getenv("BOS_ENDPOINT", _pbr.DEFAULT_BOS_ENDPOINT))
    parser.add_argument("--bos-bucket", "--input-bucket", dest="bos_bucket",
                        default=os.getenv("BOS_INPUT_BUCKET", _pbr.DEFAULT_INPUT_BUCKET))
    parser.add_argument("--bos-folder", "--input-folder", dest="bos_folder",
                        default=os.getenv("BOS_INPUT_FOLDER", _pbr.DEFAULT_INPUT_FOLDER))
    parser.add_argument("--bos-access-key-id", default=os.getenv("BOS_ACCESS_KEY_ID", ""), help=argparse.SUPPRESS)
    parser.add_argument(
        "--bos-secret-access-key",
        default=os.getenv("BOS_SECRET_ACCESS_KEY", ""),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-bucket", default=os.getenv("BOS_OUTPUT_BUCKET", _pbr.DEFAULT_OUTPUT_BUCKET))
    parser.add_argument("--upload-folder", default=os.getenv("BOS_OUTPUT_FOLDER", DEFAULT_UPLOAD_FOLDER))
    parser.add_argument("--texture-upload-folder", default=None,
                        help="Override <upload-folder>/texture")
    parser.add_argument("--geometry-upload-folder", default=None,
                        help="Override <upload-folder>/geometry")
    parser.add_argument("--upload-key", default=None, help="Explicit PBR BOS key")
    parser.add_argument("--geometry-upload-key", default=None, help="Explicit geometry BOS key")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args(parser: argparse.ArgumentParser, argv: Optional[list[str]]) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if args.uuid is None:
        args.uuid = args.uuid_option
    elif args.uuid_option is not None and args.uuid != args.uuid_option:
        parser.error("positional uuid and --uuid must match")
    if args.uuid is None:
        parser.error("a GLB uuid is required (positional argument or --uuid)")
    if args.resolution <= 1 or (args.geometry_resolution is not None and args.geometry_resolution <= 1):
        parser.error("--resolution and --geometry-resolution must be greater than 1")
    if args.output_format == "vxzm":
        if (
            args.region_resolution < 4
            or args.region_resolution > 1024
            or args.region_resolution & (args.region_resolution - 1)
        ):
            parser.error("--region-resolution must be a power of two in [4, 1024]")
        if not 0.0 < args.cluster_angle_degrees < 180.0:
            parser.error("--cluster-angle-degrees must be in (0, 180)")
    if args.normal_offset < 0:
        parser.error("--normal-offset must be non-negative")
    args.geometry_resolution = args.geometry_resolution or args.resolution
    if args.geometry_resolution != args.resolution:
        parser.error(
            "--geometry-resolution must equal --resolution: exact voxel-coordinate "
            "intersection requires both representations to use the same grid"
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = (args.cache_dir or args.output_dir / ".cache").expanduser().resolve()
    pbr_suffix = ".ply" if args.visualize else "." + args.output_format
    args.output_path = (
        args.output.expanduser().resolve()
        if args.output
        else args.output_dir / (args.uuid + pbr_suffix)
    )
    args.geometry_output_path = (
        args.geometry_output.expanduser().resolve()
        if args.geometry_output
        else args.output_dir / (args.uuid + ".geometry.vxz")
    )
    if args.output_path.suffix.lower() != pbr_suffix:
        parser.error(f"--output must use the {pbr_suffix} suffix")
    if args.geometry_output_path.suffix.lower() != ".vxz":
        parser.error("--geometry-output must use the .vxz suffix")
    if args.output_path == args.geometry_output_path:
        parser.error("PBR and geometry output paths must be different")
    for path in (args.output_path, args.geometry_output_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists (use --overwrite): {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.geometry_output_path.parent.mkdir(parents=True, exist_ok=True)
    return args


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = _parse_args(parser, argv)
    client = _make_bos_client(args.bos_endpoint, args.bos_access_key_id, args.bos_secret_access_key)
    glb_path = args.cache_dir / f"{args.uuid}.glb"
    dump_path = args.cache_dir / f"{args.uuid}.pbr.pkl"
    try:
        download_glb(client, args.bos_bucket, args.bos_folder, args.uuid, glb_path)
        run_blender_dump(args.blender, glb_path, dump_path)
        dump = _load_dump(dump_path)
        pbr_coord, pbr_attr = _convert_pbr_attributes(dump, args)
        geo_coord, geo_attr = convert_geometry(dump, args.geometry_resolution, timing=args.timing)
        pbr_coord, pbr_attr, geo_coord, geo_attr, common = intersect_voxels(
            pbr_coord, pbr_attr, geo_coord, geo_attr
        )
        if common == 0:
            raise RuntimeError(f"PBR and geometry voxel sets have no intersection for {args.uuid}")
        _write_pbr(args.output_path, pbr_coord, pbr_attr, args)
        _write_geometry(args.geometry_output_path, geo_coord, geo_attr, args)
        print(f"Wrote PBR {args.output_path} ({len(pbr_coord)} records)")
        print(f"Wrote geometry {args.geometry_output_path} ({len(geo_coord)} voxels)")
        print(f"Retained {common} common voxel coordinates")
        if args.upload:
            texture_folder = args.texture_upload_folder or posixpath.join(args.upload_folder.strip("/"), "texture")
            geometry_folder = args.geometry_upload_folder or posixpath.join(args.upload_folder.strip("/"), "geometry")
            pbr_key = args.upload_key or _default_upload_key(texture_folder, args.uuid, args.output_path)
            # Geometry is stored separately from texture, so it intentionally
            # uses the same canonical ``<uuid>.vxz`` basename as the dataset's
            # precomputed dual-grid files (the local default may be suffixed
            # ``.geometry`` to avoid colliding with the PBR artifact).
            geo_key = args.geometry_upload_key or _default_geometry_upload_key(geometry_folder, args.uuid)
            if pbr_key == geo_key:
                raise ValueError("PBR and geometry BOS object keys must be different")
            upload_file(client, args.upload_bucket, texture_folder, args.uuid, args.output_path, pbr_key)
            upload_file(client, args.upload_bucket, geometry_folder, args.uuid, args.geometry_output_path, geo_key)
            print(f"Uploaded texture -> bos://{args.upload_bucket}/{pbr_key}")
            print(f"Uploaded geometry -> bos://{args.upload_bucket}/{geo_key}")
        return 0
    finally:
        # Match create_vxz.py: never retain Blender's error sidecar, and with
        # --upload remove every local artifact even if either upload fails.
        cleanup = [glb_path, dump_path.with_name(dump_path.name + "_error.txt")]
        if args.upload:
            cleanup.extend((dump_path, args.output_path, args.geometry_output_path))
        for path in cleanup:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
