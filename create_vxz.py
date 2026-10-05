#!/usr/bin/env python3
"""Download one GLB from BOS and convert it to O-Voxel.

The script follows the material dump path used by
``data_toolkit/blender_script/dump_pbr.py`` and the surface sampling path used
by the TRELLIS.2 data pipeline.  It intentionally keeps Blender invocation
outside of this process: Blender is used to import/normalize the GLB and to
write the PBR pickle, while the O-Voxel conversion runs in the Python
environment that contains the compiled extension.

Typical usage::

    python create_vxz.py 001402b7-b08b-44fd-a3d1-79e2344040f2 \
        --resolution 1024 --output-format vxzm

Use ``--visualize`` to write a colored PLY directly without creating a VXZ or
VXZM file.  With ``--upload``, every local source, intermediate and output
artifact is temporary and is removed even when conversion or upload fails.
BOS credentials are read from the environment by default:
``BOS_ACCESS_KEY_ID`` and ``BOS_SECRET_ACCESS_KEY``.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parent
OVOXEL_ROOT = ROOT / "o-voxel"
DUMP_SCRIPT = ROOT / "data_toolkit" / "blender_script" / "dump_pbr.py"
if str(OVOXEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OVOXEL_ROOT))


DEFAULT_INPUT_BUCKET = "mesh-data-resave-glb-v2"
DEFAULT_INPUT_FOLDER = "highpoly"
DEFAULT_OUTPUT_BUCKET = "texture-surface-sample"
DEFAULT_OUTPUT_FOLDER = "sample_vxz"
DEFAULT_BOS_ENDPOINT = "bj.bcebos.com"
UUID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Keep normalized geometry strictly inside the voxelizer's [-.5, .5] AABB.
# The native scan-line implementation treats the positive AABB limit as a
# half-open boundary, so triangles exactly on +.5 can disappear.  This value
# is also the normalization convention used by the repository's historical
# PBR and flexible-dual-grid data pipelines.
NORMALIZED_MESH_EXTENT = 0.99999
_NORMALIZED_MESH_EXTENT_KEY = "_trellis2_normalized_mesh_extent"


def _validate_uuid(value: str) -> str:
    """Validate a BOS object identifier before using it in local/BOS paths."""
    value = value.strip()
    if not value or not UUID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "uuid must contain only letters, digits, '_' or '-' and no path separators"
        )
    return value


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def normalize_dump_geometry(
    dump: Dict[str, Any],
    target_extent: float = NORMALIZED_MESH_EXTENT,
) -> Dict[str, Any]:
    """Normalize all voxelized mesh objects into the strict interior AABB.

    Blender's dump script fits the scene to ``[-.5, .5]``.  Exact contact
    with the positive limit is unsafe for the native voxelizer, however, so
    reproduce the established TRELLIS data-pipeline convention: recompute a
    global center and uniformly scale the longest side to ``0.99999``.

    The dump is updated in place so PBR and geometry conversion can share the
    exact same transformed arrays without duplicating a potentially large
    scene.  A private marker makes repeated calls idempotent.
    """
    import numpy as np

    if not isinstance(dump, dict):
        raise TypeError(f"PBR dump must be a dictionary, got {type(dump).__name__}")
    if not np.isfinite(target_extent) or not 0.0 < target_extent < 1.0:
        raise ValueError("target_extent must be finite and strictly between 0 and 1")
    if dump.get(_NORMALIZED_MESH_EXTENT_KEY) == float(target_extent):
        return dump

    objects = dump.get("objects")
    if not isinstance(objects, (list, tuple)) or not objects:
        raise ValueError("PBR dump contains no mesh objects")

    active = []
    bbox_min = None
    bbox_max = None
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"object {index} is not a dictionary")
        vertices = np.asarray(obj.get("vertices"))
        faces = np.asarray(obj.get("faces"))
        if vertices.size == 0 or faces.size == 0:
            continue
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"object {index} has invalid vertices shape {vertices.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError(f"object {index} contains non-finite vertices")

        # Match the old torch.float() normalization numerics and the native
        # voxelizer's float32 input exactly.
        vertices = vertices.astype(np.float32, copy=False)
        obj_min = vertices.min(axis=0)
        obj_max = vertices.max(axis=0)
        bbox_min = obj_min if bbox_min is None else np.minimum(bbox_min, obj_min)
        bbox_max = obj_max if bbox_max is None else np.maximum(bbox_max, obj_max)
        active.append((obj, vertices))

    if not active:
        raise ValueError("PBR dump contains no non-empty mesh objects")
    extent = float(np.max(bbox_max - bbox_min))
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("PBR dump mesh has a degenerate global bounding box")

    center = (bbox_min + bbox_max) * np.float32(0.5)
    scale = np.float32(target_extent) / np.float32(extent)
    for obj, vertices in active:
        normalized = (vertices - center) * scale
        # target_extent leaves substantially more margin than float32 roundoff,
        # so touching either AABB limit here indicates malformed input/numerics.
        if np.any(normalized <= -0.5) or np.any(normalized >= 0.5):
            raise RuntimeError("normalized mesh does not lie strictly inside [-0.5, 0.5]")
        obj["vertices"] = normalized

    dump[_NORMALIZED_MESH_EXTENT_KEY] = float(target_extent)
    return dump


def _make_bos_client(endpoint: str, access_key_id: str, secret_access_key: str):
    """Create a BOS client lazily, so --help works without the SDK installed."""
    if not access_key_id or not secret_access_key:
        raise RuntimeError(
            "BOS credentials are required. Set BOS_ACCESS_KEY_ID and "
            "BOS_SECRET_ACCESS_KEY, or pass the corresponding command-line options."
        )
    try:
        from baidubce.auth.bce_credentials import BceCredentials
        from baidubce.bce_client_configuration import BceClientConfiguration
        from baidubce.services.bos.bos_client import BosClient
    except ImportError as error:
        raise RuntimeError(
            "The baidubce package is required for BOS download/upload"
        ) from error
    config = BceClientConfiguration(
        credentials=BceCredentials(access_key_id, secret_access_key),
        endpoint=endpoint,
    )
    return BosClient(config)


def _bos_key(folder: str, uuid: str, suffix: str = ".glb") -> str:
    folder = folder.strip("/")
    return posixpath.join(folder, uuid[:2], uuid + suffix)


def download_glb(client: Any, bucket: str, folder: str, uuid: str, destination: Path) -> str:
    """Download ``<folder>/<uuid[:2]>/<uuid>.glb`` and return its BOS key."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    key = _bos_key(folder, uuid)
    print(f"Downloading bos://{bucket}/{key} -> {destination}")
    client.get_object_to_file(bucket, key, str(destination))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"BOS download produced no data: {destination}")
    return key


def upload_file(
    client: Any,
    bucket: str,
    folder: str,
    uuid: str,
    local_path: Path,
    upload_key: Optional[str] = None,
) -> str:
    """Upload one result and return the BOS object key."""
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    key = upload_key or _bos_key(folder, uuid, suffix="")
    if not upload_key:
        # _bos_key(..., suffix="") leaves the UUID without a suffix; use the
        # actual result name so .vxz/.vxzm/.ply is retained remotely.
        key = posixpath.join(folder.strip("/"), uuid[:2], local_path.name)
        if not folder.strip("/"):
            key = posixpath.join(uuid[:2], local_path.name)
    print(f"Uploading {local_path} -> bos://{bucket}/{key}")
    # The SDK's super-object method is misspelled in the versions used by the
    # reference pipeline.  Ordinary uploads cover normal voxel artifacts;
    # use the super-object API only when it is available and needed.
    if local_path.stat().st_size >= 5 * 1024**3 and hasattr(client, "put_super_obejct_from_file"):
        client.put_super_obejct_from_file(bucket, key, str(local_path), chunk_size=4096)
    else:
        client.put_object_from_file(bucket, key, str(local_path))
    return key


def run_blender_dump(blender: str, glb_path: Path, dump_path: Path) -> None:
    """Run the repository's Blender PBR dump script."""
    if not DUMP_SCRIPT.is_file():
        raise FileNotFoundError(f"Blender dump script not found: {DUMP_SCRIPT}")
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender,
        "-b",
        "-P",
        str(DUMP_SCRIPT),
        "--",
        "--object",
        str(glb_path),
        "--output_path",
        str(dump_path),
    ]
    print("Running Blender PBR dump:", " ".join(cmd))
    error_path = dump_path.with_name(dump_path.name + "_error.txt")
    try:
        subprocess.run(cmd, check=True)
        if not dump_path.is_file() or dump_path.stat().st_size == 0:
            detail = error_path.read_text(errors="replace") if error_path.exists() else "no dump error detail"
            raise RuntimeError(f"Blender did not produce {dump_path}: {detail}")
    except subprocess.CalledProcessError as error:
        detail = error_path.read_text(errors="replace") if error_path.exists() else "no dump error detail"
        raise RuntimeError(f"Blender PBR dump failed for {glb_path}: {detail}") from error
    finally:
        # dump_pbr.py can create this material-debug sidecar before raising.
        # Include its contents in the exception above, but never retain it as
        # a separate cache artifact.
        try:
            error_path.unlink()
        except FileNotFoundError:
            pass


def _load_dump(dump_path: Path) -> Dict[str, Any]:
    import pickle

    with dump_path.open("rb") as stream:
        dump = pickle.load(stream)
    if not isinstance(dump, dict) or not dump.get("objects"):
        raise RuntimeError(f"PBR dump is empty or malformed: {dump_path}")
    return dump


def _voxel_points(coord, grid_size, normal=None, normal_offset: float = 0.0):
    import numpy as np

    grid = np.asarray(grid_size, dtype=np.float64)
    points = (coord.detach().cpu().numpy().astype(np.float64) + 0.5) / grid - 0.5
    if normal is not None and normal_offset:
        normals = normal.detach().cpu().numpy().astype(np.float64) / 255.0 * 2.0 - 1.0
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
        points = points + normals * float(normal_offset) / grid
    return points


def write_color_ply(path: Path, coord, attr: Dict[str, Any], grid_size, normal_offset: float = 0.0) -> None:
    """Write a conventional Open3D PLY with RGB and optional surface normals."""
    import numpy as np
    import open3d as o3d

    if "base_color" not in attr or attr["base_color"].shape[1] != 3:
        raise ValueError("voxel attributes do not contain base_color[3]")
    normal = attr.get("normal")
    points = _voxel_points(coord, grid_size, normal=normal, normal_offset=normal_offset)
    colors = np.clip(attr["base_color"].detach().cpu().numpy().astype(np.float64) / 255.0, 0.0, 1.0)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if normal is not None and normal.shape[1] == 3:
        normals = normal.detach().cpu().numpy().astype(np.float64) / 255.0 * 2.0 - 1.0
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
        cloud.normals = o3d.utility.Vector3dVector(normals)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False, print_progress=False):
        raise RuntimeError(f"Open3D failed to write {path}")


def _write_metadata(args) -> Dict[str, Any]:
    return {
        "cluster_angle_degrees": args.cluster_angle_degrees,
        "color_space": args.color_space,
        "add_emission": bool(args.add_emission),
        "max_records_per_voxel": args.max_records_per_voxel,
        "max_total_records": args.max_total_records,
    }


def convert_dump(
    dump: Dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
) -> Tuple[Path, int]:
    """Convert a PBR dump and return ``(final_path, number_of_records)``."""
    import o_voxel

    normalize_dump_geometry(dump)

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
        coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(dump, **common)
    else:
        coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr_multi(
            dump,
            cluster_angle_degrees=args.cluster_angle_degrees,
            max_records_per_voxel=args.max_records_per_voxel,
            max_total_records=args.max_total_records,
            **common,
        )

    if coord.ndim != 2 or coord.shape[1] != 3 or coord.shape[0] == 0:
        raise RuntimeError(f"Voxelizer returned invalid coordinates: {tuple(coord.shape)}")
    if args.visualize:
        # Only VXZM can contain coincident records that benefit from a small
        # display-only normal separation.  VXZ points stay at voxel centers.
        normal_offset = args.normal_offset if args.output_format == "vxzm" else 0.0
        write_color_ply(output_path, coord, attr, args.resolution, normal_offset)
        return output_path, int(coord.shape[0])

    compression = args.compression
    if compression is None:
        # Use the fast default profile for both containers.  A caller can
        # still request another codec or compression level explicitly.
        compression = "zstd"
    if args.output_format == "vxz":
        # VXZ retains its historical schema.  The normal and separate
        # emissive debug attributes are not part of the legacy file.
        vxz_attr = {name: value for name, value in attr.items() if name not in ("normal", "emissive")}
        o_voxel.io.write_vxz(
            str(output_path), coord.int().cpu(), vxz_attr,
            compression=compression,
            compression_level=args.compression_level,
        )
    else:
        o_voxel.io.write_vxzm(
            str(output_path), coord, attr,
            grid_size=args.resolution,
            region_resolution=args.region_resolution,
            compression=compression,
            compression_level=args.compression_level,
            metadata=_write_metadata(args),
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"O-Voxel writer did not produce {output_path}")
    return output_path, int(coord.shape[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uuid", nargs="?", type=_validate_uuid,
                        help="GLB UUID stored in BOS")
    parser.add_argument("--uuid", dest="uuid_option", type=_validate_uuid,
                        help="GLB UUID stored in BOS (alternative to the positional argument)")
    parser.add_argument("--output-dir", type=Path, default=Path("./vxz_outputs"))
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Intermediate cache directory (default: <output-dir>/.cache)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Final local path; otherwise <output-dir>/<uuid>.<suffix>")
    parser.add_argument("--blender", default="blender", help="Blender executable")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--output-format", "--format", dest="output_format",
                        choices=("vxz", "vxzm"), default="vxz")
    parser.add_argument("--region-resolution", type=int, default=256)
    parser.add_argument("--cluster-angle-degrees", type=float, default=15.0)
    parser.add_argument("--max-records-per-voxel", type=_non_negative_int, default=0)
    parser.add_argument("--max-total-records", type=_non_negative_int, default=0)
    parser.add_argument("--mip-level-offset", type=float, default=0.0)
    parser.add_argument("--color-space", choices=("linear", "srgb", "agx"), default="agx")
    emission = parser.add_mutually_exclusive_group()
    emission.add_argument("--add-emission", dest="add_emission", action="store_true")
    emission.add_argument("--no-add-emission", dest="add_emission", action="store_false")
    parser.set_defaults(add_emission=True)
    parser.add_argument("--compression", choices=("none", "deflate", "lzma", "zstd"), default=None,
                        help="Container compression (default: zstd level 3)")
    parser.add_argument("--compression-level", type=int, default=None)
    parser.add_argument("--normal-offset", type=float, default=0.08,
                        help="PLY-only normal separation in fine-voxel units")
    parser.add_argument("--visualize", action="store_true",
                        help="Write a colored PLY directly; do not create .vxz/.vxzm")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timing", action="store_true")

    parser.add_argument("--bos-endpoint", default=os.getenv("BOS_ENDPOINT", DEFAULT_BOS_ENDPOINT))
    parser.add_argument("--bos-bucket", "--input-bucket", dest="bos_bucket",
                        default=os.getenv("BOS_INPUT_BUCKET", DEFAULT_INPUT_BUCKET))
    parser.add_argument("--bos-folder", "--input-folder", dest="bos_folder",
                        default=os.getenv("BOS_INPUT_FOLDER", DEFAULT_INPUT_FOLDER))
    parser.add_argument("--bos-access-key-id", default=os.getenv("BOS_ACCESS_KEY_ID", ""),
                        help=argparse.SUPPRESS)
    parser.add_argument("--bos-secret-access-key", default=os.getenv("BOS_SECRET_ACCESS_KEY", ""),
                        help=argparse.SUPPRESS)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--upload-bucket", default=os.getenv("BOS_OUTPUT_BUCKET", DEFAULT_OUTPUT_BUCKET))
    parser.add_argument("--upload-folder", default=os.getenv("BOS_OUTPUT_FOLDER", DEFAULT_OUTPUT_FOLDER))
    parser.add_argument("--upload-key", default=None,
                        help="Explicit BOS object key; default is <upload-folder>/<uuid[:2]>/<filename>")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing local output")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.uuid is None:
        args.uuid = args.uuid_option
    elif args.uuid_option is not None and args.uuid != args.uuid_option:
        parser.error("positional uuid and --uuid must match")
    if args.uuid is None:
        parser.error("a GLB uuid is required (positional argument or --uuid)")
    if args.resolution <= 1:
        parser.error("--resolution must be greater than 1")
    if args.output_format == "vxzm":
        if args.region_resolution < 4 or args.region_resolution > 1024 or args.region_resolution & (args.region_resolution - 1):
            parser.error("--region-resolution must be a power of two in [4, 1024]")
        if not 0.0 < args.cluster_angle_degrees < 180.0:
            parser.error("--cluster-angle-degrees must be in (0, 180)")
    if args.normal_offset < 0:
        parser.error("--normal-offset must be non-negative")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_dir = (args.cache_dir or (args.output_dir / ".cache")).expanduser().resolve()
    suffix = ".ply" if args.visualize else "." + args.output_format
    output_path = args.output.expanduser().resolve() if args.output else args.output_dir / (args.uuid + suffix)
    if output_path.suffix.lower() != suffix:
        raise ValueError(f"--output must use the {suffix} suffix, got {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {output_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # BOS is needed for the input even when only a local artifact is retained.
    client = _make_bos_client(args.bos_endpoint, args.bos_access_key_id, args.bos_secret_access_key)
    glb_path = args.cache_dir / f"{args.uuid}.glb"
    dump_path = args.cache_dir / f"{args.uuid}.pbr.pkl"
    try:
        download_glb(client, args.bos_bucket, args.bos_folder, args.uuid, glb_path)
        run_blender_dump(args.blender, glb_path, dump_path)
        dump = _load_dump(dump_path)
        final_path, record_count = convert_dump(dump, args, output_path)
        print(f"Wrote {final_path} ({record_count} {'records' if args.output_format == 'vxzm' else 'voxels'})")

        if args.upload:
            upload_file(client, args.upload_bucket, args.upload_folder, args.uuid, final_path, args.upload_key)
            print("Upload complete")
        return 0
    finally:
        # Never retain Blender's material-error sidecar.  With --upload all
        # local artifacts are temporary and are removed regardless of success.
        cleanup = [glb_path, dump_path.with_name(dump_path.name + "_error.txt")]
        if args.upload:
            cleanup.extend((dump_path, output_path))
        for path in cleanup:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
