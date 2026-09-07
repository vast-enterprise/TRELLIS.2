"""Multi-surface voxel storage (.vxzm).

Unlike :mod:`vxz`, a VXZM file permits multiple records with the same voxel
coordinate.  Records are grouped into coarse 256^3 regions.  Each region has
one count and one offset, followed by a compact array of local coordinates and
attributes.  The format is deliberately independent from VXZ so existing
single-surface readers remain unchanged.
"""

from typing import Dict, Union, Optional, Literal
import json
import os
import struct
import tempfile

import numpy as np
import torch

from ..serialize import encode_seq, decode_seq
from .. import _C
from .vxz import _compress, _decompress, DEFAULT_COMPRESION_LEVEL

__all__ = [
    "read_vxzm", "read_vxzm_info", "read_vxzm_regions", "write_vxzm",
    "vxzm_to_ply",
]

MAGIC = b"VXZM"
VERSION = 0


def _read_file(file) -> bytes:
    if isinstance(file, (bytes, bytearray, memoryview)):
        return bytes(file)
    if isinstance(file, (str, os.PathLike)):
        with open(file, "rb") as f:
            return f.read()
    return file.read()


def _parse_header(data: bytes, file_size: Optional[int] = None) -> Dict:
    if len(data) < 9:
        raise ValueError("Truncated VXZM header")
    if data[:4] != MAGIC:
        raise ValueError("Invalid VXZM file type")
    if data[4] != VERSION:
        raise ValueError(f"Unsupported VXZM version {data[4]}")
    header_end = struct.unpack(">I", data[5:9])[0]
    if header_end < 9 or header_end > len(data):
        raise ValueError("Invalid VXZM header offset")
    try:
        info = json.loads(data[9:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid VXZM JSON header") from error
    _validate_header(info, header_end, file_size)
    return info


def _validate_header(info: Dict, header_end: int,
                     file_size: Optional[int] = None):
    if not isinstance(info, dict):
        raise ValueError("VXZM header must be a JSON object")
    if info.get("format") != "VXZM" or info.get("version") != VERSION:
        raise ValueError("Invalid VXZM header format/version")
    if int(info.get("binary_start", -1)) != header_end:
        raise ValueError("VXZM binary_start does not match JSON header end")
    compression = info.get("compression")
    if compression not in DEFAULT_COMPRESION_LEVEL:
        raise ValueError(f"Invalid VXZM compression {compression!r}")
    if not isinstance(info.get("compression_level"), int):
        raise ValueError("Invalid VXZM compression_level")
    if info.get("normal_source") != "surface_authored_with_winding_guard":
        raise ValueError(
            "VXZM normal_source must be 'surface_authored_with_winding_guard'"
        )

    try:
        grid_size = np.asarray(info["grid_size"], dtype=np.int64)
        block = np.asarray(info["region_block_size"], dtype=np.int64)
        region_grid = np.asarray(info["region_grid_size"], dtype=np.int64)
        region_resolution = int(info["region_resolution"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid VXZM grid header") from error
    if (grid_size.shape != (3,) or block.shape != (3,) or
            region_grid.shape != (3,) or np.any(grid_size <= 0) or
            np.any(block <= 0) or np.any(region_grid <= 0) or
            region_resolution < 4 or region_resolution > 1024 or
            region_resolution & (region_resolution - 1) or
            np.any(block > 256) or
            np.any(block != np.ceil(grid_size / region_resolution).astype(np.int64)) or
            np.any(region_grid != np.ceil(grid_size / block).astype(np.int64)) or
            np.any(region_grid > region_resolution)):
        raise ValueError("Invalid VXZM grid header")

    try:
        num_regions = int(info["num_regions"])
        num_records = int(info["num_records"])
        num_unique = int(info["num_unique_voxels"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid VXZM record counts") from error
    if (num_regions <= 0 or num_records <= 0 or num_unique <= 0 or
            num_unique > num_records):
        raise ValueError("Invalid VXZM record counts")
    _record_stride(info)

    sections = info.get("sections")
    required = ("region_svo", "region_counts", "region_offsets", "records")
    if not isinstance(sections, dict) or any(name not in sections for name in required):
        raise ValueError("VXZM header is missing required sections")
    ranges = []
    for name in required:
        ptr = sections[name]
        if (not isinstance(ptr, list) or len(ptr) != 2 or
                any(not isinstance(value, int) for value in ptr)):
            raise ValueError(f"Invalid VXZM section pointer for {name}")
        start, length = ptr
        if start < 0 or length < 0:
            raise ValueError(f"Invalid VXZM section pointer for {name}")
        absolute = header_end + start
        if file_size is not None and absolute + length > file_size:
            raise ValueError(f"Truncated VXZM section {name}")
        ranges.append((absolute, absolute + length, name))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"Overlapping VXZM sections {previous[2]} and {current[2]}"
            )


def _read_path_header(stream):
    """Read and validate a path-backed header without materializing payloads."""
    prefix = stream.read(9)
    if len(prefix) < 9:
        raise ValueError("Truncated VXZM header")
    if prefix[:4] != MAGIC:
        raise ValueError("Invalid VXZM file type")
    if prefix[4] != VERSION:
        raise ValueError(f"Unsupported VXZM version {prefix[4]}")
    header_end = struct.unpack(">I", prefix[5:9])[0]
    stream.seek(0, os.SEEK_END)
    file_size = stream.tell()
    if header_end < 9 or header_end > file_size:
        raise ValueError("Invalid VXZM header offset")
    stream.seek(9)
    header = prefix + stream.read(header_end - 9)
    if len(header) != header_end:
        raise ValueError("Truncated VXZM header")
    return _parse_header(header, file_size), file_size


def read_vxzm_info(file) -> Dict:
    """Read only the VXZM JSON header when ``file`` is a path."""
    if isinstance(file, (str, os.PathLike)):
        with open(file, "rb") as stream:
            info, _ = _read_path_header(stream)
        return info
    data = _read_file(file)
    return _parse_header(data, len(data))


def _section(data: bytes, info: Dict, name: str) -> bytes:
    ptr = info["sections"][name]
    start, length = int(ptr[0]), int(ptr[1])
    binary_start = int(info["binary_start"])
    if start < 0 or length < 0 or binary_start < 9:
        raise ValueError(f"Invalid VXZM section pointer for {name}")
    raw = data[binary_start + start:binary_start + start + length]
    if len(raw) != length:
        raise ValueError(f"Truncated VXZM section {name}")
    return _decompress(raw, info["compression"], info["compression_level"])


def _path_section(stream, file_size: int, info: Dict, name: str) -> bytes:
    """Read one compressed section from a seekable path-backed stream."""
    ptr = info["sections"][name]
    start, length = int(ptr[0]), int(ptr[1])
    absolute = int(info["binary_start"]) + start
    if (start < 0 or length < 0 or int(info["binary_start"]) < 9 or
            absolute < 0 or absolute + length > file_size):
        raise ValueError(f"Invalid VXZM section pointer for {name}")
    stream.seek(absolute)
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError(f"Truncated VXZM section {name}")
    return _decompress(raw, info["compression"], info["compression_level"])


def _record_stride(info: Dict) -> int:
    layout = info.get("record_layout")
    if not isinstance(layout, list) or not layout:
        raise ValueError("VXZM record layout must be non-empty")
    if any(not isinstance(x, dict) for x in layout):
        raise ValueError("Invalid VXZM record layout")
    names = [x.get("name") for x in layout]
    try:
        channels = [int(x.get("channels", 0)) for x in layout]
        dtype = np.dtype(info.get("record_dtype", "u1"))
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid VXZM record layout") from error
    if (any(not isinstance(name, str) or not name for name in names) or
            len(set(names)) != len(names) or any(ch <= 0 for ch in channels) or
            any(x.get("dtype", "u1") != "u1" for x in layout) or
            dtype != np.dtype("u1")):
        raise ValueError("Invalid VXZM record layout")
    layout_channels = dict(zip(names, channels))
    if layout_channels.get("base_color") != 3 or layout_channels.get("normal") != 3:
        raise ValueError("VXZM requires base_color[3] and surface normal[3]")
    return 3 + sum(channels)


def _count_unique_coordinates(coords: np.ndarray,
                              grid_size: np.ndarray) -> int:
    """Count XYZ rows exactly through collision-free packed uint64 keys.

    ``np.unique(coords, axis=0)`` internally constructs and sorts a structured
    array.  That dominates VXZM decode time for multi-million-record files.
    VXZM's validated grid is at most 262144 cells per axis, so the three
    coordinates fit losslessly in one uint64 value and can use NumPy's much
    faster one-dimensional unique path.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("VXZM coordinates must have shape [N,3]")
    bits = [max(1, (int(size) - 1).bit_length()) for size in grid_size]
    if sum(bits) > 64:
        raise ValueError("VXZM grid coordinates do not fit a uint64 key")
    keys = coords[:, 0].astype(np.uint64)
    keys |= coords[:, 1].astype(np.uint64) << np.uint64(bits[0])
    keys |= coords[:, 2].astype(np.uint64) << np.uint64(bits[0] + bits[1])
    return int(np.unique(keys).size)


def _decode_records_numpy(
    region_code: np.ndarray,
    counts: np.ndarray,
    raw_records: np.ndarray,
    block: np.ndarray,
    grid_size: np.ndarray,
):
    """Vectorized fallback for extensions built before native VXZM decode."""
    local = raw_records[:, :3]
    if np.any(local >= block):
        raise ValueError("VXZM local coordinate exceeds its region block")
    expanded_regions = np.repeat(
        region_code.astype(np.int32, copy=False),
        counts.astype(np.intp, copy=False),
        axis=0,
    )
    if expanded_regions.shape[0] != raw_records.shape[0]:
        raise ValueError("VXZM records were not fully consumed")
    coords = (expanded_regions * block.astype(np.int32) +
              local.astype(np.int32, copy=False))
    if len(local) > 1:
        same_region = np.all(
            expanded_regions[1:] == expanded_regions[:-1], axis=1,
        )
        out_of_order = (
            (local[1:, 0] < local[:-1, 0]) |
            ((local[1:, 0] == local[:-1, 0]) &
             (local[1:, 1] < local[:-1, 1])) |
            ((local[1:, 0] == local[:-1, 0]) &
             (local[1:, 1] == local[:-1, 1]) &
             (local[1:, 2] < local[:-1, 2]))
        )
        if np.any(same_region & out_of_order):
            raise ValueError("VXZM local coordinates are not canonically ordered")
    if np.any(coords < 0) or np.any(coords >= grid_size):
        raise ValueError("VXZM reconstructed coordinate lies outside grid_size")
    return coords, _count_unique_coordinates(coords, grid_size)


def _validate_region_arrays(info: Dict, counts: np.ndarray,
                            offsets: np.ndarray, record_stride: int):
    if counts.size == 0 or np.any(counts == 0):
        raise ValueError("VXZM must contain non-empty occupied regions")
    if offsets.size != counts.size + 1 or int(offsets[0]) != 0:
        raise ValueError("VXZM region offset/count length mismatch")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("VXZM region offsets must be monotonic")
    expected_sizes = counts.astype(np.uint64) * np.uint64(record_stride)
    if not np.array_equal(offsets[1:] - offsets[:-1], expected_sizes):
        raise ValueError("VXZM region offset does not match count")
    num_records = int(info.get("num_records", -1))
    if num_records < 0 or int(counts.sum(dtype=np.uint64)) != num_records:
        raise ValueError("VXZM region counts do not match record count")
    if int(info.get("num_regions", -1)) != counts.size:
        raise ValueError("VXZM region count does not match header")


def _decode_region_svo(region_svo_bytes: bytes, depth: int,
                       expected_leaves: int) -> torch.Tensor:
    """Decode a coarse SVO after validating every byte boundary.

    The native O-Voxel decoder assumes a trusted, complete preorder stream
    and therefore cannot report a truncated tree before dereferencing the next
    node. VXZM files are external inputs, so validate the tree in Python first
    and only pass the resulting Morton codes to the bounded z-order decoder.
    """
    if depth < 2:
        raise ValueError("VXZM coarse SVO depth must be at least 2")
    if expected_leaves <= 0:
        raise ValueError("VXZM coarse SVO must have at least one leaf")
    svo = memoryview(region_svo_bytes).cast("B")
    if len(svo) == 0:
        raise ValueError("VXZM coarse SVO is empty")
    codes = []
    ptr = 0

    def visit(level: int, prefix: int):
        nonlocal ptr
        if ptr >= len(svo):
            raise ValueError("Truncated VXZM coarse SVO")
        node = int(svo[ptr])
        ptr += 1
        if node == 0:
            raise ValueError("VXZM coarse SVO contains an empty node")
        if level == depth - 1:
            for child in range(8):
                if node & (1 << child):
                    codes.append((prefix << 3) | child)
                    if len(codes) > expected_leaves:
                        raise ValueError("VXZM coarse SVO has too many leaves")
            return
        for child in range(8):
            if node & (1 << child):
                visit(level + 1, (prefix << 3) | child)

    visit(0, 0)
    if ptr != len(svo):
        raise ValueError("VXZM coarse SVO contains trailing nodes")
    if len(codes) != expected_leaves:
        raise ValueError("VXZM coarse SVO leaf/count mismatch")
    morton = torch.tensor(codes, dtype=torch.int32)
    return decode_seq(morton).to(torch.int32)


def read_vxzm_regions(file):
    """Read the coarse-region index without decoding fine records.

    Returns ``(region_coord, counts, offsets, info)``.  ``region_coord`` is
    ordered exactly like the count/offset arrays and contains one row for
    every occupied coarse SVO leaf.  ``offsets`` are byte offsets into the
    records section and include the final sentinel.
    """
    if isinstance(file, (str, os.PathLike)):
        with open(file, "rb") as stream:
            info, file_size = _read_path_header(stream)
            region_svo_bytes = _path_section(stream, file_size, info, "region_svo")
            count_bytes = _path_section(stream, file_size, info, "region_counts")
            offset_bytes = _path_section(stream, file_size, info, "region_offsets")
    else:
        data = _read_file(file)
        info = _parse_header(data, len(data))
        region_svo_bytes = _section(data, info, "region_svo")
        count_bytes = _section(data, info, "region_counts")
        offset_bytes = _section(data, info, "region_offsets")
    region_resolution = int(info.get("region_resolution", 0))
    if (region_resolution < 4 or region_resolution > 1024 or
            region_resolution & (region_resolution - 1)):
        raise ValueError("VXZM region_resolution must be a power of two in [4, 1024]")
    counts = np.frombuffer(count_bytes, dtype="<u4")
    offsets = np.frombuffer(offset_bytes, dtype="<u8")
    _validate_region_arrays(info, counts, offsets, _record_stride(info))
    depth = region_resolution.bit_length() - 1
    region_code = _decode_region_svo(region_svo_bytes, depth, len(counts))
    return (
        region_code,
        torch.from_numpy(counts.copy()).to(torch.int64),
        torch.from_numpy(offsets.copy()).to(torch.int64),
        info,
    )


def read_vxzm(file, num_threads: int = -1, return_regions: bool = False):
    """Read a VXZM file and return ``(coord, attr)``.

    ``coord`` has one row per record and is intentionally allowed to contain
    duplicates.  No clustering or deduplication is performed while reading.

    When ``return_regions`` is true, the return value additionally includes
    ``(region_coord, region_counts, region_offsets, info)``.  This avoids a
    second file read/decompression in multi-surface training loaders that
    require both the records and their coarse-region index.  ``offsets`` are
    byte offsets into the uncompressed records section, including its final
    sentinel.
    """
    data = _read_file(file)
    info = _parse_header(data, len(data))
    if info.get("format") != "VXZM" or int(info.get("version", -1)) != VERSION:
        raise ValueError("Invalid VXZM header")

    region_svo_bytes = _section(data, info, "region_svo")
    counts = np.frombuffer(_section(data, info, "region_counts"), dtype="<u4")
    offsets = np.frombuffer(_section(data, info, "region_offsets"), dtype="<u8")
    record_bytes = _section(data, info, "records")
    region_resolution = int(info["region_resolution"])
    if (region_resolution < 4 or region_resolution > 1024 or
            region_resolution & (region_resolution - 1)):
        raise ValueError("VXZM region_resolution must be a power of two in [4, 1024]")
    depth = region_resolution.bit_length() - 1

    layout = info["record_layout"]
    channels = sum(int(x["channels"]) for x in layout)
    dtype = np.dtype(info.get("record_dtype", "u1"))
    record_size = _record_stride(info) * dtype.itemsize
    _validate_region_arrays(info, counts, offsets, record_size)
    if len(record_bytes) != int(offsets[-1]):
        raise ValueError("VXZM record section length mismatch")
    raw_records = np.frombuffer(record_bytes, dtype=dtype)
    if raw_records.size != int(info["num_records"]) * (3 + channels):
        raise ValueError("VXZM record count does not match payload")
    raw_records = raw_records.reshape(-1, 3 + channels)

    block = np.asarray(info["region_block_size"], dtype=np.int64)
    grid_size = np.asarray(info["grid_size"], dtype=np.int64)
    region_grid_size = np.asarray(info["region_grid_size"], dtype=np.int64)
    if (block.shape != (3,) or grid_size.shape != (3,) or region_grid_size.shape != (3,) or
            np.any(block <= 0) or np.any(grid_size <= 0) or np.any(region_grid_size <= 0) or
            np.any(block != np.ceil(grid_size / region_resolution).astype(np.int64)) or
            np.any(region_grid_size != np.ceil(grid_size / block).astype(np.int64))):
        raise ValueError("Invalid VXZM grid or region layout")
    if int(info["num_records"]) != raw_records.shape[0]:
        raise ValueError("VXZM record count does not match header")

    if hasattr(_C, "decode_vxzm_records_cpu"):
        # One writable copy supplies the native decoder and all attribute
        # slices.  Native code performs bounded SVO parsing, canonical-order
        # validation, region expansion, coordinate reconstruction, bounds
        # checks, and exact unique counting in a single pass.
        records_tensor = torch.from_numpy(raw_records.copy())
        coords, region_coord, unique_count = _C.decode_vxzm_records_cpu(
            torch.from_numpy(np.frombuffer(region_svo_bytes, dtype=np.uint8).copy()),
            torch.from_numpy(counts.astype(np.int64)),
            records_tensor,
            torch.from_numpy(grid_size.copy()),
            torch.from_numpy(block.copy()),
            depth,
        )
        region_code = region_coord.numpy().astype(np.int64, copy=False)
        if np.any(region_code < 0) or np.any(region_code >= region_grid_size):
            raise ValueError("Invalid VXZM grid or region layout")
        attr_source = records_tensor
    else:
        # Keep new Python code usable with a stale prebuilt extension.  It is
        # already an order of magnitude faster than the original per-region
        # implementation and retains collision-free unique-count validation.
        region_coord = _decode_region_svo(region_svo_bytes, depth, len(counts))
        region_code = region_coord.numpy().astype(np.int64)
        if np.any(region_code < 0) or np.any(region_code >= region_grid_size):
            raise ValueError("Invalid VXZM grid or region layout")
        coords_np, unique_count = _decode_records_numpy(
            region_code, counts, raw_records, block, grid_size,
        )
        coords = torch.from_numpy(coords_np)
        attr_source = torch.from_numpy(raw_records.copy())

    attr = {}
    ch = 3
    for x in layout:
        n = int(x["channels"])
        # Preserve the historical contiguous tensor contract.  The source is
        # row-interleaved, so each named channel group needs one compact copy.
        attr[x["name"]] = attr_source[:, ch:ch + n].contiguous()
        ch += n
    if int(unique_count) != int(info["num_unique_voxels"]):
        raise ValueError("VXZM unique voxel count does not match payload")
    coord = coords
    if return_regions:
        return (
            coord,
            attr,
            region_coord,
            torch.from_numpy(counts.copy()).to(torch.int64),
            torch.from_numpy(offsets.copy()).to(torch.int64),
            info,
        )
    return coord, attr


def write_vxzm(
    file,
    coord: torch.Tensor,
    attr: Dict[str, torch.Tensor],
    grid_size: Optional[Union[int, tuple, list]] = None,
    region_resolution: int = 256,
    compression: Literal["none", "deflate", "lzma", "zstd"] = "zstd",
    compression_level: Optional[int] = None,
    metadata: Optional[Dict] = None,
):
    """Write multi-surface voxel records to VXZM.

    ``coord`` may contain duplicates.  Attributes must be uint8 tensors with
    one row per record.  The three coordinate bytes stored in each region are
    local coordinates, while the coarse region tree stores the 256^3 region
    positions in Morton order.
    """
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("coord must have shape [N, 3]")
    if coord.dtype not in (torch.int32, torch.int64, torch.int16, torch.uint16):
        raise ValueError(f"coord must be an integer tensor, got {coord.dtype}")
    coord_np = coord.detach().cpu().numpy().astype(np.int64, copy=False)
    if np.any(coord_np < 0):
        raise ValueError("VXZM coordinates must be non-negative")
    n = coord_np.shape[0]
    names = list(attr.keys())
    if not names:
        raise ValueError("VXZM requires at least one attribute")
    for name, value in attr.items():
        if not isinstance(name, str) or not name:
            raise ValueError("VXZM attribute names must be non-empty strings")
        if value.ndim != 2 or value.shape[0] != n or value.shape[1] <= 0 or value.dtype != torch.uint8:
            raise ValueError(f"attribute {name} must be uint8 [N,C]")
    if ("base_color" not in attr or attr["base_color"].shape[1] != 3 or
            "normal" not in attr or attr["normal"].shape[1] != 3):
        raise ValueError("VXZM requires base_color[3] and surface normal[3]")
    if isinstance(grid_size, int):
        grid_size = [grid_size] * 3
    if grid_size is None:
        grid_size = (coord_np.max(axis=0) + 1).tolist() if n else [1, 1, 1]
    grid_size = np.asarray(grid_size, dtype=np.int64)
    if grid_size.shape != (3,) or np.any(grid_size <= 0):
        raise ValueError("grid_size must contain three positive values")
    if (region_resolution < 4 or region_resolution > 1024 or
            region_resolution & (region_resolution - 1)):
        raise ValueError("region_resolution must be a power of two in [4, 1024]")
    block = np.ceil(grid_size / region_resolution).astype(np.int64)
    if np.any(block > 256):
        raise ValueError(
            "region block does not fit uint8 local coordinates; increase "
            "region_resolution"
        )
    region_grid = np.ceil(grid_size / block).astype(np.int64)
    if np.any(region_grid > region_resolution):
        raise ValueError("region grid exceeds region_resolution")
    if n == 0:
        raise ValueError("VXZM requires at least one record")
    if np.any(coord_np >= grid_size):
        raise ValueError("VXZM coordinate lies outside grid_size")
    region = coord_np // block
    local = coord_np % block
    if np.any(local > 255):
        raise ValueError("local coordinate does not fit uint8; use a smaller region block")

    region_t = torch.from_numpy(region.astype(np.int32))
    unique_region, inverse = torch.unique(region_t, dim=0, sorted=True, return_inverse=True)
    unique_codes = encode_seq(unique_region)
    region_order = torch.argsort(unique_codes)
    unique_region = unique_region[region_order]
    # map each record to its sorted region index
    remap = torch.empty_like(region_order)
    remap[region_order] = torch.arange(region_order.numel(), dtype=region_order.dtype)
    rid = remap[inverse].numpy().astype(np.int64)
    order = np.lexsort((local[:, 2], local[:, 1], local[:, 0], rid))
    rid = rid[order]
    local = local[order]
    attr_np = {name: value.detach().cpu().numpy()[order] for name, value in attr.items()}
    counts64 = np.bincount(rid, minlength=len(unique_region)).astype(np.uint64)
    if np.any(counts64 > np.iinfo(np.uint32).max):
        raise ValueError("a VXZM region record count does not fit uint32")
    counts = counts64.astype("<u4")

    record_parts = [local.astype(np.uint8, copy=False)]
    for name in names:
        record_parts.append(attr_np[name].astype(np.uint8, copy=False))
    records = np.concatenate(record_parts, axis=1).tobytes()
    depth = int(round(np.log2(region_resolution)))
    region_codes = encode_seq(unique_region.to(torch.int32)).cpu()
    region_svo = _C.encode_sparse_voxel_octree_cpu(region_codes, depth).cpu().numpy().tobytes()
    record_stride_bytes = (3 + sum(int(v.shape[1]) for v in attr.values()))
    offsets = np.concatenate([[0], np.cumsum(counts.astype("<u8") * record_stride_bytes,
                                             dtype="<u8")])
    offsets_bytes = offsets.astype("<u8").tobytes()
    if compression not in DEFAULT_COMPRESION_LEVEL:
        raise ValueError(f"Invalid compression algorithm: {compression}")
    level = compression_level if compression_level is not None else DEFAULT_COMPRESION_LEVEL[compression]
    if not isinstance(level, int):
        raise ValueError("compression_level must be an integer")
    if compression == "deflate" and not (-1 <= level <= 9):
        raise ValueError("deflate compression_level must be in [-1, 9]")
    if compression == "lzma" and not (0 <= level <= 9):
        raise ValueError("lzma compression_level must be in [0, 9]")
    sections_raw = {
        "region_svo": region_svo,
        "region_counts": counts.tobytes(),
        "region_offsets": offsets_bytes,
        "records": records,
    }
    sections = {}
    binary = b""
    for key, raw in sections_raw.items():
        packed = _compress(raw, compression, level)
        sections[key] = [len(binary), len(packed)]
        binary += packed
    header = {
        "format": "VXZM",
        "version": VERSION,
        "grid_size": grid_size.tolist(),
        "region_resolution": int(region_resolution),
        "region_block_size": block.tolist(),
        "region_grid_size": region_grid.tolist(),
        "num_regions": int(len(unique_region)),
        "num_records": int(n),
        "num_unique_voxels": int(np.unique(coord_np, axis=0).shape[0]),
        "sections": sections,
        "compression": compression,
        "compression_level": int(level),
        "normal_source": "surface_authored_with_winding_guard",
        "record_dtype": "u1",
        "record_layout": [{"name": name, "dtype": "u1", "channels": int(value.shape[1])}
                          for name, value in attr.items()],
    }
    if metadata is not None:
        # Keep format-controlled keys authoritative and store caller details
        # in a namespaced object that remains JSON serializable.
        json.dumps(metadata)
        header["metadata"] = dict(metadata)
    # ``binary_start`` changes the encoded JSON length at most once when its
    # decimal width grows; iterate to a fixed point instead of assuming two
    # passes are sufficient.
    header["binary_start"] = 0
    while True:
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        binary_start = 9 + len(encoded)
        if header["binary_start"] == binary_start:
            break
        header["binary_start"] = binary_start
    blob = MAGIC + bytes([VERSION]) + struct.pack(">I", 9 + len(encoded)) + encoded + binary
    if isinstance(file, (str, os.PathLike)):
        # Complete the file atomically so an interrupted compression/write is
        # never mistaken for a valid cached asset by the data pipeline.
        target = os.fspath(file)
        parent = os.path.dirname(os.path.abspath(target))
        fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(target)}.",
                                        suffix=".tmp", dir=parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
    else:
        file.write(blob)


def vxzm_to_ply(
    vxzm_file,
    ply_file,
    aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    normal_offset: float = 0.0,
):
    """Export every VXZM record as an Open3D point with RGB and normal.

    ``normal_offset`` is expressed in fine-voxel units and affects only the
    visualization. It can make coincident normal clusters separately visible;
    stored VXZM coordinates are never changed.
    """
    import open3d as o3d

    # Materialize once so a file-like object is not consumed by the header
    # read before the payload read.  Paths are still handled without exposing
    # any file descriptor to the caller.
    data = _read_file(vxzm_file)
    info = read_vxzm_info(data)
    coord, attr = read_vxzm(data)
    if "base_color" not in attr or attr["base_color"].shape[1] != 3:
        raise ValueError("VXZM PLY export requires base_color[3]")
    if "normal" not in attr or attr["normal"].shape[1] != 3:
        raise ValueError("VXZM PLY export requires normal[3]")
    bounds = np.asarray(aabb, dtype=np.float64)
    if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
        raise ValueError("aabb must have shape [2,3] with positive extent")
    grid = np.asarray(info["grid_size"], dtype=np.float64)
    points = bounds[0] + (coord.numpy().astype(np.float64) + 0.5) / grid * (bounds[1] - bounds[0])
    normals = attr["normal"].numpy().astype(np.float64) / 255.0 * 2.0 - 1.0
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
    points += normals * float(normal_offset) / grid * (bounds[1] - bounds[0])
    colors = attr["base_color"].numpy().astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    cloud.normals = o3d.utility.Vector3dVector(normals)
    if not o3d.io.write_point_cloud(str(ply_file), cloud, write_ascii=False,
                                    compressed=False, print_progress=False):
        raise RuntimeError(f"Open3D failed to write {ply_file}")
    return coord, attr
