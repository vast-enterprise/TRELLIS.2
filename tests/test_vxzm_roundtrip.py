"""Format-level tests for duplicate-coordinate VXZM records."""
from pathlib import Path
import importlib
import io
import json
import struct
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "o-voxel"))

import o_voxel

vxzm_module = importlib.import_module("o_voxel.io.vxzm")


def main():
    # Two records intentionally share a high-resolution voxel coordinate but
    # have opposite normals.  The region boundary and a second region are
    # covered as well.
    coord = torch.tensor([
        [0, 0, 0], [0, 0, 0], [3, 3, 3], [4, 0, 0], [1023, 1023, 1023]
    ], dtype=torch.int32)
    attr = {
        "base_color": torch.tensor([
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [12, 34, 56], [1, 2, 3]
        ], dtype=torch.uint8),
        "metallic": torch.tensor([[0], [63], [127], [191], [255]], dtype=torch.uint8),
        "roughness": torch.tensor([[255], [191], [127], [63], [0]], dtype=torch.uint8),
        "emissive": torch.tensor([
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
        ], dtype=torch.uint8),
        "alpha": torch.tensor([[255], [192], [128], [64], [0]], dtype=torch.uint8),
        "normal": torch.tensor([
            [127, 127, 255], [127, 127, 0], [255, 127, 127], [0, 255, 127], [127, 255, 127]
        ], dtype=torch.uint8),
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sample.vxzm"
        # The writer canonicalizes records by coarse Morton region and local
        # coordinate.  Deliberately shuffle the input to verify that each
        # duplicate coordinate remains paired with its own attributes.
        permutation = torch.tensor([4, 1, 3, 0, 2])
        shuffled_coord = coord[permutation]
        shuffled_attr = {name: value[permutation] for name, value in attr.items()}
        o_voxel.io.write_vxzm(path, shuffled_coord, shuffled_attr, grid_size=1024,
                               region_resolution=256, compression="none",
                               metadata={"cluster_angle_degrees": 15.0})
        got_coord, got_attr = o_voxel.io.read_vxzm(path)
        attr_names = tuple(sorted(shuffled_attr))
        expected = sorted((
            tuple(c.tolist()),
            *(tuple(shuffled_attr[name][i].tolist()) for name in attr_names),
        ) for i, c in enumerate(shuffled_coord))
        actual = sorted((
            tuple(c.tolist()),
            *(tuple(got_attr[name][i].tolist()) for name in attr_names),
        ) for i, c in enumerate(got_coord))
        assert actual == expected, (actual, expected)

        # The public PLY helper must also work when handed an already-open
        # file-like object (its header and payload reads share one buffer).
        blob = path.read_bytes()
        info_from_stream = o_voxel.io.read_vxzm_info(io.BytesIO(blob))
        stream_coord, stream_attr = o_voxel.io.read_vxzm(io.BytesIO(blob))
        assert info_from_stream == o_voxel.io.read_vxzm_info(path)
        assert torch.equal(stream_coord, got_coord)
        for name in shuffled_attr:
            assert torch.equal(stream_attr[name], got_attr[name]), name
        info = o_voxel.io.read_vxzm_info(path)
        assert info["num_records"] == 5
        assert info["num_unique_voxels"] == 4
        assert info["region_block_size"] == [4, 4, 4]
        assert info["normal_source"] == "surface_authored_with_winding_guard"
        assert info["metadata"]["cluster_angle_degrees"] == 15.0
        region_coord, region_counts, region_offsets, region_info = \
            o_voxel.io.read_vxzm_regions(path)
        assert region_info == info
        assert len(region_coord) == info["num_regions"]
        assert int(region_counts.sum()) == info["num_records"]
        expected_stride = 3 + sum(value.shape[1] for value in attr.values())
        assert int(region_offsets[-1]) == info["num_records"] * expected_stride

        # Generic extension dispatch must preserve the same multi-surface
        # records; callers should not need a VXZM-specific code path.
        generic_path = Path(td) / "generic.vxzm"
        o_voxel.io.write(
            str(generic_path), shuffled_coord, shuffled_attr,
            grid_size=1024, region_resolution=256, compression="none",
        )
        generic_coord, generic_attr = o_voxel.io.read(str(generic_path))
        assert torch.equal(generic_coord, got_coord)
        for name in shuffled_attr:
            assert torch.equal(generic_attr[name], got_attr[name]), name

        # Non-divisible, anisotropic grids still reconstruct exact global
        # coordinates from their per-axis ceil-sized region blocks.
        odd_coord = torch.tensor([
            [0, 0, 0], [4, 6, 8], [9, 10, 12], [9, 10, 12],
        ], dtype=torch.int32)
        odd_attr = {
            "base_color": torch.tensor([
                [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
            ], dtype=torch.uint8),
            "normal": torch.tensor([
                [127, 127, 255], [255, 127, 127],
                [127, 255, 127], [127, 127, 0],
            ], dtype=torch.uint8),
        }
        odd_path = Path(td) / "odd.vxzm"
        o_voxel.io.write_vxzm(
            odd_path, odd_coord, odd_attr, grid_size=[10, 11, 13],
            region_resolution=4, compression="none",
        )
        odd_got_coord, odd_got_attr = o_voxel.io.read_vxzm(odd_path)
        odd_info = o_voxel.io.read_vxzm_info(odd_path)
        assert odd_info["region_block_size"] == [3, 3, 4]
        assert odd_info["region_grid_size"] == [4, 4, 4]
        odd_expected = sorted(
            (tuple(c.tolist()), tuple(odd_attr["base_color"][i].tolist()),
             tuple(odd_attr["normal"][i].tolist()))
            for i, c in enumerate(odd_coord)
        )
        odd_actual = sorted(
            (tuple(c.tolist()), tuple(odd_got_attr["base_color"][i].tolist()),
             tuple(odd_got_attr["normal"][i].tolist()))
            for i, c in enumerate(odd_got_coord)
        )
        assert odd_actual == odd_expected

        # Corrupt/truncated data must fail validation instead of reaching the
        # native SVO decoder with an invalid buffer.
        try:
            o_voxel.io.read_vxzm(blob[:8])
            raise AssertionError("truncated header was accepted")
        except ValueError as error:
            assert "Truncated" in str(error)

        # A structurally valid file whose count table disagrees with the
        # record byte offsets must be rejected before record reconstruction.
        raw = bytearray(blob)
        header_end = struct.unpack(">I", raw[5:9])[0]
        header = json.loads(raw[9:header_end].decode("utf-8"))
        assert header["compression"] == "none"
        count_start = header["binary_start"] + header["sections"]["region_counts"][0]
        first_count = struct.unpack("<I", raw[count_start:count_start + 4])[0]
        raw[count_start:count_start + 4] = struct.pack("<I", first_count + 1)
        try:
            o_voxel.io.read_vxzm(bytes(raw))
            raise AssertionError("inconsistent region count was accepted")
        except ValueError as error:
            assert "offset does not match count" in str(error)

        wrong_unique = bytearray(blob)
        wrong_unique_header = dict(header)
        wrong_unique_header["num_unique_voxels"] += 1
        encoded = json.dumps(wrong_unique_header, separators=(",", ":")).encode("utf-8")
        assert len(encoded) == header_end - 9
        wrong_unique[9:header_end] = encoded
        try:
            o_voxel.io.read_vxzm(bytes(wrong_unique))
            raise AssertionError("incorrect unique voxel count was accepted")
        except ValueError as error:
            assert "unique voxel count" in str(error)

        # The records section is canonical within each coarse region. Native
        # decode uses adjacent local XYZ values for exact unique counting, so
        # reject a payload that breaks this format invariant rather than
        # silently accepting a misleading header count.
        unordered = bytearray(blob)
        record_start = header["binary_start"] + header["sections"]["records"][0]
        record_stride = 3 + sum(value.shape[1] for value in attr.values())
        # The first region contains local [0,0,0], [0,0,0], and [3,3,3].
        first = bytes(unordered[record_start:record_start + record_stride])
        third_start = record_start + 2 * record_stride
        third = bytes(unordered[third_start:third_start + record_stride])
        unordered[record_start:record_start + record_stride] = third
        unordered[third_start:third_start + record_stride] = first
        try:
            o_voxel.io.read_vxzm(bytes(unordered))
            raise AssertionError("non-canonical local coordinate order was accepted")
        except ValueError as error:
            assert "canonically ordered" in str(error), str(error)

        # The vectorized NumPy fallback and compiled path must reconstruct the
        # same record order. This keeps source updates usable before an
        # extension rebuild without changing observable decode semantics.
        region_svo = vxzm_module._section(blob, header, "region_svo")
        counts = np.frombuffer(
            vxzm_module._section(blob, header, "region_counts"), dtype="<u4",
        )
        record_bytes = vxzm_module._section(blob, header, "records")
        raw_records = np.frombuffer(record_bytes, dtype=np.uint8).reshape(
            -1, record_stride,
        )
        fallback_region = vxzm_module._decode_region_svo(
            region_svo, int(header["region_resolution"]).bit_length() - 1,
            len(counts),
        ).numpy()
        fallback_coord, fallback_unique = vxzm_module._decode_records_numpy(
            fallback_region, counts, raw_records,
            np.asarray(header["region_block_size"], dtype=np.int64),
            np.asarray(header["grid_size"], dtype=np.int64),
        )
        assert np.array_equal(fallback_coord, got_coord.numpy())
        assert fallback_unique == header["num_unique_voxels"]

        # The native SVO decoder assumes trusted bytes. VXZM validates the
        # preorder tree first so truncated or trailing nodes raise Python
        # errors rather than reading beyond the section and crashing.
        svo_start = header["binary_start"] + header["sections"]["region_svo"][0]
        svo_length = header["sections"]["region_svo"][1]
        assert svo_length > 1
        for replacement, expected_message in (
            (blob[svo_start:svo_start + svo_length - 1], "Truncated VXZM coarse SVO"),
            (blob[svo_start:svo_start + svo_length] + b"\x01",
             "trailing nodes"),
        ):
            corrupted = bytearray(blob)
            delta = len(replacement) - svo_length
            corrupted[svo_start:svo_start + svo_length] = replacement
            bad_header = dict(header)
            bad_header["sections"] = {
                name: list(pointer) for name, pointer in header["sections"].items()
            }
            bad_header["sections"]["region_svo"][1] = len(replacement)
            for section_name in ("region_counts", "region_offsets", "records"):
                bad_header["sections"][section_name][0] += delta
            # Rebuilding a fixed-size header is unnecessary here: the changed
            # pointer values retain their decimal widths for this fixture.
            encoded = json.dumps(bad_header, separators=(",", ":")).encode("utf-8")
            assert len(encoded) == header_end - 9
            corrupted[9:header_end] = encoded
            try:
                o_voxel.io.read_vxzm(bytes(corrupted))
                raise AssertionError("invalid coarse SVO was accepted")
            except ValueError as error:
                assert expected_message in str(error), str(error)

        for bad_resolution in (1, 2, 3, 6, 2048):
            try:
                o_voxel.io.write_vxzm(
                    io.BytesIO(), coord[:1], {name: value[:1] for name, value in attr.items()},
                    grid_size=1024, region_resolution=bad_resolution,
                )
                raise AssertionError(f"invalid region resolution accepted: {bad_resolution}")
            except ValueError as error:
                assert "region_resolution" in str(error)

        # Three uint8 local-coordinate fields can represent at most a
        # 256-wide block on each axis, regardless of which positions happen
        # to be occupied in this particular file.
        try:
            o_voxel.io.write_vxzm(
                io.BytesIO(), coord[:1], {name: value[:1] for name, value in attr.items()},
                grid_size=1025, region_resolution=4, compression="none",
            )
            raise AssertionError("unrepresentable uint8 region block was accepted")
        except ValueError as error:
            assert "uint8 local coordinates" in str(error)

        try:
            o_voxel.io.write_vxzm(
                io.BytesIO(), coord[:1], {"base_color": attr["base_color"][:1]},
                grid_size=1024, region_resolution=256,
            )
            raise AssertionError("VXZM without surface normals was accepted")
        except ValueError as error:
            assert "surface normal" in str(error)
    print("VXZM round-trip test passed")


if __name__ == "__main__":
    main()
