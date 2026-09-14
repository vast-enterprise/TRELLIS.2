"""Local checks for geometry/PBR coordinate intersection and CLI contracts."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import create_geo_pbr


def test_vxz_intersection_filters_both_coordinate_and_attribute_rows():
    pbr_coord = torch.tensor([[0, 0, 0], [1, 2, 3], [9, 9, 9]], dtype=torch.int32)
    pbr_attr = {"base_color": torch.arange(9, dtype=torch.uint8).reshape(3, 3)}
    geometry_coord = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    geometry_attr = {
        "vertices": torch.tensor([[10, 11, 12], [13, 14, 15]], dtype=torch.uint8),
        "intersected": torch.tensor([[3], [7]], dtype=torch.uint8),
    }

    pbr_coord, pbr_attr, geometry_coord, geometry_attr, common = create_geo_pbr.intersect_voxels(
        pbr_coord, pbr_attr, geometry_coord, geometry_attr
    )
    assert common == 1
    assert pbr_coord.tolist() == [[1, 2, 3]]
    assert pbr_attr["base_color"].tolist() == [[3, 4, 5]]
    assert geometry_coord.tolist() == [[1, 2, 3]]
    assert geometry_attr["intersected"].tolist() == [[3]]


def test_geometry_uses_repository_dual_grid_quantization():
    captured = {}

    def fake_convert(**kwargs):
        captured.update(kwargs)
        coord = torch.tensor([[2, 3, 4]], dtype=torch.int32)
        # Converter output is an AABB-relative position; these values produce
        # local offsets [0.5, 0.25, 0.75] in an 8^3 grid.
        vertices = torch.tensor([[(2 + 0.5) / 8, (3 + 0.25) / 8, (4 + 0.75) / 8]], dtype=torch.float32)
        intersected = torch.tensor([[True, False, True]])
        return coord, vertices, intersected

    fake_o_voxel = SimpleNamespace(
        convert=SimpleNamespace(mesh_to_flexible_dual_grid=fake_convert)
    )
    dump = {
        "objects": [{
            "vertices": np.asarray([
                [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0],
            ], dtype=np.float32),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
        }]
    }
    with mock.patch.dict(sys.modules, {"o_voxel": fake_o_voxel}):
        coord, attr = create_geo_pbr.convert_geometry(dump, 8)

    assert captured["grid_size"] == 8
    assert captured["aabb"] == [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    assert captured["face_weight"] == 1.0
    assert captured["boundary_weight"] == 0.2
    assert captured["regularization_weight"] == 1e-2
    assert torch.all(captured["vertices"] > -0.5)
    assert torch.all(captured["vertices"] < 0.5)
    assert torch.isclose(
        captured["vertices"].amax(dim=0).sub(captured["vertices"].amin(dim=0)).max(),
        torch.tensor(0.99999),
    )
    assert coord.tolist() == [[2, 3, 4]]
    assert attr["vertices"].tolist() == [[127, 63, 191]]
    assert attr["intersected"].tolist() == [[5]]


def test_bos_upload_uses_texture_and_geometry_subfolders():
    uploaded = []

    def download(_client, _bucket, _folder, _uuid, destination):
        destination.write_bytes(b"glb")

    def dump(_blender, _glb_path, dump_path):
        dump_path.write_bytes(b"dump")
        dump_path.with_name(dump_path.name + "_error.txt").write_bytes(b"debug")

    def write_pbr(path, coord, attr, _args):
        assert coord.tolist() == [[1, 2, 3]]
        assert attr["base_color"].tolist() == [[4, 5, 6]]
        path.write_bytes(b"pbr")

    def write_geometry(path, coord, attr, _args):
        assert coord.tolist() == [[1, 2, 3]]
        assert attr["intersected"].tolist() == [[5]]
        path.write_bytes(b"geometry")

    def upload(_client, bucket, folder, uuid, path, key):
        uploaded.append((bucket, folder, uuid, path.name, key))
        return key

    pbr_coord = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.int32)
    pbr_attr = {"base_color": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.uint8)}
    geo_coord = torch.tensor([[1, 2, 3], [7, 8, 9]], dtype=torch.int32)
    geo_attr = {
        "vertices": torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.uint8),
        "intersected": torch.tensor([[5], [3]], dtype=torch.uint8),
    }
    with tempfile.TemporaryDirectory() as directory, \
            mock.patch.object(create_geo_pbr, "_make_bos_client", return_value=object()), \
            mock.patch.object(create_geo_pbr, "download_glb", side_effect=download), \
            mock.patch.object(create_geo_pbr, "run_blender_dump", side_effect=dump), \
            mock.patch.object(create_geo_pbr, "_load_dump", return_value={"objects": [object()]}), \
            mock.patch.object(create_geo_pbr, "_convert_pbr_attributes", return_value=(pbr_coord, pbr_attr)), \
            mock.patch.object(create_geo_pbr, "convert_geometry", return_value=(geo_coord, geo_attr)), \
            mock.patch.object(create_geo_pbr, "_write_pbr", side_effect=write_pbr), \
            mock.patch.object(create_geo_pbr, "_write_geometry", side_effect=write_geometry), \
            mock.patch.object(create_geo_pbr, "upload_file", side_effect=upload):
        root = Path(directory)
        assert create_geo_pbr.main([
            "asset-uuid", "--output-dir", str(root), "--upload",
            "--upload-folder", "target/base",
        ]) == 0
        assert uploaded == [
            (
                create_geo_pbr._pbr.DEFAULT_OUTPUT_BUCKET,
                "target/base/texture", "asset-uuid", "asset-uuid.vxz",
                "target/base/texture/as/asset-uuid.vxz",
            ),
            (
                create_geo_pbr._pbr.DEFAULT_OUTPUT_BUCKET,
                "target/base/geometry", "asset-uuid", "asset-uuid.geometry.vxz",
                "target/base/geometry/as/asset-uuid.vxz",
            ),
        ]
        assert not (root / "asset-uuid.vxz").exists()
        assert not (root / "asset-uuid.geometry.vxz").exists()
        assert not (root / ".cache" / "asset-uuid.glb").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl_error.txt").exists()


def test_failed_upload_removes_all_local_artifacts():
    def download(_client, _bucket, _folder, _uuid, destination):
        destination.write_bytes(b"glb")

    def dump(_blender, _glb_path, dump_path):
        dump_path.write_bytes(b"dump")
        dump_path.with_name(dump_path.name + "_error.txt").write_bytes(b"debug")

    pbr_coord = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    pbr_attr = {"base_color": torch.tensor([[4, 5, 6]], dtype=torch.uint8)}
    geo_coord = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    geo_attr = {
        "vertices": torch.tensor([[7, 8, 9]], dtype=torch.uint8),
        "intersected": torch.tensor([[5]], dtype=torch.uint8),
    }
    with tempfile.TemporaryDirectory() as directory, \
            mock.patch.object(create_geo_pbr, "_make_bos_client", return_value=object()), \
            mock.patch.object(create_geo_pbr, "download_glb", side_effect=download), \
            mock.patch.object(create_geo_pbr, "run_blender_dump", side_effect=dump), \
            mock.patch.object(create_geo_pbr, "_load_dump", return_value={"objects": [object()]}), \
            mock.patch.object(create_geo_pbr, "_convert_pbr_attributes", return_value=(pbr_coord, pbr_attr)), \
            mock.patch.object(create_geo_pbr, "convert_geometry", return_value=(geo_coord, geo_attr)), \
            mock.patch.object(create_geo_pbr, "_write_pbr", side_effect=lambda path, *_: path.write_bytes(b"pbr")), \
            mock.patch.object(create_geo_pbr, "_write_geometry", side_effect=lambda path, *_: path.write_bytes(b"geo")), \
            mock.patch.object(create_geo_pbr, "upload_file", side_effect=RuntimeError("upload failed")):
        root = Path(directory)
        try:
            create_geo_pbr.main(["asset-uuid", "--output-dir", str(root), "--upload"])
        except RuntimeError as error:
            assert str(error) == "upload failed"
        else:  # pragma: no cover - assertion documents retry behavior
            raise AssertionError("failed upload was accepted")
        assert not (root / "asset-uuid.vxz").exists()
        assert not (root / "asset-uuid.geometry.vxz").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl_error.txt").exists()
        assert not (root / ".cache" / "asset-uuid.glb").exists()


def test_parser_rejects_different_resolutions():
    parser = create_geo_pbr.build_parser()
    try:
        create_geo_pbr._parse_args(parser, [
            "asset", "--resolution", "64", "--geometry-resolution", "32",
        ])
    except SystemExit as error:
        assert error.code == 2
    else:  # pragma: no cover - assertion documents the CLI contract
        raise AssertionError("different resolutions must not be intersected")


if __name__ == "__main__":
    test_vxz_intersection_filters_both_coordinate_and_attribute_rows()
    test_geometry_uses_repository_dual_grid_quantization()
    test_bos_upload_uses_texture_and_geometry_subfolders()
    test_failed_upload_removes_all_local_artifacts()
    test_parser_rejects_different_resolutions()
    print("create_geo_pbr tests passed")
