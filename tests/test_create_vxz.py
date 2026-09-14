#!/usr/bin/env python3
"""Command orchestration and cache-lifecycle tests for create_vxz.py."""

from __future__ import annotations

import tempfile
import sys
import subprocess
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import create_vxz


def test_normalize_dump_geometry_is_strict_centered_and_idempotent():
    dump = {
        "objects": [
            {
                "vertices": np.asarray([
                    [-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 1.0, 0.0],
                ], dtype=np.float64),
                "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            },
            {
                "vertices": np.asarray([
                    [-2.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 1.0, 0.0],
                ], dtype=np.float64),
                "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            },
        ]
    }

    returned = create_vxz.normalize_dump_geometry(dump)
    first = [obj["vertices"].copy() for obj in dump["objects"]]
    assert returned is dump
    vertices = np.concatenate(first, axis=0)
    assert vertices.dtype == np.float32
    assert np.all(vertices > -0.5)
    assert np.all(vertices < 0.5)
    assert np.allclose((vertices.min(axis=0) + vertices.max(axis=0)) / 2, 0)
    assert np.isclose((vertices.max(axis=0) - vertices.min(axis=0)).max(), 0.99999)

    # create_geo_pbr invokes the helper from both its PBR and geometry paths;
    # repeated application must not introduce cumulative shrinkage.
    assert create_vxz.normalize_dump_geometry(dump) is dump
    for obj, expected in zip(dump["objects"], first):
        assert np.array_equal(obj["vertices"], expected)


def test_convert_dump_normalizes_geometry_before_voxelization():
    captured = {}

    def convert(dump, **_kwargs):
        vertices = np.concatenate([obj["vertices"] for obj in dump["objects"]], axis=0)
        captured["vertices"] = vertices.copy()
        return (
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            {"base_color": torch.tensor([[1, 2, 3]], dtype=torch.uint8)},
        )

    def write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"vxz")

    fake_o_voxel = SimpleNamespace(
        convert=SimpleNamespace(blender_dump_to_volumetric_attr=convert),
        io=SimpleNamespace(write_vxz=write),
    )
    dump = {
        "objects": [{
            "vertices": np.asarray([
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, 0.5],
            ], dtype=np.float32),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
        }]
    }
    args = SimpleNamespace(
        resolution=64,
        mip_level_offset=0.0,
        verbose=False,
        timing=False,
        color_space="agx",
        add_emission=True,
        output_format="vxz",
        visualize=False,
        compression=None,
        compression_level=None,
    )

    with tempfile.TemporaryDirectory() as directory, \
            mock.patch.dict(sys.modules, {"o_voxel": fake_o_voxel}):
        path = Path(directory) / "box.vxz"
        result, count = create_vxz.convert_dump(dump, args, path)

    assert result == path
    assert count == 1
    assert np.all(captured["vertices"] > -0.5)
    assert np.all(captured["vertices"] < 0.5)
    assert np.isclose(np.ptp(captured["vertices"], axis=0).max(), 0.99999)


def test_blender_error_sidecar_is_reported_then_removed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        glb_path = root / "asset.glb"
        dump_path = root / "asset.pbr.pkl"

        def fail(_cmd, check):
            assert check
            dump_path.with_name(dump_path.name + "_error.txt").write_text("unsupported nodes")
            raise subprocess.CalledProcessError(1, "blender")

        with mock.patch.object(create_vxz.subprocess, "run", side_effect=fail):
            try:
                create_vxz.run_blender_dump("blender", glb_path, dump_path)
            except RuntimeError as error:
                assert "unsupported nodes" in str(error)
            else:  # pragma: no cover - assertion documents error propagation
                raise AssertionError("failed Blender dump was accepted")
        assert not dump_path.with_name(dump_path.name + "_error.txt").exists()


def _patch_pipeline(expected_suffix: str, uploaded: list):
    def download(_client, _bucket, _folder, uuid, destination):
        destination.write_bytes(b"glTF")
        return f"highpoly/{uuid[:2]}/{uuid}.glb"

    def dump(_blender, _glb, dump_path):
        dump_path.write_bytes(b"pickle")
        dump_path.with_name(dump_path.name + "_error.txt").write_bytes(b"debug")

    def convert(_dump, _args, output_path):
        assert output_path.suffix == expected_suffix
        output_path.write_bytes(b"result")
        return output_path, 7

    def upload(_client, bucket, folder, uuid, local_path, upload_key):
        assert local_path.is_file()
        uploaded.append((bucket, folder, uuid, local_path.name, upload_key))
        return upload_key or f"{folder}/{uuid[:2]}/{local_path.name}"

    stack = ExitStack()
    stack.enter_context(mock.patch.object(create_vxz, "_make_bos_client", return_value=object()))
    stack.enter_context(mock.patch.object(create_vxz, "download_glb", side_effect=download))
    stack.enter_context(mock.patch.object(create_vxz, "run_blender_dump", side_effect=dump))
    stack.enter_context(mock.patch.object(create_vxz, "_load_dump", return_value={"objects": [object()]}))
    stack.enter_context(mock.patch.object(create_vxz, "convert_dump", side_effect=convert))
    stack.enter_context(mock.patch.object(create_vxz, "upload_file", side_effect=upload))
    return stack


def test_upload_removes_vxz_and_cache():
    uploaded = []
    with tempfile.TemporaryDirectory() as directory, _patch_pipeline(".vxz", uploaded):
        root = Path(directory)
        assert create_vxz.main([
            "asset-uuid", "--output-dir", str(root), "--upload",
        ]) == 0
        assert uploaded == [(
            create_vxz.DEFAULT_OUTPUT_BUCKET,
            create_vxz.DEFAULT_OUTPUT_FOLDER,
            "asset-uuid",
            "asset-uuid.vxz",
            None,
        )]
        assert not (root / "asset-uuid.vxz").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl_error.txt").exists()
        assert not (root / ".cache" / "asset-uuid.glb").exists()


def test_visualize_writes_only_ply():
    uploaded = []
    with tempfile.TemporaryDirectory() as directory, _patch_pipeline(".ply", uploaded):
        root = Path(directory)
        assert create_vxz.main([
            "--uuid", "asset-uuid", "--output-dir", str(root),
            "--output-format", "vxzm", "--visualize",
        ]) == 0
        assert (root / "asset-uuid.ply").read_bytes() == b"result"
        assert not (root / "asset-uuid.vxz").exists()
        assert not (root / "asset-uuid.vxzm").exists()
        # Without --upload the PBR dump remains reusable, while the downloaded
        # source GLB follows the reference sampler and is always removed.
        assert (root / ".cache" / "asset-uuid.pbr.pkl").is_file()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl_error.txt").exists()
        assert not (root / ".cache" / "asset-uuid.glb").exists()
        assert not uploaded


def test_visualize_upload_removes_all_local_artifacts():
    uploaded = []
    with tempfile.TemporaryDirectory() as directory, _patch_pipeline(".ply", uploaded):
        root = Path(directory)
        assert create_vxz.main([
            "asset-uuid", "--output-dir", str(root),
            "--visualize", "--upload",
        ]) == 0
        assert not (root / "asset-uuid.ply").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl_error.txt").exists()
        assert not (root / ".cache" / "asset-uuid.glb").exists()
        assert uploaded[0][3] == "asset-uuid.ply"


def test_failed_upload_removes_all_local_artifacts():
    uploaded = []
    with tempfile.TemporaryDirectory() as directory, _patch_pipeline(".vxzm", uploaded):
        root = Path(directory)
        with mock.patch.object(create_vxz, "upload_file", side_effect=RuntimeError("upload failed")):
            try:
                create_vxz.main([
                    "asset-uuid", "--output-dir", str(root),
                    "--output-format", "vxzm", "--upload",
                ])
                raise AssertionError("failed upload was accepted")
            except RuntimeError as error:
                assert str(error) == "upload failed"
        assert not (root / "asset-uuid.vxzm").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl").exists()
        assert not (root / ".cache" / "asset-uuid.pbr.pkl_error.txt").exists()
        assert not (root / ".cache" / "asset-uuid.glb").exists()


def main():
    test_normalize_dump_geometry_is_strict_centered_and_idempotent()
    test_convert_dump_normalizes_geometry_before_voxelization()
    test_blender_error_sidecar_is_reported_then_removed()
    test_upload_removes_vxz_and_cache()
    test_visualize_writes_only_ply()
    test_visualize_upload_removes_all_local_artifacts()
    test_failed_upload_removes_all_local_artifacts()
    print("create_vxz orchestration tests passed")


if __name__ == "__main__":
    main()
