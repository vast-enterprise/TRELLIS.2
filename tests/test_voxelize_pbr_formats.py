#!/usr/bin/env python3
"""Synthetic data-pipeline regression for VXZ/VXZM `both` output."""
from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "o-voxel"))
sys.path.insert(0, str(ROOT))

import o_voxel
from data_toolkit import voxelize_pbr
from data_toolkit import build_metadata


def fixture_dump():
    return {
        "surface_normal_source": "blender_authored_corner_world_v1",
        "materials": [{
            "baseColorFactor": [0.25, 0.5, 0.75, 1.0],
            "baseColorTexture": None,
            "metallicFactor": 0.1,
            "metallicTexture": None,
            "roughnessFactor": 0.8,
            "roughnessTexture": None,
            "emissiveFactor": [0.0, 0.0, 0.0],
            "emissiveTexture": None,
            "emissionStrength": 0.0,
            "alphaFactor": 1.0,
            "alphaTexture": None,
            "alphaMode": "OPAQUE",
            "alphaCutoff": 0.5,
            "shaderType": "Principled",
        }],
        "objects": [{
            "vertices": np.asarray([
                [-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [-0.4, 0.4, 0.0],
            ], dtype=np.float32),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
            "normals": np.asarray([[[0.0, 0.0, 1.0]] * 3], dtype=np.float32),
            "uvs": np.zeros((1, 3, 2), dtype=np.float32),
            "mat_ids": np.asarray([0], dtype=np.int32),
        }],
    }


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sha = "synthetic"
        (root / "pbr_dumps").mkdir()
        (root / "pbr_voxels_8").mkdir()
        with (root / "pbr_dumps" / f"{sha}.pickle").open("wb") as stream:
            pickle.dump(fixture_dump(), stream)

        stale_dump = fixture_dump()
        stale_dump.pop("surface_normal_source")
        for convert, kwargs in (
            (o_voxel.convert.blender_dump_to_volumetric_attr_multi, {}),
            (o_voxel.convert.blender_dump_to_volumetric_attr,
             {"multi_surface": True}),
        ):
            try:
                convert(
                    stale_dump, grid_size=8,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    color_space="linear", **kwargs,
                )
                raise AssertionError("legacy dump without normal provenance was accepted")
            except ValueError as error:
                assert "authored corner normals" in str(error)

        voxelize_pbr.opt = SimpleNamespace(
            resolution=[8], output_format="both", add_emission=False,
            color_space="linear", cluster_angle_degrees=15.0,
            region_resolution=4,
            max_records_per_voxel=0, max_total_records=0,
        )
        result = voxelize_pbr._pbr_voxelize(
            None, {"sha256": sha}, str(root), str(root)
        )
        assert "error" not in result, result
        assert result["pbr_voxelized_8"] is True
        assert result["pbrm_voxelized_8"] is True
        vxz = root / "pbr_voxels_8" / f"{sha}.vxz"
        vxzm = root / "pbr_voxels_8" / f"{sha}.vxzm"
        assert vxz.is_file() and vxzm.is_file()
        assert o_voxel.io.read_vxz_info(str(vxz))["num_voxel"] > 0
        info = o_voxel.io.read_vxzm_info(vxzm)
        assert info["num_records"] >= info["num_unique_voxels"] > 0
        assert info["region_block_size"] == [2, 2, 2]
        assert info["normal_source"] == "surface_authored_with_winding_guard"
        assert {entry["name"] for entry in info["record_layout"]} == {
            "base_color", "metallic", "roughness", "emissive", "alpha", "normal",
        }
        assert result["pbrm_config_8"] == voxelize_pbr._expected_vxzm_config(8)
        assert voxelize_pbr._vxzm_config_from_info(info) is not None

        # A cached file from an older record schema (for example one without
        # the separately retained emissive field) must be regenerated.
        old_layout_info = dict(info)
        old_layout_info["record_layout"] = [
            entry for entry in info["record_layout"]
            if entry["name"] != "emissive"
        ]
        assert voxelize_pbr._vxzm_config_from_info(old_layout_info) != \
            voxelize_pbr._expected_vxzm_config(8)

        cached_metadata = pd.DataFrame([result])
        assert len(voxelize_pbr._select_pending_metadata(cached_metadata)) == 0
        stale_metadata = cached_metadata.copy()
        stale_metadata.loc[0, "pbrm_config_8"] = "stale"
        assert len(voxelize_pbr._select_pending_metadata(stale_metadata)) == 1

        # Cached VXZM output is configuration-specific. Changing an option
        # must regenerate it rather than silently reusing incompatible data.
        voxelize_pbr.opt.cluster_angle_degrees = 12.0
        regenerated = voxelize_pbr._pbr_voxelize(
            None, {"sha256": sha}, str(root), str(root)
        )
        assert "error" not in regenerated, regenerated
        assert o_voxel.io.read_vxzm_info(vxzm)["metadata"]["cluster_angle_degrees"] == 12.0
        assert regenerated["pbrm_config_8"] == voxelize_pbr._expected_vxzm_config(8)

        # VXZ and VXZM status must be emitted in one combined metadata row.
        # Separate partial rows can overwrite each other during metadata merge.
        combined_columns = {
            "pbr_voxelized_8", "num_pbr_voxels_8", "pbrm_voxelized_8",
            "num_pbrm_records_8", "num_pbrm_unique_voxels_8", "pbrm_config_8",
        }
        assert combined_columns <= set(regenerated)

        parts = root / "pbr_voxels_8" / "new_records"
        parts.mkdir(exist_ok=True)
        part_path = voxelize_pbr._write_metadata_part(
            pd.DataFrame([regenerated]), 8, str(root), rank=0,
        )
        assert part_path is not None
        part = pd.read_csv(part_path)
        expected_columns = {
            "sha256", "pbr_voxelized", "num_pbr_voxels",
            "pbrm_voxelized", "num_pbrm_records",
            "num_pbrm_unique_voxels", "pbrm_config",
        }
        assert expected_columns <= set(part.columns)
        merged = build_metadata.update_metadata(
            str(root / "pbr_voxels_8"),
            SimpleNamespace(from_merged_records=False, record_start=0),
        )
        assert expected_columns - {"sha256"} <= set(merged.columns)
        assert merged.loc[sha, "pbr_voxelized"]
        assert merged.loc[sha, "pbrm_voxelized"]

        # Every CLI output mode has distinct worker behavior: single-format
        # requests must not create the other suffix, while ``both`` above
        # creates and reports both in one metadata row.
        for output_format, expected_suffix in (("vxz", ".vxz"),
                                                ("vxzm", ".vxzm")):
            format_root = root / f"only_{output_format}"
            (format_root / "pbr_voxels_8").mkdir(parents=True)
            voxelize_pbr.opt.output_format = output_format
            result_one = voxelize_pbr._pbr_voxelize(
                None, {"sha256": sha}, str(root), str(format_root)
            )
            assert "error" not in result_one, result_one
            assert (format_root / "pbr_voxels_8" /
                    f"{sha}{expected_suffix}").is_file()
            other_suffix = ".vxzm" if expected_suffix == ".vxz" else ".vxz"
            assert not (format_root / "pbr_voxels_8" /
                        f"{sha}{other_suffix}").exists()

        voxelize_pbr.opt.output_format = "both"

        # Existing valid outputs should be inspected and reused without
        # loading or rewriting the source dump.
        (root / "pbr_dumps" / f"{sha}.pickle").unlink()
        reused = voxelize_pbr._pbr_voxelize(
            None, {"sha256": sha}, str(root), str(root)
        )
        assert "error" not in reused, reused
        assert reused["num_pbr_voxels_8"] == result["num_pbr_voxels_8"]
        assert reused["num_pbrm_records_8"] == result["num_pbrm_records_8"]
    print("VXZ/VXZM pipeline format test passed")


if __name__ == "__main__":
    main()
