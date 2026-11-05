"""Regression for glTF COLOR_0 dump and voxel sampling support."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "o-voxel"))
from o_voxel.convert.volumetic_attr import (
    blender_dump_to_volumetric_attr,
    blender_dump_to_volumetric_attr_multi,
)


def _dump(with_vertex_color: bool, with_texture: bool = False):
    material = {
        "baseColorFactor": [0.5, 0.25, 0.75, 1.0],
        "baseColorTexture": None,
        "baseColorUseVertexColor": with_vertex_color,
        "metallicFactor": 0.0,
        "metallicTexture": None,
        "roughnessFactor": 1.0,
        "roughnessTexture": None,
        "emissiveFactor": [0.0, 0.0, 0.0],
        "emissiveTexture": None,
        "emissionStrength": 0.0,
        "alphaFactor": 1.0,
        "alphaTexture": None,
        "alphaMode": "BLEND",
        "alphaCutoff": 0.5,
    }
    if with_texture:
        image = Image.new("RGBA", (1, 1), (255, 255, 255, 255))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        material["baseColorTexture"] = {
            "image": stream.getvalue(),
            "interpolation": "Closest",
            "extension": "REPEAT",
            "color_space": "sRGB",
        }
    obj = {
        "vertices": np.asarray(
            [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [-0.4, 0.4, 0.0]],
            dtype=np.float32,
        ),
        "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
        "normals": np.asarray([[[0.0, 0.0, 1.0]] * 3], dtype=np.float32),
        "uvs": None,
        "mat_ids": np.asarray([0], dtype=np.int32),
    }
    if with_vertex_color:
        obj["vertex_colors"] = np.asarray(
            [[[1.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, 0.6], [0.0, 0.0, 1.0, 1.0]]],
            dtype=np.float32,
        )
    return {
        "surface_normal_source": "blender_authored_corner_world_v1",
        "materials": [material],
        "objects": [obj],
    }


def test_vertex_color_is_interpolated_and_multiplied_by_factor():
    kwargs = dict(grid_size=8, aabb=[[-0.5] * 3, [0.5] * 3], color_space="linear", add_emission=False)
    plain_coord, plain = blender_dump_to_volumetric_attr(_dump(False), **kwargs)
    color_coord, colored = blender_dump_to_volumetric_attr(_dump(True), **kwargs)
    assert torch.equal(plain_coord, color_coord)
    assert torch.all(colored["base_color"] <= plain["base_color"])
    assert torch.any(colored["base_color"] < plain["base_color"])
    assert torch.all(colored["alpha"] <= plain["alpha"])
    assert torch.any(colored["alpha"] < plain["alpha"])


def test_vertex_color_is_applied_after_base_color_texture_and_factor():
    kwargs = dict(grid_size=8, aabb=[[-0.5] * 3, [0.5] * 3], color_space="linear", add_emission=False)
    plain_coord, plain = blender_dump_to_volumetric_attr(_dump(False, True), **kwargs)
    color_coord, colored = blender_dump_to_volumetric_attr(_dump(True, True), **kwargs)
    assert torch.equal(plain_coord, color_coord)
    assert torch.all(colored["base_color"] <= plain["base_color"])
    assert torch.any(colored["base_color"] < plain["base_color"])


def test_vertex_color_supports_vxzm_and_does_not_leak_to_plain_calls():
    kwargs = dict(grid_size=8, aabb=[[-0.5] * 3, [0.5] * 3], color_space="linear", add_emission=False)
    coord, attr = blender_dump_to_volumetric_attr_multi(
        _dump(True), cluster_angle_degrees=15.0,
        max_records_per_voxel=0, max_total_records=0, **kwargs,
    )
    assert len(coord) == len(attr["base_color"])
    assert {"base_color", "normal"} <= set(attr)
    plain = _dump(False)
    plain_coord, plain_attr = blender_dump_to_volumetric_attr(plain, **kwargs)
    assert len(plain_coord) == len(plain_attr["base_color"])


if __name__ == "__main__":
    test_vertex_color_is_interpolated_and_multiplied_by_factor()
    print("vertex color test passed")
