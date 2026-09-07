#!/usr/bin/env python3
"""Native normal-clustering regression for multi-surface voxels."""
from pathlib import Path
import math
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "o-voxel"))
from o_voxel import _C


def triangle_from_normal(normal):
    """Build a centered triangle whose signed geometric normal is ``normal``."""
    n = torch.tensor(normal, dtype=torch.float32)
    n = n / torch.linalg.vector_norm(n)
    reference = torch.tensor([0.0, 1.0, 0.0])
    if abs(float(torch.dot(reference, n))) > 0.99:
        reference = torch.tensor([1.0, 0.0, 0.0])
    tangent = torch.linalg.cross(reference, n)
    tangent = tangent / torch.linalg.vector_norm(tangent)
    bitangent = torch.linalg.cross(n, tangent)
    center = torch.tensor([0.5, 0.5, 0.5])
    return torch.stack([
        center - 0.4 * tangent - 0.4 * bitangent,
        center + 0.4 * tangent - 0.4 * bitangent,
        center - 0.4 * tangent + 0.4 * bitangent,
    ])


def run(surface_normals, multi_surface=True, normal_texture=None,
        authored_normals=None, cluster_angle_degrees=15.0):
    resolution = 8
    vertices = torch.stack([triangle_from_normal(n) for n in surface_normals])
    if authored_normals is None:
        authored_normals = surface_normals
    normals = torch.stack([
        torch.tensor(n, dtype=torch.float32).repeat(3, 1)
        for n in authored_normals
    ])
    uvs = torch.zeros((len(surface_normals), 3, 2), dtype=torch.float32)
    empty = torch.empty(0, dtype=torch.float32)
    normal_texture = empty if normal_texture is None else normal_texture
    fn = (_C.textured_mesh_to_volumetric_attr_multi_cpu if multi_surface
          else _C.textured_mesh_to_volumetric_attr_cpu)
    args = [
        torch.full((3,), 1.0 / resolution, dtype=torch.float32),
        torch.tensor([[0, 0, 0], [resolution] * 3], dtype=torch.int32),
        vertices, normals, uvs, torch.zeros(len(surface_normals), dtype=torch.int32),
        [torch.ones(3)], [empty], [0], [0],
        [0.0], [empty], [0], [0], [1.0], [empty], [0], [0],
        [torch.zeros(3)], [empty], [0], [0],
        [0], [0.5], [1.0], [empty], [0], [0],
    ]
    if multi_surface:
        args.extend([0.0, False, False, cluster_angle_degrees, 0, 0])
    else:
        args.extend([[normal_texture], [0], [0], 0.0, False, False])
    return fn(*args)


def counts(coord):
    _, c = torch.unique(coord, dim=0, return_counts=True)
    return sorted(c.tolist())


def run_geometry(triangles, normals_per_triangle, multi_surface=True,
                 base_color_factors=None, material_ids=None):
    """Run the native path on explicitly different triangle geometries."""
    resolution = 8
    vertices = torch.tensor(triangles, dtype=torch.float32)
    normals = torch.tensor(normals_per_triangle, dtype=torch.float32)
    uvs = torch.zeros((len(triangles), 3, 2), dtype=torch.float32)
    empty = torch.empty(0, dtype=torch.float32)
    if base_color_factors is None:
        base_color_factors = [[1.0, 1.0, 1.0]]
    if material_ids is None:
        material_ids = [0] * len(triangles)
    material_count = len(base_color_factors)
    fn = (_C.textured_mesh_to_volumetric_attr_multi_cpu if multi_surface
          else _C.textured_mesh_to_volumetric_attr_cpu)
    args = [
        torch.full((3,), 1.0 / resolution, dtype=torch.float32),
        torch.tensor([[0, 0, 0], [resolution] * 3], dtype=torch.int32),
        vertices, normals, uvs, torch.tensor(material_ids, dtype=torch.int32),
        [torch.tensor(c, dtype=torch.float32) for c in base_color_factors],
        [empty] * material_count, [0] * material_count, [0] * material_count,
        [0.0] * material_count, [empty] * material_count,
        [0] * material_count, [0] * material_count,
        [1.0] * material_count, [empty] * material_count,
        [0] * material_count, [0] * material_count,
        [torch.zeros(3)] * material_count, [empty] * material_count,
        [0] * material_count, [0] * material_count,
        [0] * material_count, [0.5] * material_count,
        [1.0] * material_count, [empty] * material_count,
        [0] * material_count, [0] * material_count,
    ]
    if multi_surface:
        args.extend([0.0, False, False, 15.0, 0, 0])
    else:
        args.extend([
            [empty] * material_count, [0] * material_count,
            [0] * material_count, 0.0, False, False,
        ])
    return fn(*args)


def main():
    z = (0.0, 0.0, 1.0)
    n14 = (math.sin(math.radians(14)), 0.0, math.cos(math.radians(14)))
    n15 = (math.sin(math.radians(15)), 0.0, math.cos(math.radians(15)))
    n16 = (math.sin(math.radians(16)), 0.0, math.cos(math.radians(16)))
    neg = (0.0, 0.0, -1.0)
    legacy = run([z, neg], multi_surface=False)[0]
    assert all(c == 1 for c in counts(legacy))
    merged = run([z, n14])[0]
    assert all(c == 1 for c in counts(merged)), counts(merged)
    threshold = run([z, n15])[0]
    assert all(c == 1 for c in counts(threshold)), counts(threshold)
    split = run([z, n16])[0]
    assert max(counts(split)) == 2, counts(split)
    custom_merged = run([z, n16], cluster_angle_degrees=20.0)[0]
    assert all(c == 1 for c in counts(custom_merged)), counts(custom_merged)
    custom_split = run([z, n14], cluster_angle_degrees=10.0)[0]
    assert max(counts(custom_split)) == 2, counts(custom_split)
    reversed_faces = run([z, neg])[0]
    assert max(counts(reversed_faces)) == 2, counts(reversed_faces)

    # Complete-link clustering does not merge a 28-degree span through an
    # intermediate normal, even though each neighboring pair is within 15°.
    n28 = (math.sin(math.radians(28)), 0.0, math.cos(math.radians(28)))
    chained = run([z, n14, n28])[0]
    assert max(counts(chained)) == 2, counts(chained)

    # A normal map can only be passed to the legacy VXZ entry point. VXZM's
    # native signature has no normal-texture inputs and stores authored mesh
    # surface normals.
    nx = torch.tensor([[[1.0, 0.5, 0.5]]], dtype=torch.float32)
    legacy_mapped = run([z], multi_surface=False, normal_texture=nx)
    surface_only = run([z])
    assert torch.all(legacy_mapped[6][:, 0] > 0.99), legacy_mapped[6]
    assert torch.all(surface_only[6][:, 2] > 0.99), surface_only[6]

    # Smooth authored normals avoid false clusters between tessellated faces
    # whose geometric face normals differ by more than the threshold.
    smooth = run([z, n16], authored_normals=[z, z])
    assert all(c == 1 for c in counts(smooth[0])), counts(smooth[0])

    # Incorrectly reused +Z authored normals cannot merge a back-facing
    # coincident primitive because the signed winding guard remains active.
    xy = [[0.10, 0.10, 0.50], [0.90, 0.10, 0.50], [0.10, 0.90, 0.50]]
    xy_reversed = [xy[0], xy[2], xy[1]]
    reversed_winding = run_geometry(
        [xy, xy_reversed],
        [[[0.0, 0.0, 1.0]] * 3, [[0.0, 0.0, 1.0]] * 3],
    )[0]
    assert all(c == 2 for c in counts(reversed_winding)), counts(reversed_winding)

    # Splitting must preserve the full per-surface record, not merely emit a
    # second coordinate. Give the two windings different materials while
    # deliberately reusing the same incorrect authored normal: every voxel
    # retains one red and one green record, and the stored winding-guard
    # fallback normals point in opposite directions.
    colored_reversed = run_geometry(
        [xy, xy_reversed],
        [[[0.0, 0.0, 1.0]] * 3, [[0.0, 0.0, 1.0]] * 3],
        base_color_factors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        material_ids=[0, 1],
    )
    _, inverse, colored_counts = torch.unique(
        colored_reversed[0], dim=0, return_inverse=True, return_counts=True,
    )
    assert torch.all(colored_counts == 2), colored_counts
    expected_colors = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    for group in range(len(colored_counts)):
        mask = inverse == group
        colors = colored_reversed[1][mask]
        colors = colors[torch.argsort(colors[:, 0])]
        assert torch.allclose(colors, expected_colors, atol=1e-6), colors
        normals_out = colored_reversed[6][mask]
        assert float(torch.dot(normals_out[0], normals_out[1])) < -0.9999

    # Samples admitted to one normal cluster must use the exact legacy
    # weighted material aggregation.  Only the multi-surface normal receives
    # the required final unit normalization.
    aggregated = run_geometry(
        [xy, xy],
        [[[0.0, 0.0, 1.0]] * 3, [list(n14)] * 3],
        base_color_factors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        material_ids=[0, 1],
    )
    legacy_aggregated = run_geometry(
        [xy, xy],
        [[[0.0, 0.0, 1.0]] * 3, [list(n14)] * 3],
        multi_surface=False,
        base_color_factors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        material_ids=[0, 1],
    )
    multi_order = torch.from_numpy(np.lexsort(tuple(
        aggregated[0][:, axis].numpy() for axis in (2, 1, 0)
    ))).long()
    legacy_order = torch.from_numpy(np.lexsort(tuple(
        legacy_aggregated[0][:, axis].numpy() for axis in (2, 1, 0)
    ))).long()
    assert torch.equal(aggregated[0][multi_order], legacy_aggregated[0][legacy_order])
    for multi_value, legacy_value in zip(aggregated[1:6], legacy_aggregated[1:6]):
        assert torch.allclose(
            multi_value[multi_order], legacy_value[legacy_order], atol=1e-6
        )
    expected_normal = torch.tensor(z) + torch.tensor(n14)
    expected_normal /= torch.linalg.vector_norm(expected_normal)
    assert torch.allclose(
        aggregated[6], expected_normal.repeat(len(aggregated[6]), 1), atol=1e-6
    ), aggregated[6]
    # Face order must not alter coordinates or clustered outputs.
    a = run([z, n16, neg])
    b = run([neg, z, n16])
    assert torch.equal(a[0], b[0])
    for lhs, rhs in zip(a[1:], b[1:]):
        assert torch.allclose(lhs, rhs, atol=1e-6), (lhs, rhs)

    # Two genuinely intersecting surfaces (XY and YZ) must retain both
    # normals in voxels around their intersection, even though the surfaces
    # are not coincident and do not share UV seams.
    yz = [[0.50, 0.10, 0.10], [0.50, 0.90, 0.10], [0.50, 0.10, 0.90]]
    cross = run_geometry(
        [xy, yz],
        [[[0.0, 0.0, 1.0]] * 3, [[1.0, 0.0, 0.0]] * 3],
        base_color_factors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        material_ids=[0, 1],
    )
    _, cross_inverse, cross_counts = torch.unique(
        cross[0], dim=0, return_inverse=True, return_counts=True,
    )
    assert int((cross_counts >= 2).sum()) > 0, "crossing surfaces lost duplicate voxel records"
    for group in torch.nonzero(cross_counts >= 2).flatten().tolist():
        mask = cross_inverse == group
        colors = cross[1][mask]
        normals_out = cross[6][mask]
        assert bool(((colors[:, 0] > 0.999) & (normals_out[:, 2] > 0.999)).any())
        assert bool(((colors[:, 1] > 0.999) & (normals_out[:, 0] > 0.999)).any())

    # Safety limits fail explicitly; they never silently discard a surface.
    try:
        limited_args = [z, neg]
        resolution = 8
        triangle = torch.tensor([
            [0.10, 0.10, 0.50], [0.90, 0.10, 0.50], [0.10, 0.90, 0.50]
        ], dtype=torch.float32)
        vertices = triangle.repeat(2, 1, 1)
        normals = torch.stack([torch.tensor(n).repeat(3, 1) for n in limited_args]).float()
        empty = torch.empty(0, dtype=torch.float32)
        native_args = [
            torch.full((3,), 1.0 / resolution),
            torch.tensor([[0, 0, 0], [resolution] * 3], dtype=torch.int32),
            vertices, normals, torch.zeros((2, 3, 2)), torch.zeros(2, dtype=torch.int32),
            [torch.ones(3)], [empty], [0], [0], [0.0], [empty], [0], [0],
            [1.0], [empty], [0], [0], [torch.zeros(3)], [empty], [0], [0],
            [0], [0.5], [1.0], [empty], [0], [0],
            0.0, False, False, 15.0, 1, 0,
        ]
        _C.textured_mesh_to_volumetric_attr_multi_cpu(*native_args)
        raise AssertionError("max_records_per_voxel did not reject two clusters")
    except RuntimeError as error:
        assert "max_records_per_voxel" in str(error)

    # The total limit also guards the pre-clustering triangle/voxel samples,
    # so a dense input fails before an unbounded temporary vector is built.
    raw_cap_args = list(native_args)
    raw_cap_args[-2] = 0
    raw_cap_args[-1] = 1
    try:
        _C.textured_mesh_to_volumetric_attr_multi_cpu(*raw_cap_args)
        raise AssertionError("max_total_records did not reject raw samples")
    except RuntimeError as error:
        assert "input samples exceed max_total_records" in str(error).lower()

    # The public native entry point rejects non-finite authored normals before
    # they can invalidate std::sort's comparison ordering or cluster math.
    invalid_args = list(native_args)
    invalid_normals = normals.clone()
    invalid_normals[0, 0, 0] = float("nan")
    invalid_args[3] = invalid_normals
    try:
        _C.textured_mesh_to_volumetric_attr_multi_cpu(*invalid_args)
        raise AssertionError("non-finite surface normal was accepted")
    except RuntimeError as error:
        assert "normals must contain only finite values" in str(error)
    print("multi-surface voxel clustering test passed")


if __name__ == "__main__":
    main()
