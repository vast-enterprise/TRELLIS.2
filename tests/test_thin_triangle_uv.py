#!/usr/bin/env python3
"""Native regression for UVs on the conservative border of a thin triangle.

The voxel scan deliberately emits a one-voxel conservative border.  For a
point outside a triangle, UVs must come from the closest *point on the
triangle*, not from independently clamping its three plane-projection
barycentric weights.  The latter moves a point along an edge toward a vertex,
which is conspicuous with a gradient texture on a thin triangle.

Run in an environment where O-Voxel has been built:
    PYTHONPATH=o-voxel:. python tests/test_thin_triangle_uv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OVOXEL_ROOT = ROOT / "o-voxel"
if str(OVOXEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OVOXEL_ROOT))

from o_voxel import _C


def closest_barycentric(point: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Ericson closest-point barycentrics, matching the C++ implementation."""
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = torch.dot(ab, ap).item(), torch.dot(ac, ap).item()
    if d1 <= 0.0 and d2 <= 0.0:
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    bp = point - b
    d3, d4 = torch.dot(ab, bp).item(), torch.dot(ac, bp).item()
    if d3 >= 0.0 and d4 <= d3:
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return torch.tensor([1.0 - v, v, 0.0], dtype=torch.float64)
    cp = point - c
    d5, d6 = torch.dot(ab, cp).item(), torch.dot(ac, cp).item()
    if d6 >= 0.0 and d5 <= d6:
        return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return torch.tensor([1.0 - w, 0.0, w], dtype=torch.float64)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return torch.tensor([0.0, 1.0 - w, w], dtype=torch.float64)
    inv_sum = 1.0 / (va + vb + vc)
    v, w = vb * inv_sum, vc * inv_sum
    return torch.tensor([1.0 - v - w, v, w], dtype=torch.float64)


def plane_barycentric(point: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ab, ac, ap = b - a, c - a, point - a
    d00, d01, d11 = torch.dot(ab, ab), torch.dot(ab, ac), torch.dot(ac, ac)
    d20, d21 = torch.dot(ap, ab), torch.dot(ap, ac)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return torch.stack((1.0 - v - w, v, w))


def run_test() -> None:
    resolution = 64
    a = torch.tensor([0.10, 0.10, 0.50], dtype=torch.float64)
    b = torch.tensor([0.90, 0.10, 0.50], dtype=torch.float64)
    c = torch.tensor([0.10, 0.13, 0.50], dtype=torch.float64)
    vertices = torch.stack((a, b, c)).float().unsqueeze(0)
    normals = torch.tensor([[[0.0, 0.0, 1.0]] * 3], dtype=torch.float32)
    # A horizontal linear texture makes sampled red equal to u at every mip.
    tex_width, tex_height = 128, 8
    texel_u = (torch.arange(tex_width, dtype=torch.float32) + 0.5) / tex_width
    texture = torch.zeros((tex_height, tex_width, 3), dtype=torch.float32)
    texture[..., 0] = texel_u.unsqueeze(0)
    uvs = torch.tensor([[[0.0, 0.5], [1.0, 0.5], [0.02, 0.5]]], dtype=torch.float32)
    empty = torch.empty(0, dtype=torch.float32)

    out = _C.textured_mesh_to_volumetric_attr_cpu(
        torch.full((3,), 1.0 / resolution, dtype=torch.float32),
        torch.tensor([[0, 0, 0], [resolution, resolution, resolution]], dtype=torch.int32),
        vertices, normals, uvs, torch.zeros(1, dtype=torch.int32),
        [torch.ones(3, dtype=torch.float32)], [texture], [1], [1],
        [0.0], [empty], [0], [0],
        [1.0], [empty], [0], [0],
        [torch.zeros(3, dtype=torch.float32)], [empty], [0], [0],
        [0], [0.5], [1.0], [empty], [0], [0],
        [empty], [0], [0], 0.0, False, False,
    )
    coords, colors = out[0], out[1]
    points = (coords.double() + 0.5) / resolution
    uv_vertices = uvs[0].double()
    exterior_count = 0
    worst_error = 0.0
    old_wrong_distance = 0.0
    for point, color in zip(points, colors.double()):
        raw = plane_barycentric(point, a, b, c)
        expected_bary = closest_barycentric(point, a, b, c)
        expected_u = torch.dot(expected_bary, uv_vertices[:, 0]).item()
        error = abs(color[0].item() - expected_u)
        worst_error = max(worst_error, error)
        # Check that the fixture truly covers conservative-border samples and
        # that closest-point projection differs materially from the old code.
        if raw.min().item() < -1e-5:
            exterior_count += 1
            old_bary = torch.clamp(raw, 0.0, 1.0)
            old_bary /= old_bary.sum()
            old_u = torch.dot(old_bary, uv_vertices[:, 0]).item()
            old_wrong_distance = max(old_wrong_distance, abs(expected_u - old_u))

    assert exterior_count > 0, "fixture did not hit the conservative triangle border"
    assert old_wrong_distance > 0.02, "fixture does not distinguish old edge clamping"
    assert worst_error < 0.015, f"thin-triangle UV error too high: {worst_error:.6f}"
    print(
        f"thin triangle UV test passed: {len(coords)} voxels, "
        f"{exterior_count} exterior samples, max error {worst_error:.6f}, "
        f"old-clamp discrepancy {old_wrong_distance:.6f}"
    )


if __name__ == "__main__":
    run_test()
