#!/usr/bin/env python3
"""Verify that Blender dumping preserves transformed GLB authored normals.

This is intentionally a standalone Blender integration test.  The fixture's
authored normal lies in the triangle plane, so it cannot be confused with a
recomputed geometric face normal.  It is also parallel to the first local
edge, which makes the expected world-space direction observable after an
arbitrary glTF/Blender axis conversion and a non-uniform node transform.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_glb_to_vxz import run_blender_dump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default="blender")
    args = parser.parse_args()

    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.4, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    # Deliberately non-geometric: this is parallel to v1-v0, whereas the
    # geometric face normal is perpendicular to that edge.
    authored = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (3, 1))
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=authored,
        process=False,
    )
    angle = np.deg2rad(37.0)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0, 0.0],
        [np.sin(angle), np.cos(angle), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scale = np.diag([1.7, 0.6, 1.2, 1.0])
    transform = rotation @ scale
    transform[:3, 3] = [0.2, -0.1, 0.3]
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="authored-normal", transform=transform)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        glb_path = temp / "authored_normal.glb"
        dump_path = temp / "authored_normal.pickle"
        # trimesh omits NORMAL accessors by default; force them into the GLB
        # so the test exercises authored data instead of Blender's fallback.
        from trimesh.exchange.gltf import export_glb
        glb_path.write_bytes(export_glb(scene, include_normals=True))
        run_blender_dump(args.blender, glb_path, dump_path)
        with dump_path.open("rb") as stream:
            dump = pickle.load(stream)

    objects = [obj for obj in dump["objects"] if len(obj["faces"])]
    if len(objects) != 1 or objects[0]["faces"].shape != (1, 3):
        raise AssertionError("unexpected Blender fixture topology")
    obj = objects[0]
    triangle = obj["vertices"][obj["faces"][0]].astype(np.float64)
    dumped = obj["normals"][0].astype(np.float64)
    dumped /= np.linalg.norm(dumped, axis=1, keepdims=True)
    edge = triangle[1] - triangle[0]
    edge /= np.linalg.norm(edge)
    face = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    face /= np.linalg.norm(face)

    # All three imported authored normals must follow the transformed local X
    # edge.  They must not have been replaced by the geometric face normal.
    if not np.all(dumped @ edge > 0.9999):
        raise AssertionError(("authored normals are not in the dumped world frame", dumped, edge))
    if not np.all(np.abs(dumped @ face) < 1e-4):
        raise AssertionError(("authored normals were replaced by face normals", dumped, face))
    print("Blender authored surface-normal transform test passed")


if __name__ == "__main__":
    main()
