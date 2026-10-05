"""Run inside Blender: UV/material preservation, repair, mirrors and discard."""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data_toolkit/blender_script'))
from geometry_normals import clean_geometry, triangle_normals, triangle_topology


def mesh(vertices, faces):
    m = bpy.data.meshes.new('renormal-fixture')
    m.from_pydata(vertices, [], faces)
    m.update()
    layer = m.uv_layers.new()
    for i, p in enumerate(m.polygons):
        p.material_index = i
        for j in p.loop_indices:
            v = m.vertices[m.loops[j].vertex_index].co
            layer.data[j].uv = (v.x + i * 3, v.y + i * 3)
    return m


def test_repair():
    # Split seam vertices, two same-direction half-edges, distinct loop UVs.
    m = mesh([(0,0,0),(1,0,0),(0,1,0),(0,0,0),(1,0,0),(1,-1,0)],
             [(0,1,2),(3,4,5)])
    stats = clean_geometry(m, Matrix.Identity(4))
    assert stats['merged_vertices'] == 2
    assert len(m.polygons) == 2 and len(m.vertices) == 4
    v = np.array([v.co[:] for v in m.vertices])
    f = np.array([t.vertices[:] for t in m.loop_triangles])
    n = triangle_normals(v, f)
    assert np.dot(n[0,0], n[1,0]) > .9999
    # Face reversal must not detach UVs from their corresponding corners.
    for p in m.polygons:
        for j in p.loop_indices:
            co = m.vertices[m.loops[j].vertex_index].co
            assert np.allclose(m.uv_layers.active.data[j].uv[:],
                               (co.x + p.material_index*3, co.y + p.material_index*3))


def test_mirror_and_custom_normals():
    m = mesh([(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
             [(0,2,1),(0,1,3),(1,2,3),(2,0,3)])
    m.normals_split_custom_set([(1,0,0)] * len(m.loops))
    clean_geometry(m, Matrix.Diagonal((-2.0, 3.0, .5, 1.0)))
    v = np.array([v.co[:] for v in m.vertices]); f = np.array([t.vertices[:] for t in m.loop_triangles])
    n = triangle_normals(v, f)[:, 0]
    assert np.allclose(np.linalg.norm(n, axis=1), 1)
    assert np.all(np.sum(n * (v[f].mean(axis=1) - v.mean(axis=0)), axis=1) > 0)


def test_discard_and_boundary():
    for faces in [[(0,1,2),(1,0,3),(0,1,4)], [(0,1,2),(2,1,0)]]:
        m = mesh([(0,0,0),(1,0,0),(0,1,0),(0,-1,0),(0,0,1)], faces)
        try:
            clean_geometry(m, Matrix.Identity(4), max_edge_faces=2)
        except ValueError as e:
            assert 'discard asset' in str(e)
        else:
            raise AssertionError('bad geometry accepted')
    m = mesh([(0,0,0),(1,0,0),(0,1,0)], [(0,1,2)])
    clean_geometry(m, Matrix.Identity(4))  # open boundaries are allowed
    assert len(m.polygons) == 1

    # Exported GLBs often split coincident vertices across materials/UVs.
    # Welding must not silently delete either differently colored face.
    m = mesh([(0,0,0),(1,0,0),(0,1,0)] * 2, [(0,1,2),(5,4,3)])
    stats = clean_geometry(m, Matrix.Identity(4))
    assert stats['skipped_weld_vertices'] == 3
    assert len(m.polygons) == 2 and len(m.vertices) == 6
    v = np.array([v.co[:] for v in m.vertices]); f = np.array([t.vertices[:] for t in m.loop_triangles])
    n = triangle_normals(v, f)[:,0]
    assert np.dot(n[0], n[1]) < -.999


def test_three_face_junction():
    m = mesh([(0,0,0),(1,0,0),(0,1,0),(0,-1,0),(0,0,1)],
             [(0,1,2),(1,0,3),(0,1,4)])
    original = np.array([p.normal[:] for p in m.polygons])
    stats = clean_geometry(m, Matrix.Identity(4), max_edge_faces=3)
    assert stats['multi_face_edges'] == 1 and stats['boundary_edges'] == 6
    assert len(m.polygons) == 3
    assert np.all(triangle_topology(m) == 1)
    assert np.allclose(original, np.array([p.normal[:] for p in m.polygons]))


test_repair()
test_mirror_and_custom_normals()
test_discard_and_boundary()
test_three_face_junction()
print('geometry re-normal tests passed')
