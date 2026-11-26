"""Blender reference extraction with the original per-element Python reads.

Uses the SAME geometry-only import/normalization as dump_pbr.py, isolating
bulk extraction speed from the cost of disabling material/repair processing.
"""
import argparse
import pickle
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data_toolkit/blender_script'))
from dump_pbr import init_scene, load_object, normalize_scene

parser = argparse.ArgumentParser()
parser.add_argument('--object', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
init_scene()
load_object(args.object, merge_vertices=False)
normalize_scene()
depsgraph = bpy.context.evaluated_depsgraph_get()
out = {'objects': [], 'materials': [], 'geometry_only': True}
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        matrix = evaluated.matrix_world.copy()
        vertices = np.asarray([(matrix @ v.co)[:] for v in mesh.vertices], dtype=np.float32).reshape(-1, 3)
        triangles = list(mesh.loop_triangles)
        faces = np.asarray([list(t.vertices) for t in triangles], dtype=np.int32).reshape(-1, 3)
        out['objects'].append({'vertices': vertices, 'faces': faces})
    finally:
        evaluated.to_mesh_clear()
with open(args.output_path, 'wb') as stream:
    pickle.dump(out, stream)
