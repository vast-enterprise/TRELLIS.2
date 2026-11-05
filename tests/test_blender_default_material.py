"""Run with blender -b --python-exit-code 1 -P tests/test_blender_default_material.py."""
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data_toolkit' / 'blender_script'))
import dump_pbr


def check_dump(directory, name, slots, face_indices, expected_fallback):
    dump_pbr.init_scene()
    # Deliberately collide with the fallback's preferred name: existing
    # materials must remain intact even when Blender renames the new one.
    authored = bpy.data.materials.new(name='DefaultPBR')
    authored.use_nodes = True
    bsdf = authored.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.1, 0.2, 0.3, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.25
    mesh = bpy.data.meshes.new(name)
    vertices = [(float(i), float(y), float(z))
                for i in range(len(face_indices))
                for y, z in ((0, 0), (1, 0), (0, 1))]
    mesh.from_pydata(vertices, [], [(3*i, 3*i+1, 3*i+2)
                                  for i in range(len(face_indices))])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    for present in slots:
        mesh.materials.append(authored if present else None)
    for face, index in zip(mesh.polygons, face_indices):
        face.material_index = index

    output = Path(directory) / f'{name}.pickle'
    dump_pbr.main(SimpleNamespace(
        object='fixture.blend', output_path=str(output), renormal=True,
        merge_distance=0.0, max_edge_faces=0,
    ))
    with output.open('rb') as stream:
        dumped = pickle.load(stream)
    assert len(dumped['materials']) == (2 if any(expected_fallback) else 1)
    assert len(dumped['objects']) == 1
    ids = dumped['objects'][0]['mat_ids']
    assert len(ids) == len(face_indices)
    assert np.all((ids >= 0) & (ids < len(dumped['materials'])))
    for material_id, fallback in zip(ids, expected_fallback):
        material = dumped['materials'][material_id]
        assert material['shaderType'] == 'Principled'
        np.testing.assert_allclose(material['baseColorFactor'],
                                   [0.8, 0.8, 0.8, 1.0] if fallback
                                   else [0.1, 0.2, 0.3, 1.0])
        assert material['roughnessFactor'] == (0.5 if fallback else 0.25)
        assert material['metallicFactor'] == 0.0
        assert material['alphaFactor'] == 1.0
    np.testing.assert_allclose(bsdf.inputs['Base Color'].default_value,
                               [0.1, 0.2, 0.3, 1.0])


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        check_dump(directory, 'no_slots', [], [0], [True])
        check_dump(directory, 'empty_slot', [False], [0], [True])
        check_dump(directory, 'mixed_slots', [True, False], [0, 1], [False, True])
        check_dump(directory, 'invalid_index', [True], [0, 7], [False, True])
        check_dump(directory, 'authored_only', [True], [0], [False])
    print('Default PBR material regression checks passed (5 cases)')
