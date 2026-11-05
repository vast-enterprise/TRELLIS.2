"""Run with blender -b --python-exit-code 1 -P tests/test_blender_vertex_color.py."""
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data_toolkit/blender_script'))
import dump_pbr


def check_dump(directory, domain, multiply):
    dump_pbr.init_scene()
    mesh = bpy.data.meshes.new('colors')
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    obj = bpy.data.objects.new('colors', mesh)
    bpy.context.scene.collection.objects.link(obj)
    colors = mesh.color_attributes.new(name='paint', type='FLOAT_COLOR', domain=domain)
    expected = np.asarray([[.8, .1, .2, .3], [.2, .7, .3, .6], [.3, .2, .9, 1]], np.float32)
    for item, value in zip(colors.data, expected):
        item.color = value
    material = bpy.data.materials.new('paint')
    material.use_nodes = True
    mesh.materials.append(material)
    tree = material.node_tree
    color_node = tree.nodes.new('ShaderNodeVertexColor')
    color_node.layer_name = colors.name
    source = color_node.outputs['Color']
    factor = [1, 1, 1]
    if multiply:
        mix = tree.nodes.new('ShaderNodeMixRGB')
        mix.blend_type = 'MULTIPLY'
        mix.inputs['Fac'].default_value = 1
        mix.inputs['Color2'].default_value = (.5, .25, .75, 1)
        tree.links.new(source, mix.inputs['Color1'])
        source = mix.outputs['Color']
        factor = [.5, .25, .75]
    tree.links.new(source, tree.nodes.get('Principled BSDF').inputs['Base Color'])
    output = Path(directory) / f'{domain}_{multiply}.pkl'
    dump_pbr.main(SimpleNamespace(object='fixture.blend', output_path=str(output),
                                  renormal=True, merge_distance=0.0, max_edge_faces=0))
    with output.open('rb') as stream:
        dump = pickle.load(stream)
    pack = dump['objects'][0]
    mat = dump['materials'][int(pack['mat_ids'][0])]
    assert mat['baseColorUseVertexColor']
    assert mat['baseColorVertexColorLayer'] == 'paint'
    np.testing.assert_allclose(mat['baseColorFactor'], factor)
    np.testing.assert_allclose(pack['vertex_colors'], expected[None], atol=1e-6)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        for domain in ('CORNER', 'POINT'):
            for multiply in (False, True):
                check_dump(directory, domain, multiply)
    print('Blender vertex-color export checks passed (4 cases)')
