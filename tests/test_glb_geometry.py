"""Direct geometry parsing and CLI contracts; no Blender or GPU required."""
import copy
import json
import pickle
import struct
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glb_geometry import load_glb_geometry, UnsupportedGeometryFeature
import create_geo_pbr as pipeline


def write_glb(path, *, strided=False, indexed=True, unused=False, edit=None):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype='<f4')
    if unused:
        vertices = np.vstack((vertices, [100, 100, 100])).astype('<f4')
    if strided:
        packed = np.zeros((len(vertices), 5), dtype='<f4')
        packed[:, 1:4] = vertices
        data = packed.tobytes()
        position_view = {'buffer': 0, 'byteOffset': 0, 'byteLength': len(data), 'byteStride': 20}
    else:
        data = vertices.tobytes()
        position_view = {'buffer': 0, 'byteOffset': 0, 'byteLength': len(data)}
    doc = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
           'nodes': [{'mesh': 0}], 'meshes': [{'primitives': [{'attributes': {'POSITION': 0}}]}],
           'bufferViews': [position_view],
           'accessors': [{'bufferView': 0, 'byteOffset': 4 if strided else 0,
                          'componentType': 5126, 'count': len(vertices), 'type': 'VEC3'}]}
    if indexed:
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(data), 'byteLength': 8})
        doc['accessors'].append({'bufferView': 1, 'byteOffset': 2, 'componentType': 5123,
                                 'count': 3, 'type': 'SCALAR'})
        doc['meshes'][0]['primitives'][0]['indices'] = 1
        data += np.asarray([999, 0, 1, 2], dtype='<u2').tobytes()
    doc['buffers'] = [{'byteLength': len(data)}]
    if edit:
        edit(doc)
    metadata = json.dumps(doc).encode()
    metadata += b' ' * (-len(metadata) % 4)
    data += b'\0' * (-len(data) % 4)
    path.write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(metadata) + len(data)) +
                     struct.pack('<II', len(metadata), 0x4E4F534A) + metadata +
                     struct.pack('<II', len(data), 0x004E4942) + data)
    return vertices


@pytest.mark.parametrize('strided,indexed', [(False, True), (True, True), (False, False)])
def test_accessor_layout_and_axis_conversion(tmp_path, strided, indexed):
    path = tmp_path / 'mesh.glb'
    write_glb(path, strided=strided, indexed=indexed)
    dump = load_glb_geometry(path, normalize=False)
    np.testing.assert_array_equal(dump['objects'][0]['vertices'], [[0, 0, 0], [1, 0, 0], [0, -1, 1]])
    np.testing.assert_array_equal(dump['objects'][0]['faces'], [[0, 1, 2]])
    assert dump['coordinate_system'] == 'blender_world_z_up'
    # All arrays remain valid and writable after the mmap has closed.
    dump['objects'][0]['vertices'][0, 0] = 7


def test_unused_positions_do_not_change_bounds(tmp_path):
    path = tmp_path / 'mesh.glb'
    write_glb(path, unused=True)
    v = load_glb_geometry(path)['objects'][0]['vertices']
    np.testing.assert_array_equal(v, [[-.5, .5, -.5], [.5, .5, -.5], [-.5, -.5, .5]])


def test_scene_hierarchy_instances_matrix_trs_and_primitives(tmp_path):
    def edit(doc):
        m = np.eye(4); m[:3, 3] = [0, 0, 4]
        doc['nodes'] = [{'translation': [1, 2, 3], 'children': [1, 2]},
                        {'mesh': 0, 'scale': [-2, 1, 1], 'rotation': [0, 0, 2**-.5, 2**-.5]},
                        {'mesh': 0, 'matrix': m.T.ravel().tolist()},
                        {'mesh': 0, 'translation': [100, 100, 100]}]
        doc['meshes'][0]['primitives'] *= 2
    path = tmp_path / 'mesh.glb'
    write_glb(path, edit=edit)
    objects = load_glb_geometry(path, normalize=False)['objects']
    assert len(objects) == 4  # node 3 is outside the selected scene
    np.testing.assert_allclose(objects[0]['vertices'], [[1, -3, 2], [1, -3, 0], [0, -4, 2]], atol=1e-6)
    np.testing.assert_array_equal(objects[2]['vertices'], [[1, -7, 2], [2, -7, 2], [1, -8, 3]])
    np.testing.assert_array_equal(objects[0]['vertices'], objects[1]['vertices'])


@pytest.mark.parametrize('feature', ['compression', 'animation', 'skin', 'morph', 'sparse', 'mode', 'external'])
def test_unsupported_geometry_is_explicit(tmp_path, feature):
    def edit(doc):
        if feature == 'compression': doc['extensionsUsed'] = ['KHR_draco_mesh_compression']
        if feature == 'animation': doc['animations'] = [{}]
        if feature == 'skin': doc['nodes'][0]['skin'] = 0
        if feature == 'morph': doc['meshes'][0]['primitives'][0]['targets'] = [{'POSITION': 0}]
        if feature == 'sparse': doc['accessors'][0]['sparse'] = {}
        if feature == 'mode': doc['meshes'][0]['primitives'][0]['mode'] = 5
        if feature == 'external': doc['buffers'][0]['uri'] = 'mesh.bin'
    path = tmp_path / 'mesh.glb'
    write_glb(path, edit=edit)
    with pytest.raises(UnsupportedGeometryFeature): load_glb_geometry(path)


@pytest.mark.parametrize('malformed', ['bounds', 'cycle', 'matrix', 'length', 'indices'])
def test_malformed_input_is_rejected(tmp_path, malformed):
    def edit(doc):
        if malformed == 'bounds': doc['accessors'][0]['count'] = 1000
        if malformed == 'cycle': doc['nodes'][0]['children'] = [0]
        if malformed == 'matrix': doc['nodes'][0]['matrix'] = [0] * 16
        if malformed == 'indices': doc['accessors'][1]['byteOffset'] = 0  # reads out-of-range 999
    path = tmp_path / 'mesh.glb'
    write_glb(path, edit=edit)
    if malformed == 'length': path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError): load_glb_geometry(path)


def test_auto_fallback_only_for_unsupported_features(tmp_path):
    path, output = tmp_path / 'mesh.glb', tmp_path / 'mesh.pkl'
    write_glb(path, edit=lambda doc: doc.update(extensionsUsed=['KHR_draco_mesh_compression']))
    def fake_blender(*args, **kwargs):
        assert kwargs == {'geometry_only': True}
        output.write_bytes(pickle.dumps({'objects': [{}]}))
    with mock.patch.object(pipeline, 'run_blender_dump', side_effect=fake_blender) as blender:
        assert pipeline._load_geometry_input(path, output, loader='auto', blender='unused') == {'objects': [{}]}
        assert blender.call_count == 1
        with pytest.raises(UnsupportedGeometryFeature):
            pipeline._load_geometry_input(path, output, loader='glb', blender='unused')
        path.write_bytes(b'invalid')
        with pytest.raises(ValueError):
            pipeline._load_geometry_input(path, output, loader='auto', blender='unused')
        assert blender.call_count == 1


def test_direct_cli_rotation_pickle_without_blender(tmp_path):
    path = tmp_path / 'mesh.glb'
    write_glb(path)
    expected = copy.deepcopy(load_glb_geometry(path))
    pipeline.rotate_dump_geometry(expected, [0, 0, 90])
    with mock.patch.object(pipeline, 'run_blender_dump', side_effect=AssertionError('Blender called')):
        assert pipeline.main(['asset', '--glb', str(path), '--geometry-only', '--pkl-only',
                              '--rotation', '0', '0', '90', '--output-dir', str(tmp_path)], local=True) == 0
    with (tmp_path / 'asset.pbr.pkl').open('rb') as stream: actual = pickle.load(stream)
    np.testing.assert_array_equal(actual['objects'][0]['vertices'], expected['objects'][0]['vertices'])
    assert not (tmp_path / '.cache/asset.pbr.pkl').exists()


def test_direct_cli_default_converts_without_pickle(tmp_path):
    path = tmp_path / 'mesh.glb'
    write_glb(path)
    with mock.patch.object(pipeline, 'run_blender_dump', side_effect=AssertionError('Blender called')), \
         mock.patch.object(pipeline, '_load_dump', side_effect=AssertionError('Pickle read')), \
         mock.patch.object(pipeline, 'convert_geometry', return_value=([], {})) as convert, \
         mock.patch.object(pipeline, '_write_geometry'):
        assert pipeline.main(['asset', '--glb', str(path), '--geometry-only',
                              '--output-dir', str(tmp_path)], local=True) == 0
    assert convert.call_args.args[0]['geometry_loader'] == 'glb'
