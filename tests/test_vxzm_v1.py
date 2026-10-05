"""VXZM v1 semantics: final cluster counts, schema, normals and region I/O."""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'o-voxel'), str(ROOT / 'tests')]
import o_voxel
from o_voxel.convert.volumetic_attr import record_confidence
from test_glb_to_vxzm import validate_roundtrip
from test_voxelize_pbr_formats import fixture_dump
from test_multi_surface_voxel import run_geometry
import create_vxzm


def test_counts_and_roundtrip():
    coord = torch.tensor([[m, 0, 0] for m in range(1, 10) for _ in range(m)], dtype=torch.int32)
    q = record_confidence(coord)
    expected = [255, 255, 230, 179, 128, 77, 26, 26, 26]
    start = 0
    for m, value in enumerate(expected, 1):
        assert torch.all(q[start:start+m] == value), (m, q[start:start+m])
        start += m
    attr = {'base_color': torch.arange(len(coord)*3).byte().reshape(-1,3),
            'normal': torch.tensor([[127,127,255]]).byte().repeat(len(coord),1),
            'confidence': q}
    order = torch.randperm(len(coord))
    with tempfile.TemporaryDirectory() as td:
        for compression in ('none', 'deflate', 'lzma', 'zstd'):
            path = Path(td) / 'sample.vxzm'
            o_voxel.io.write_vxzm(path, coord[order], {k:v[order] for k,v in attr.items()},
                                 grid_size=16, region_resolution=4, compression=compression,
                                 normal_source='geometry_reoriented_face_v1')
            info = o_voxel.io.read_vxzm_info(path)
            assert info['version'] == path.read_bytes()[4] == 1
            assert info['normal_source'] == 'geometry_reoriented_face_v1'
            assert {v['name'] for v in info['record_layout']} == set(attr)
            rc, ra = o_voxel.io.read_vxzm(path)
            validate_roundtrip(coord, attr, rc, ra)
            assert torch.equal(ra['confidence'], record_confidence(rc))
            rc2, ra2 = o_voxel.io.read_vxzm(io.BytesIO(path.read_bytes()))
            validate_roundtrip(coord, attr, rc2, ra2)
            corrupt = bytearray(path.read_bytes()); corrupt[4] = 0
            try:
                o_voxel.io.read_vxzm(corrupt)
            except ValueError:
                pass
            else:
                raise AssertionError('version/normal-source mismatch accepted')


def test_converter_and_pure_cli():
    dump = fixture_dump()
    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr_multi(
        dump, grid_size=8, aabb=[[-.5]*3,[.5]*3], color_space='linear')
    assert set(attr) == {'base_color','normal','confidence','topology'}
    assert torch.all(attr['confidence'] == 255)
    assert torch.all(attr['normal'][:,2] == 255)
    args = create_vxzm.build_parser().parse_args(['asset'])
    assert args.renormal and args.pure_vxzm and not args.visualize
    assert args.output_format == 'vxzm'
    with tempfile.TemporaryDirectory() as td:
        from create_vxz import convert_dump
        args.resolution = 8; args.region_resolution = 4; args.color_space = 'linear'
        path = Path(td) / 'pure.vxzm'
        convert_dump(dump, args, path)
        assert list(Path(td).iterdir()) == [path]
        assert o_voxel.io.read_vxzm_info(path)['version'] == 2


def test_symmetric_footprint():
    # Two parallel colored triangles in one cluster. Reverse both windings:
    # closest-point distances and RGB weights must remain exactly symmetric.
    a = [[.1,.1,.46],[.9,.1,.46],[.1,.9,.46]]
    b = [[.1,.1,.54],[.9,.1,.54],[.1,.9,.54]]
    colors = [[1.,0.,0.],[0.,0.,1.]]
    front = run_geometry([a,b], [[[0,0,1]]*3]*2,
                         base_color_factors=colors, material_ids=[0,1])
    back = run_geometry([a[::-1],b[::-1]], [[[0,0,-1]]*3]*2,
                        base_color_factors=colors, material_ids=[0,1])
    assert torch.equal(front[0], back[0])
    assert torch.allclose(front[1], back[1], atol=1e-6)
    assert torch.isfinite(front[1]).all()


def test_geometry_pbr_writer_source():
    import create_geo_pbr
    dump = fixture_dump()
    args = create_geo_pbr.build_parser().parse_args(['asset', '--output-format', 'vxzm',
                                                  '--resolution', '8', '--region-resolution', '4',
                                                  '--color-space', 'linear'])
    coord, attr = create_geo_pbr._convert_pbr_attributes(dump, args)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'sample.vxzm'
        create_geo_pbr._write_pbr(path, coord, attr, args)
        assert o_voxel.io.read_vxzm_info(path)['version'] == 2


def test_topology_preserves_samples():
    import copy
    dump = fixture_dump()
    # Two same-normal contributions merge; provenance must OR, not average.
    second = copy.deepcopy(dump['objects'][0])
    second['topology'][:] = 1
    dump['objects'].append(second)
    coord, marked = o_voxel.convert.blender_dump_to_volumetric_attr_multi(
        dump, grid_size=8, aabb=[[-.5]*3,[.5]*3], color_space='linear')
    assert torch.all(marked['topology'] == 1)
    second['topology'][:] = 0
    other_coord, unmarked = o_voxel.convert.blender_dump_to_volumetric_attr_multi(
        dump, grid_size=8, aabb=[[-.5]*3,[.5]*3], color_space='linear')
    assert torch.equal(coord, other_coord)
    for name in ('base_color', 'normal', 'confidence'):
        assert torch.equal(marked[name], unmarked[name])
    assert torch.all(unmarked['topology'] == 0)
    # Flag provenance follows each normal cluster, not every record at XYZ.
    second['faces'] = second['faces'][:, [0,2,1]].copy()
    second['normals'] *= -1
    second['topology'][:] = 1
    split_coord, split = o_voxel.convert.blender_dump_to_volumetric_attr_multi(
        dump, grid_size=8, aabb=[[-.5]*3,[.5]*3], color_space='linear')
    assert torch.any(split['normal'][:,2] == 0) and torch.any(split['normal'][:,2] == 255)
    assert torch.equal(split['topology'][:,0] != 0, split['normal'][:,2] == 0)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'marked.vxzm'
        o_voxel.io.write_vxzm(path, coord, marked, grid_size=8, region_resolution=4,
                             normal_source='geometry_reoriented_face_v1')
        info = o_voxel.io.read_vxzm_info(path)
        assert info['version'] == 2 and '0' in info['topology_bits']
        rc, ra = o_voxel.io.read_vxzm(path)
        validate_roundtrip(coord, marked, rc, ra)


if __name__ == '__main__':
    test_counts_and_roundtrip()
    test_converter_and_pure_cli()
    test_symmetric_footprint()
    test_geometry_pbr_writer_source()
    test_topology_preserves_samples()
    print('VXZM v1/v2 tests passed')
