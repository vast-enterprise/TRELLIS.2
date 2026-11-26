#!/usr/bin/env python3
"""Compare full Blender and direct GLB load -> packed CUDA geometry pipelines.

No PBR sampling, welding, repair, UV export, VXZ compression/write or result
comparison is timed. Timings include file reads, axis/world transforms, both
normalization stages, validation, sampling and CPU output transfers. CUDA and
Python imports are warmed first. Input files can be in the filesystem cache.
"""
import argparse
import json
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import create_geo_pbr as pipeline
from glb_geometry import load_glb_geometry
import o_voxel


def compare_packed(a, b, resolution):
    ca, cb = a[0], b[0]
    ka = ca.long() @ torch.tensor([resolution**2, resolution, 1])
    kb = cb.long() @ torch.tensor([resolution**2, resolution, 1])
    assert torch.all(ka[1:] > ka[:-1]) and torch.all(kb[1:] > kb[:-1])
    pos = torch.searchsorted(kb, ka)
    valid = pos < len(kb)
    valid[valid.clone()] &= kb[pos[valid]] == ka[valid]
    pos = pos[valid]
    va, vb = a[1]['vertices'][valid], b[1]['vertices'][pos]
    return {
        'blender_voxels': len(ca), 'direct_voxels': len(cb),
        'common_voxels': int(valid.sum()), 'coordinates_exact': torch.equal(ca, cb),
        'intersection_flag_mismatches': int((a[1]['intersected'][valid] != b[1]['intersected'][pos]).sum()),
        'quantized_vertices_exact': torch.equal(va, vb),
        'quantized_vertex_mismatch_rows': int((va != vb).any(1).sum()),
        'quantized_vertex_max_abs_difference': int((va.short() - vb.short()).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('meshes', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--resolution', type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    o_voxel.convert.mesh_to_flexible_dual_grid(
        torch.tensor([[.1, .1, .1], [.8, .2, .2], [.2, .8, .7]]),
        torch.tensor([[0, 1, 2]]), grid_size=16, aabb=[[0]*3, [1]*3], backend='cuda')
    torch.cuda.synchronize()
    for mesh in args.meshes:
        folder = args.output_dir / mesh.stem
        folder.mkdir(parents=True, exist_ok=True)
        results = {'mesh': str(mesh), 'resolution': args.resolution,
                   'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                   'repeats': args.repeats, 'timings': {}}
        outputs, raw = {}, {}
        for loader in ('blender', 'glb'):
            load_times, convert_times, total_times = [], [], []
            for rep in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if loader == 'blender':
                    dump_path = folder / 'blender.pkl'
                    with (folder / f'blender_{rep}.log').open('w') as log:
                        subprocess.run([args.blender, '-b', '--python-exit-code', '1', '-P',
                                        str(ROOT / 'data_toolkit/blender_script/dump_pbr.py'), '--',
                                        '--object', str(mesh), '--output_path', str(dump_path),
                                        '--geometry-only'], check=True, stdout=log, stderr=subprocess.STDOUT)
                    with dump_path.open('rb') as stream:
                        dump = pickle.load(stream)
                else:
                    dump = load_glb_geometry(mesh)
                load_end = time.perf_counter()
                # convert_geometry replaces normalized arrays, so shallow-copy
                # objects to retain the raw arrays for untimed comparisons.
                fresh = {**dump, 'objects': [dict(obj) for obj in dump['objects']]}
                packed = pipeline.convert_geometry(fresh, args.resolution, backend='cuda')
                torch.cuda.synchronize()
                end = time.perf_counter()
                load_times.append(load_end - start)
                convert_times.append(end - load_end)
                total_times.append(end - start)
                print(mesh.stem, loader, rep, 'load', load_times[-1],
                      'convert', convert_times[-1], 'total', total_times[-1], flush=True)
            outputs[loader], raw[loader] = packed, dump
            results['timings'][loader] = {
                'load_seconds': load_times, 'convert_seconds': convert_times,
                'total_seconds': total_times, 'median_load_seconds': statistics.median(load_times),
                'median_total_seconds': statistics.median(total_times)}
            torch.save({'coords': packed[0], **packed[1]}, folder / f'{loader}_packed.pt')
        a, b = raw['blender']['objects'], raw['glb']['objects']
        results['input'] = {
            'triangles': sum(len(obj['faces']) for obj in b),
            'vertices': sum(len(obj['vertices']) for obj in b),
            'objects_equal': len(a) == len(b),
            'faces_exact': len(a) == len(b) and all(np.array_equal(x['faces'], y['faces']) for x, y in zip(a, b)),
            'vertices_exact': len(a) == len(b) and all(np.array_equal(x['vertices'], y['vertices']) for x, y in zip(a, b)),
            'vertices_max_abs_difference': max(float(np.abs(x['vertices'] - y['vertices']).max()) for x, y in zip(a, b)),
        }
        results['comparison'] = compare_packed(outputs['blender'], outputs['glb'], args.resolution)
        results['speedup'] = (results['timings']['blender']['median_total_seconds'] /
                              results['timings']['glb']['median_total_seconds'])
        (folder / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
