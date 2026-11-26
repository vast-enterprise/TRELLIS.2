#!/usr/bin/env python3
"""Compare original CPU and CUDA geometry with identical unmodified triangles.

Default: no weld/normal repair/material/UV export; resolution 1024. Includes
process wall time for Blender (import + extraction + pickle), separate input
preparation and CPU/CUDA sampling timings, and coordinate/flag/position checks.
First run initializes CUDA/native imports before timing sampling. No PBR work,
VXZ compression, result sorting, or comparison work enters sampling timings.
"""
from __future__ import annotations
import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import create_geo_pbr
import o_voxel


def dump_mesh(glb, out, reference, blender):
    script = (ROOT / 'benchmarks/blender_geometry_reference.py' if reference else
              ROOT / 'data_toolkit/blender_script/dump_pbr.py')
    cmd = [blender, '-b', '--python-exit-code', '1', '-P', str(script), '--',
           '--object', str(glb), '--output_path', str(out)]
    if not reference:
        cmd.append('--geometry-only')
    with out.with_suffix('.log').open('w') as log:
        start = time.perf_counter()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        seconds = time.perf_counter() - start
    with out.open('rb') as stream:
        data = pickle.load(stream)
    return data, seconds


def sort_result(result, resolution):
    c,v,f = [x.cpu() for x in result]
    order = torch.argsort(c.long() @ torch.tensor([resolution**2, resolution, 1]))
    return c[order],v[order],f[order]


def compare_results(cpu, gpu, resolution):
    c0,v0,f0 = sort_result(cpu, resolution)
    c1,v1,f1 = sort_result(gpu, resolution)
    k0 = c0.long() @ torch.tensor([resolution**2,resolution,1])
    k1 = c1.long() @ torch.tensor([resolution**2,resolution,1])
    pos = torch.searchsorted(k1,k0)
    valid = pos < len(k1)
    valid[valid.clone()] &= k1[pos[valid]] == k0[valid]
    indices = pos[valid]
    error = ((v0[valid]-v1[indices])*resolution).abs()
    q0 = ((v0[valid]*resolution-c0[valid]).clamp(0,1)*255).byte()
    q1 = ((v1[indices]*resolution-c1[indices]).clamp(0,1)*255).byte()
    return {
        'cpu_voxels': len(c0), 'cuda_voxels': len(c1), 'common_voxels': int(valid.sum()),
        'coordinate_sets_equal': torch.equal(c0,c1),
        'intersection_flag_mismatches': int((f0[valid] != f1[indices]).any(1).sum()),
        'position_abs_error_voxels_mean': float(error.mean()) if error.numel() else 0,
        'position_abs_error_voxels_p99': float(np.quantile(error.numpy(),.99)) if error.numel() else 0,
        'position_abs_error_voxels_max': float(error.max()) if error.numel() else 0,
        'quantized_vertices_equal_fraction': float((q0==q1).all(1).float().mean()) if len(q0) else 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('glb', type=Path)
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reuse-dumps', action='store_true')
    parser.add_argument('--measure-only-dump', action='store_true', help='Only compare extraction, no sampling')
    parser.add_argument('--pipeline', action='store_true', help='Also time convert_geometry including preparation and quantization')
    args = parser.parse_args()
    if args.resolution < 2 or args.repeats < 1:
        parser.error('resolution >= 2 and repeats >= 1 required')
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {'mesh': str(args.glb), 'resolution': args.resolution,
               'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(), 'timings': {}}
    dumps = {}
    for name in ['reference','optimized']:
        path = args.output_dir / (name+'.pkl')
        if args.reuse_dumps and path.exists():
            with path.open('rb') as stream:
                dumps[name] = pickle.load(stream)
        else:
            dumps[name], seconds = dump_mesh(args.glb, path, name=='reference', args.blender)
            results['timings'][name+'_dump_seconds'] = seconds
    a,b = dumps['reference']['objects'],dumps['optimized']['objects']
    results['dump_comparison'] = {
        'object_count_equal': len(a)==len(b),
        'faces_exact': len(a)==len(b) and all(np.array_equal(x['faces'],y['faces']) for x,y in zip(a,b)),
        'vertices_max_abs': max(float(np.abs(x['vertices']-y['vertices']).max()) for x,y in zip(a,b)),
    }
    if args.measure_only_dump:
        (args.output_dir/'dump_results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results,indent=2),flush=True)
        return
    raw_dump = {**dumps['optimized'], 'objects': [dict(obj) for obj in dumps['optimized']['objects']]}
    start=time.perf_counter()
    v,f=create_geo_pbr._mesh_from_dump(dumps['optimized'])
    results['timings']['prepare_seconds']=time.perf_counter()-start
    results['triangles']=len(f)
    torch.cuda.init()
    warm_v=torch.tensor([[-.1,-.1,0.],[.1,-.1,0.],[0.,.1,.1]])
    kw=dict(grid_size=args.resolution,aabb=[[-.5]*3,[.5]*3],face_weight=1.,boundary_weight=.2,regularization_weight=.01)
    o_voxel.convert.mesh_to_flexible_dual_grid(warm_v,torch.tensor([[0,1,2]]),backend='cuda',**{**kw,'grid_size':16})
    torch.cuda.synchronize()
    samples={}
    for backend in ['cpu','cuda']:
        times=[]
        torch.cuda.reset_peak_memory_stats()
        for rep in range(args.repeats):
            torch.cuda.synchronize();start=time.perf_counter()
            output=o_voxel.convert.mesh_to_flexible_dual_grid(v,f,backend=backend,timing=True,**kw)
            # Include returning the result to CPU, as convert_geometry does.
            output=tuple(x.cpu() for x in output)
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
            print(backend,rep,times[-1],flush=True)
        samples[backend]=output
        results['timings'][backend+'_sampling_seconds']=times
        results[backend+'_peak_cuda_bytes']=torch.cuda.max_memory_allocated()
    for backend,output in samples.items():
        torch.save(dict(zip(('coords','vertices','intersected'),output)),args.output_dir/(backend+'.pt'))
    results['comparison']=compare_results(samples['cpu'],samples['cuda'],args.resolution)
    if args.pipeline:
        for backend in ('cpu','cuda'):
            times=[]
            for rep in range(args.repeats):
                torch.cuda.synchronize(); start=time.perf_counter()
                fresh_dump={**raw_dump,'objects':[dict(obj) for obj in raw_dump['objects']]}
                packed=create_geo_pbr.convert_geometry(fresh_dump,args.resolution,backend=backend)
                torch.cuda.synchronize(); times.append(time.perf_counter()-start)
            results['timings'][backend+'_convert_geometry_seconds']=times
            torch.save({'coords':packed[0],**packed[1]},args.output_dir/(backend+'_packed.pt'))
    results['sampling_speedup']=float(np.median(results['timings']['cpu_sampling_seconds'])/np.median(results['timings']['cuda_sampling_seconds']))
    (args.output_dir/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    main()
