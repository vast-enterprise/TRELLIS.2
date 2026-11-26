#!/usr/bin/env python3
"""Build aligned geometry/PBR VXZ inputs from an existing textured Blender dump."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import create_geo_pbr as pipeline
import o_voxel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--resolution', type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with args.dump.open('rb') as stream:
        dump = pickle.load(stream)
    pipeline._pbr.normalize_dump_geometry(dump)
    load_time = time.perf_counter() - start
    torch.cuda.init()
    torch.cuda.synchronize()
    start = time.perf_counter()
    pcoord, pattr = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump, grid_size=args.resolution, aabb=[[-.5]*3, [.5]*3],
        color_space='srgb', add_emission=True, timing=True)
    torch.cuda.synchronize()
    ptime = time.perf_counter() - start
    start = time.perf_counter()
    gcoord, gattr = pipeline.convert_geometry(dump, args.resolution, backend='cuda')
    torch.cuda.synchronize()
    gtime = time.perf_counter() - start
    counts = {'texture_before_intersection': len(pcoord), 'geometry_before_intersection': len(gcoord)}
    start = time.perf_counter()
    pcoord, pattr, gcoord, gattr, common = pipeline.intersect_voxels(pcoord, pattr, gcoord, gattr)
    torch.cuda.synchronize()
    join_time = time.perf_counter() - start
    start = time.perf_counter()
    # Writers may reorder rows; compare coordinate sets rather than row numbers.
    for name, coord, attr in [('texture', pcoord, pattr), ('geometry', gcoord, gattr)]:
        if name == 'texture':
            attr = {key: attr[key] for key in ('base_color', 'metallic', 'roughness', 'alpha')}
        o_voxel.io.write_vxz(str(args.output_dir / (name + '.vxz')), coord, attr, compression='zstd')
    result = {**counts, 'common_voxels': common, 'resolution': args.resolution,
              'texture_voxelize_seconds': ptime, 'geometry_voxelize_seconds': gtime,
              'dump_read_normalize_seconds': load_time, 'intersection_seconds': join_time,
              'vxz_write_seconds': time.perf_counter() - start,
              'source_dump': str(args.dump), 'color_space': 'srgb',
              'material_channels_zeroed': False, 'normal_preprocessing': dump.get('normal_preprocessing')}
    (args.output_dir / 'prepare.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
