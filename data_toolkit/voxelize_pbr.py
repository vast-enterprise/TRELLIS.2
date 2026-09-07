import os
import copy
import sys
import importlib
import argparse
import json
import pandas as pd
import pickle
import numpy as np
import torch
from easydict import EasyDict as edict
from functools import partial
import o_voxel


VXZM_NORMAL_SOURCE = 'surface_authored_with_winding_guard'
VXZM_RECORD_LAYOUT = [
    ['base_color', 3],
    ['metallic', 1],
    ['roughness', 1],
    ['emissive', 3],
    ['alpha', 1],
    ['normal', 3],
]


def _expected_vxzm_config(resolution):
    """Stable signature persisted in CSV metadata for cache filtering."""
    return json.dumps({
        'grid_size': [int(resolution)] * 3,
        'region_resolution': int(opt.region_resolution),
        'normal_source': VXZM_NORMAL_SOURCE,
        'record_layout': VXZM_RECORD_LAYOUT,
        'color_space': opt.color_space,
        'add_emission': bool(opt.add_emission),
        'cluster_angle_degrees': float(opt.cluster_angle_degrees),
        'max_records_per_voxel': int(opt.max_records_per_voxel),
        'max_total_records': int(opt.max_total_records),
    }, sort_keys=True, separators=(',', ':'))


def _vxzm_config_from_info(info):
    """Build the cache signature from authoritative VXZM header values."""
    metadata = info.get('metadata')
    if not isinstance(metadata, dict):
        return None
    try:
        add_emission = metadata['add_emission']
        if not isinstance(add_emission, bool):
            return None
        record_layout = [
            [entry['name'], int(entry['channels'])]
            for entry in info['record_layout']
        ]
        return json.dumps({
            'grid_size': [int(value) for value in info['grid_size']],
            'region_resolution': int(info['region_resolution']),
            'normal_source': info['normal_source'],
            'record_layout': record_layout,
            'color_space': metadata['color_space'],
            'add_emission': add_emission,
            'cluster_angle_degrees': float(metadata['cluster_angle_degrees']),
            'max_records_per_voxel': int(metadata['max_records_per_voxel']),
            'max_total_records': int(metadata['max_total_records']),
        }, sort_keys=True, separators=(',', ':'))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _vxzm_matches_options(info, resolution):
    """Return whether a cached VXZM was produced with current CLI semantics."""
    return (info.get('format') == 'VXZM' and
            _vxzm_config_from_info(info) == _expected_vxzm_config(resolution))


def _select_pending_metadata(metadata):
    """Select assets whose requested VXZ/VXZM outputs are not reusable."""
    mask = np.zeros(len(metadata), dtype=bool)
    for res in opt.resolution:
        if opt.output_format in ('vxz', 'both'):
            column = f'pbr_voxelized_{res}'
            if column not in metadata.columns:
                mask[:] = True
            else:
                mask |= metadata[column] != True
        if opt.output_format in ('vxzm', 'both'):
            status = f'pbrm_voxelized_{res}'
            config = f'pbrm_config_{res}'
            if status not in metadata.columns or config not in metadata.columns:
                mask[:] = True
            else:
                mask |= ((metadata[status] != True) |
                         (metadata[config] != _expected_vxzm_config(res)))
    return metadata[mask]


def _write_metadata_part(pbr_voxelized, resolution, output_root, rank):
    """Write one merge-safe VXZ/VXZM metadata part for a resolution."""
    columns = ['sha256']
    rename = {}
    for source, target in (
        (f'pbr_voxelized_{resolution}', 'pbr_voxelized'),
        (f'num_pbr_voxels_{resolution}', 'num_pbr_voxels'),
        (f'pbrm_voxelized_{resolution}', 'pbrm_voxelized'),
        (f'num_pbrm_records_{resolution}', 'num_pbrm_records'),
        (f'num_pbrm_unique_voxels_{resolution}', 'num_pbrm_unique_voxels'),
        (f'pbrm_config_{resolution}', 'pbrm_config'),
    ):
        if source in pbr_voxelized.columns:
            columns.append(source)
            rename[source] = target
    if len(columns) == 1:
        return None
    complete = pbr_voxelized[columns].copy()
    status_columns = [name for name in columns if name != 'sha256' and
                      name.startswith(('pbr_voxelized_', 'pbrm_voxelized_'))]
    complete = complete[complete[status_columns].eq(True).any(axis=1)]
    if len(complete) == 0:
        return None
    path = os.path.join(output_root, f'pbr_voxels_{resolution}',
                        'new_records', f'part_{rank}.csv')
    complete.rename(columns=rename).to_csv(path, index=False)
    return path


def _pbr_voxelize(file, metadatum, pbr_dump_root, root):
    sha256 = metadatum['sha256']
    try:
        pack = {'sha256': sha256}
        dump = None
        for res in opt.resolution:
            need_process = False
            want_vxz = opt.output_format in ('vxz', 'both')
            want_vxzm = opt.output_format in ('vxzm', 'both')
            vxz_path = os.path.join(root, f'pbr_voxels_{res}', f'{sha256}.vxz')
            vxzm_path = os.path.join(root, f'pbr_voxels_{res}', f'{sha256}.vxzm')

            # check if already processed
            vxz_ok = not want_vxz
            vxzm_ok = not want_vxzm
            if want_vxz and os.path.exists(vxz_path):
                try:
                    info = o_voxel.io.read_vxz_info(vxz_path)
                    pack[f'pbr_voxelized_{res}'] = True
                    pack[f'num_pbr_voxels_{res}'] = info['num_voxel']
                    vxz_ok = True
                except Exception as e:
                    print(f'Error reading {sha256}.vxz: {e}')
            if want_vxzm and os.path.exists(vxzm_path):
                try:
                    info = o_voxel.io.read_vxzm_info(vxzm_path)
                    if _vxzm_matches_options(info, res):
                        pack[f'pbrm_voxelized_{res}'] = True
                        pack[f'num_pbrm_records_{res}'] = info['num_records']
                        pack[f'num_pbrm_unique_voxels_{res}'] = info['num_unique_voxels']
                        pack[f'pbrm_config_{res}'] = _vxzm_config_from_info(info)
                        vxzm_ok = True
                    else:
                        print(f'Existing {sha256}.vxzm uses different options; regenerating')
                except Exception as e:
                    print(f'Error reading {sha256}.vxzm: {e}')
            need_process = not (vxz_ok and vxzm_ok)

            # process if necessary
            if need_process:
                if dump is None:
                    with open(os.path.join(pbr_dump_root, 'pbr_dumps', f'{sha256}.pickle'), 'rb') as f:
                        dump = pickle.load(f)
                    # Fix dump alpha map
                    for mat in dump['materials']:
                        if mat['alphaTexture'] is not None and mat['alphaMode'] == 'OPAQUE':
                            mat['alphaMode'] = 'BLEND'
                    dump['materials'].append({
                        "baseColorFactor": [0.8, 0.8, 0.8],
                        "alphaFactor": 1.0,
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.5,
                        "alphaMode": "OPAQUE",
                        "alphaCutoff": 0.5,
                        "baseColorTexture": None,
                        "alphaTexture": None,
                        "metallicTexture": None,
                        "roughnessTexture": None,
                        "emissiveFactor": [0.0, 0.0, 0.0],
                        "emissiveTexture": None,
                        "emissionStrength": 0.0,
                        "shaderType": "Principled",
                    })      # append default material
                    dump['objects'] = [
                        obj for obj in dump['objects']
                        if obj['vertices'].size != 0 and obj['faces'].size != 0
                    ]
                    vertices = torch.from_numpy(np.concatenate([obj['vertices'] for obj in dump['objects']], axis=0)).float()
                    vertices_min = vertices.min(dim=0)[0]
                    vertices_max = vertices.max(dim=0)[0]
                    center = (vertices_min + vertices_max) / 2
                    scale = 0.99999 / (vertices_max - vertices_min).max()
                    for obj in dump['objects']:
                        obj['vertices'] = (torch.from_numpy(obj['vertices']).float() - center) * scale
                        obj['vertices'] = obj['vertices'].numpy()
                        obj['mat_ids'][obj['mat_ids'] == -1] = len(dump['materials']) - 1
                        assert np.all(obj['mat_ids'] >= 0), 'invalid mat_ids'
                        assert np.all(obj['vertices'] >= -0.5) and np.all(obj['vertices'] <= 0.5), 'vertices out of range'

                common = dict(grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                              mip_level_offset=0, verbose=False, timing=False,
                              add_emission=opt.add_emission, color_space=opt.color_space)
                if want_vxz and not vxz_ok:
                    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(dump, **common)
                    del attr['normal']
                    del attr['emissive']
                    o_voxel.io.write_vxz(vxz_path, coord, attr)
                    pack[f'pbr_voxelized_{res}'] = True
                    pack[f'num_pbr_voxels_{res}'] = len(coord)
                if want_vxzm and not vxzm_ok:
                    coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr_multi(
                        dump, cluster_angle_degrees=opt.cluster_angle_degrees,
                        max_records_per_voxel=opt.max_records_per_voxel,
                        max_total_records=opt.max_total_records, **common)
                    # Keep the full PBR record in VXZM. ``base_color`` already
                    # contains emission when add_emission is enabled, while
                    # the separate emissive field remains useful for debugging
                    # and future consumers. The default training feature list
                    # intentionally omits it, so current supervision is
                    # unchanged. Legacy VXZ retains its historical layout.
                    o_voxel.io.write_vxzm(vxzm_path, coord, attr, grid_size=res,
                                           region_resolution=opt.region_resolution,
                                           metadata={
                                               'cluster_angle_degrees': opt.cluster_angle_degrees,
                                               'color_space': opt.color_space,
                                               'add_emission': opt.add_emission,
                                               'max_records_per_voxel': opt.max_records_per_voxel,
                                               'max_total_records': opt.max_total_records,
                                           })
                    info = o_voxel.io.read_vxzm_info(vxzm_path)
                    pack[f'pbrm_voxelized_{res}'] = True
                    pack[f'num_pbrm_records_{res}'] = info['num_records']
                    pack[f'num_pbrm_unique_voxels_{res}'] = info['num_unique_voxels']
                    pack[f'pbrm_config_{res}'] = _vxzm_config_from_info(info)

        return pack
    except Exception as e:
        print(f'Error voxelizing {sha256}: {e}')
        return {'sha256': sha256, 'error': str(e)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to load mesh dumps')
    parser.add_argument('--pbr_voxel_root', type=str, default=None,
                        help='Directory to save voxelized pbr attributes')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--resolution', type=str, default=1024)
    parser.add_argument('--color_space', choices=['linear', 'srgb', 'agx'], default='agx',
                        help='Color encoding written into the base_color voxel attribute')
    parser.add_argument('--output-format', choices=['vxz', 'vxzm', 'both'], default='vxz')
    parser.add_argument('--cluster-angle-degrees', type=float, default=15.0)
    parser.add_argument('--region-resolution', type=int, default=256)
    parser.add_argument('--max-records-per-voxel', type=int, default=0,
                        help='Fail if one voxel produces more clusters (0 disables the limit)')
    parser.add_argument('--max-total-records', type=int, default=0,
                        help='Fail if input samples or output VXZM records exceed this count (0 disables the limit)')
    emission_group = parser.add_mutually_exclusive_group()
    emission_group.add_argument('--add_emission', dest='add_emission', action='store_true',
                                help='Add Principled emission to base color before color-space conversion')
    emission_group.add_argument('--no-add-emission', dest='add_emission', action='store_false',
                                help='Do not add emission to base color')
    parser.set_defaults(add_emission=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.resolution = sorted([int(x) for x in opt.resolution.split(',')], reverse=True)
    if not np.isfinite(opt.cluster_angle_degrees) or not (0.0 < opt.cluster_angle_degrees < 180.0):
        parser.error('--cluster-angle-degrees must be in (0, 180)')
    if (opt.region_resolution < 4 or opt.region_resolution > 1024 or
            opt.region_resolution & (opt.region_resolution - 1)):
        parser.error('--region-resolution must be a power of two in [4, 1024]')
    if opt.max_records_per_voxel < 0 or opt.max_total_records < 0:
        parser.error('--max-records-per-voxel and --max-total-records must be non-negative')
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root
    opt.pbr_voxel_root = opt.pbr_voxel_root or opt.root

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}', 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')).set_index('sha256'))
    for res in opt.resolution:
        if os.path.exists(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}', 'metadata.csv')):
            pbr_voxel_metadata = pd.read_csv(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}','metadata.csv')).set_index('sha256')
            pbr_voxel_metadata = pbr_voxel_metadata.rename(columns={
                'pbr_voxelized': f'pbr_voxelized_{res}',
                'num_pbr_voxels': f'num_pbr_voxels_{res}',
                'pbrm_voxelized': f'pbrm_voxelized_{res}',
                'num_pbrm_records': f'num_pbrm_records_{res}',
                'num_pbrm_unique_voxels': f'num_pbrm_unique_voxels_{res}',
                'pbrm_config': f'pbrm_config_{res}',
            })
            metadata = metadata.combine_first(pbr_voxel_metadata)
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['pbr_dumped'] == True]
        metadata = _select_pending_metadata(metadata)
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_pbr_voxelize, pbr_dump_root=opt.pbr_dump_root, root=opt.pbr_voxel_root)
    pbr_voxelized = dataset_utils.foreach_instance(metadata, None, func, max_workers=opt.max_workers, no_file=True, desc='Voxelizing')
    if 'error' in pbr_voxelized.columns:
        errors = pbr_voxelized[pbr_voxelized['error'].notna()]
        with open('errors.txt', 'w') as f:
            f.write('\n'.join(errors['sha256'].tolist()))
    for res in opt.resolution:
        # VXZ and VXZM share one metadata.csv. Emit one combined row so merge
        # ordering can never replace a complete row with a partial row.
        _write_metadata_part(pbr_voxelized, res, opt.pbr_voxel_root, opt.rank)
