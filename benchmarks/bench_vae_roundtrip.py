#!/usr/bin/env python3
"""Encode/decode aligned real geometry/PBR, comparing full and halo chunking.

Each invocation handles one phase so CUDA peak memory is independently measured.
Saved latents deliberately omit encoder caches: geometry predicts subdivisions,
texture uses explicit coordinate-aligned guidance rather than hidden caches.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'o-voxel'))
import o_voxel
from trellis2 import models
from trellis2.modules import sparse as sp
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder


def order(coords):
    xyz = coords.long()
    side = int(xyz[:, 1:].max()) + 1
    keys = ((xyz[:, 0] * side + xyz[:, 1]) * side + xyz[:, 2]) * side + xyz[:, 3]
    return torch.argsort(keys)


def sparse(pack, device='cuda'):
    return sp.SparseTensor(pack['feats'].to(device), pack['coords'].to(device))


def pack(x):
    return {'coords': x.coords.cpu(), 'feats': x.feats.float().cpu()}


def read_input(path, kind):
    coords, attr = o_voxel.io.read_vxz(str(path))
    coords = torch.cat((torch.zeros_like(coords[:, :1]), coords), 1).int()
    if kind == 'geometry':
        feats = attr['vertices'].float() / 255
        bits = attr['intersected']
        flags = torch.cat([bits % 2, bits // 2 % 2, bits // 4 % 2], -1).bool()
        return {'coords': coords, 'feats': feats}, flags
    feats = torch.cat([attr[k] for k in ('base_color', 'metallic', 'roughness', 'alpha')], -1)
    return {'coords': coords, 'feats': feats.float() / 255 * 2 - 1}, None


def input_guides(coords, levels=4):
    """Coarse -> fine subdivision masks from source voxels, sorted by XYZ."""
    result = []
    coords = coords.int().cuda()
    for _ in range(levels):
        parents = coords.clone()
        parents[:, 1:] //= 2
        parents, idx = torch.unique(parents, dim=0, return_inverse=True)
        bit = ((coords[:, 1:] % 2) * coords.new_tensor([1, 2, 4])).sum(1)
        flags = torch.zeros((len(parents), 8), device='cuda', dtype=torch.bool)
        flags[idx, bit] = True
        result.append(sp.SparseTensor(flags, parents))
        coords = parents
    result = result[::-1]
    # Upsampling emits children in parent/bit order, not global XYZ order.
    # Align each next mask to exactly that order, as geometry decoder does.
    for i in range(1, len(result)):
        prev = result[i - 1]
        parent, bit = torch.nonzero(prev.feats, as_tuple=True)
        children = prev.coords[parent].clone()
        children[:, 1:] = children[:, 1:] * 2 + torch.stack(
            [bit // 2**axis % 2 for axis in range(3)], -1)
        result[i] = align_latent_to_guide(result[i], sp.SparseTensor(
            torch.empty((len(children), 0), device='cuda'), children))
    return result


def align_latent_to_guide(z, guide):
    a, b = order(z.coords), order(guide.coords)
    if not torch.equal(z.coords[a], guide.coords[b]):
        raise ValueError('Latent and guide parent coordinate sets differ')
    dest = torch.empty_like(a)
    dest[b] = a
    return sp.SparseTensor(z.feats[dest], guide.coords)


def compare(a, b):
    ai, bi = order(a['coords']), order(b['coords'])
    ca, cb = a['coords'][ai], b['coords'][bi]
    same = torch.equal(ca, cb)
    result = {'coordinates_equal': same, 'reference_voxels': len(ca), 'chunked_voxels': len(cb)}
    if not same:
        side = max(int(ca[:, 1:].max()), int(cb[:, 1:].max())) + 1
        def keys(c):
            c = c.long()
            return ((c[:, 0] * side + c[:, 1]) * side + c[:, 2]) * side + c[:, 3]
        ka, kb = keys(ca), keys(cb)
        pos = torch.searchsorted(kb, ka)
        valid = pos < len(kb)
        valid[valid.clone()] &= kb[pos[valid]] == ka[valid]
        ai, bi = ai[valid], bi[pos[valid]]
    delta = (a['feats'][ai] - b['feats'][bi]).abs()
    result.update(common_voxels=len(ai), coordinate_iou=len(ai)/(len(ca)+len(cb)-len(ai)),
                  max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                  rmse=float(delta.square().mean().sqrt()),
                  exact_fraction=float((delta == 0).float().mean()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--phase', choices=['encode', 'decode', 'mesh'], required=True)
    parser.add_argument('--kind', choices=['geometry', 'texture'], required=True)
    parser.add_argument('--models', type=Path, default=Path('/mnt/pfs/users/chenjiahao/TRELLIS2Plus/.hf_models/ckpts'))
    parser.add_argument('--chunk-size', type=lambda s: 'auto' if s == 'auto' else int(s))
    parser.add_argument('--chunk-memory-mb', type=float)
    parser.add_argument('--chunk-threshold-mb', type=float, default=6144)
    parser.add_argument('--context', choices=['inference', 'no_grad'], default='inference')
    parser.add_argument('--guidance', choices=['input', 'geometry'], default='input')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3, help='Warm up once, then time fresh-cache runs')
    args = parser.parse_args()
    torch.set_num_threads(4)
    phase = args.phase
    label = 'auto' if args.chunk_size == 'auto' else ('chunked' if args.chunk_size else 'full')
    name = args.kind + '_' + phase + ('_' + label if phase == 'decode' else '')
    if args.kind == 'texture' and phase == 'decode':
        name += '_' + args.guidance
    prefix = 'shape' if args.kind == 'geometry' else 'tex'
    if phase != 'mesh':
        model = models.from_pretrained(str(args.models / f'{prefix}_{"enc" if phase == "encode" else "dec"}_next_dc_f16c32_fp16')).cuda().eval()
        if hasattr(model, 'set_resolution'):
            model.set_resolution(args.resolution)
    if phase == 'encode':
        source, flags = read_input(args.folder / (args.kind + '.vxz'), args.kind)
        x = sparse(source)
        if flags is not None:
            flags = x.replace(flags.cuda())
    elif phase == 'decode':
        z = sparse(torch.load(args.folder / (args.kind + '_latent.pt'), weights_only=True))
        guides = None
        if args.kind == 'texture':
            if args.guidance == 'input':
                source, _ = read_input(args.folder / 'texture.vxz', 'texture')
                guides = input_guides(source['coords'])
            else:
                guide_path = args.folder / 'geometry_decode_chunked.pt'
                guides = [sparse(p) for p in torch.load(guide_path, weights_only=True)['subs']]
            z = align_latent_to_guide(z, guides[0])
    else:
        raw = torch.load(args.folder / 'geometry_decode_chunked.pt', weights_only=True)
        h = sparse(raw)
    context = torch.inference_mode if args.context == 'inference' else torch.no_grad
    def run():
        if phase == 'encode':
            fresh = sp.SparseTensor(x.feats, x.coords)
            return model(fresh, fresh.replace(flags.feats)) if flags is not None else model(fresh)
        elif phase == 'decode':
            fresh = sp.SparseTensor(z.feats, z.coords)
            kwargs = dict(chunk_size=args.chunk_size, chunk_memory_mb=args.chunk_memory_mb,
                          chunk_threshold_mb=args.chunk_threshold_mb)
            if args.kind == 'geometry':
                # Raw network outputs allow numerical comparisons before the
                # discontinuous sign tests used for mesh connectivity.
                return SparseUnetVaeDecoder.forward(model, fresh, return_subs=True, **kwargs)
            else:
                return model(fresh, guide_subs=guides, **kwargs)
        else:
            vertices = 2 * h.feats[:, :3].sigmoid() - .5
            intersected = h.feats[:, 3:6] > 0
            quad_lerp = torch.nn.functional.softplus(h.feats[:, 6:7])
            return o_voxel.convert.flexible_dual_grid_to_mesh(
                h.coords[:, 1:], vertices, intersected, quad_lerp,
                aabb=[[-.5]*3, [.5]*3], grid_size=args.resolution, train=False)
    with context():
        warm = run()
        del warm
        times, peaks, reserved = [], [], []
        for rep in range(args.repeats):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            started = time.perf_counter()
            value = run()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            peaks.append(torch.cuda.max_memory_allocated() / 1024**3)
            reserved.append(torch.cuda.max_memory_reserved() / 1024**3)
            if rep != args.repeats - 1:
                del value
        if phase == 'mesh': mesh = value
        elif phase == 'decode' and args.kind == 'geometry': out, subs = value
        else: out = value
    import statistics
    seconds = statistics.median(times)
    result = {'phase': phase, 'kind': args.kind, 'context': args.context,
              'seconds': seconds, 'baseline_cuda_gib': baseline / 1024**3,
              'peak_cuda_gib': max(peaks), 'seconds_runs': times,
              'peak_reserved_cuda_gib': max(reserved),
              'chunk_size': args.chunk_size, 'chunk_memory_mb': args.chunk_memory_mb}
    if phase == 'mesh':
        result.update(vertices=len(mesh[0]), faces=len(mesh[1]), finite=bool(torch.isfinite(mesh[0]).all()))
        torch.save({'vertices': mesh[0].cpu(), 'faces': mesh[1].cpu()}, args.folder / 'decoded_mesh.pt')
    else:
        result.update(output_rows=len(out.feats), finite=bool(torch.isfinite(out.feats).all()))
        output = pack(out)
        if phase == 'encode':
            torch.save(output, args.folder / (args.kind + '_latent.pt'))
            result['input_rows'] = len(x.feats)
        else:
            result['chunk_stats'] = getattr(model, 'last_chunk_stats', {})
            if args.kind == 'geometry':
                output['subs'] = [pack(s) for s in subs]
            torch.save(output, args.folder / (name + '.pt'))
            # Both explicit chunking and auto chunking compare against the
            # same unchunked raw decoder output.
            ref_name = f'{args.kind}_decode_full'
            if args.kind == 'texture':
                ref_name += f'_{args.guidance}'
            ref_path = args.folder / f'{ref_name}.pt'
            if args.chunk_size and ref_path.exists():
                reference = torch.load(ref_path, weights_only=True)
                result['full_vs_chunked'] = compare(reference, output)
                if args.kind == 'geometry':
                    result['subdivision_comparison'] = [compare(a, b) for a, b in zip(reference['subs'], output['subs'])]
            if args.kind == 'texture' and args.guidance == 'input':
                result['reconstruction'] = compare(source, output)
                if result['reconstruction']['coordinates_equal']:
                    delta = ((source['feats'][order(source['coords'])] - output['feats'][order(output['coords'])]) * .5)
                    result['material_mae'] = {key: float(delta[:, sl].abs().mean()) for key, sl in
                        [('rgb', slice(0, 3)), ('metallic', slice(3, 4)), ('roughness', slice(4, 5)), ('alpha', slice(5, 6))]}
    (args.folder / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
