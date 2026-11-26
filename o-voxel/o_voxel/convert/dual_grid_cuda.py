"""Sparse CUDA scan conversion, QEF reduction and constrained minimization.

Only small grid parameters and hash-overflow counters cross back to the host.
Output rows are ordered by flattened XYZ key, independently of CUDA scheduling.
The CPU implementation is retained as the reference backend.
"""
import math
import time

import torch
from .. import _C


@torch.no_grad()
def mesh_to_dual_grid_cuda(vertices, faces, voxel_size, grid_size,
                           face_weight, boundary_weight, regularization_weight,
                           timing=False):
    if not hasattr(_C, 'dual_grid_accumulate_cuda'):
        raise RuntimeError('Rebuild o-voxel to enable the CUDA geometry backend')
    clock = time.perf_counter()
    device = vertices.device if vertices.is_cuda else torch.device('cuda', torch.cuda.current_device())
    vertices = vertices.to(device=device, dtype=torch.float32).contiguous()
    faces = faces.to(device=device, dtype=torch.int32).contiguous()
    size = voxel_size.cpu().tolist()
    grid = grid_size.cpu().tolist()
    if any(not math.isfinite(x) or x <= 0 for x in size):
        raise ValueError('Voxel sizes must be finite and positive')
    if any(x <= 0 for x in grid) or math.prod(grid) >= 2**63:
        raise ValueError('CUDA geometry requires positive grid dimensions with volume < 2**63')
    if len(vertices) >= 2**31 or len(faces) >= 2**29:
        raise ValueError('Mesh exceeds CUDA int32 indexing limits')
    if not torch.isfinite(vertices).all():
        raise ValueError('Non-finite mesh vertices')
    if faces.numel() and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Face index outside vertex array')
    if not all(math.isfinite(w) and w >= 0 for w in (face_weight, boundary_weight, regularization_weight)):
        raise ValueError('QEF weights must be finite and non-negative')
    # Match the CPU legacy contract: face_weight enables the face term;
    # its magnitude is not applied by the reference implementation.
    empty_edges = torch.empty(0, dtype=torch.int64, device=device)
    empty_rows = torch.empty(0, dtype=torch.int32, device=device)
    empty_data = torch.empty((0, 20), dtype=torch.float64, device=device)
    empty_flags = torch.empty(0, dtype=torch.int32, device=device)
    overflow = torch.zeros(1, dtype=torch.int32, device=device)

    def stamp(label):
        nonlocal clock
        if timing:
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            print(f'CUDA {label}: {now - clock:.6f}s', flush=True)
            clock = now

    stamp('upload/validate')
    # Sparse keys only during discovery; allocate QEF storage after compaction.
    estimate = min(max(4096, len(faces) * 2), max(4096, max(grid)**2 * 8))
    capacity = 1 << (estimate - 1).bit_length()
    while True:
        if capacity >= 2**31:
            raise RuntimeError('CUDA voxel hash table exceeds int32 capacity')
        keys = torch.full((capacity,), -1, dtype=torch.int64, device=device)
        overflow.zero_()
        _C.dual_grid_accumulate_cuda(vertices, faces, empty_edges, size, grid,
                                    keys, empty_rows, empty_data, empty_flags, overflow, 0, boundary_weight)
        if not overflow.item():
            break
        del keys
        capacity *= 2
    active = torch.nonzero(keys != -1).flatten()
    order = torch.argsort(keys[active])
    active = active[order]
    voxel_keys = keys[active]
    coords = torch.stack((voxel_keys // (grid[1] * grid[2]),
                          voxel_keys // grid[2] % grid[1], voxel_keys % grid[2]), dim=1).int()
    rows = torch.empty(capacity, dtype=torch.int32, device=device)
    rows[active] = torch.arange(len(active), dtype=torch.int32, device=device)
    del active, order, voxel_keys
    data = torch.zeros((len(coords), 20), dtype=torch.float64, device=device)
    flags = torch.zeros(len(coords), dtype=torch.int32, device=device)
    stamp('voxel discovery/merge')
    _C.dual_grid_accumulate_cuda(vertices, faces, empty_edges, size, grid,
                                keys, rows, data, flags, overflow, 1, boundary_weight)
    stamp('intersection QEF')
    if face_weight > 0:
        _C.dual_grid_accumulate_cuda(vertices, faces, empty_edges, size, grid,
                                    keys, rows, data, flags, overflow, 2, boundary_weight)
    stamp('face QEF')
    if boundary_weight > 0 and faces.numel():
        # Edge counting is an exact integer sort/reduction on GPU. No welding:
        # edges across separate source vertex indices remain separate.
        f = faces.long()
        other = f.roll(-1, dims=1)
        edge_keys = (torch.minimum(f, other) << 32) | torch.maximum(f, other)
        edge_keys, counts = torch.unique(edge_keys.flatten(), return_counts=True)
        boundary = edge_keys[counts == 1].contiguous()
        del f, other, edge_keys, counts
        stamp('boundary edge sort/reduce')
        _C.dual_grid_accumulate_cuda(vertices, faces, boundary, size, grid,
                                    keys, rows, data, flags, overflow, 3, boundary_weight)
        del boundary
    stamp('boundary QEF')
    dual, intersected = _C.dual_grid_solve_cuda(coords, data, flags, size, regularization_weight)
    stamp('constrained QEF solve')
    return coords, dual, intersected
