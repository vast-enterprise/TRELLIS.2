"""Bounded inference workspaces with exact per-convolution spatial halos.

Each convolution assembles every input neighbor of a disjoint output core,
evaluates that core plus its halo, and discards halo outputs. The next layer
sees the assembled global result, so a one-layer halo is sufficient even for
deep networks. No feature blending or independent whole-network tiles are used.
Full input/output features still reside on the device; this bounds temporary
convolution and pointwise workspaces, not the size of the final sparse grid.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math

import torch


_ACTIVE = ContextVar('trellis_sparse_inference_chunks', default=None)


@dataclass
class ChunkPolicy:
    max_points: int = 1_000_000
    memory_budget_bytes: int | None = None
    threshold_bytes: int | None = None
    stats: dict = field(default_factory=dict)
    plans: dict = field(default_factory=dict, repr=False)

    def rows(self, feats, channels, *, convolution=False):
        if self.threshold_bytes is not None:
            # Count the actual rows after each predicted subdivision. Future
            # geometry occupancy cannot be inferred reliably from latent size.
            live = torch.cuda.memory_allocated(feats.device) if feats.is_cuda else 0
            workspace = len(feats) * max(channels, 1) * max(feats.element_size(), 4) * 4
            threshold = self.threshold_bytes
            if feats.is_cuda:
                free, _ = torch.cuda.mem_get_info(feats.device)
                threshold = min(threshold, live + int(free * .8))
            estimate = live + workspace
            self.stats['auto_max_voxels'] = max(self.stats.get('auto_max_voxels', 0), len(feats))
            self.stats['auto_max_estimated_peak_mib'] = max(
                self.stats.get('auto_max_estimated_peak_mib', 0), estimate / 1024**2)
            if estimate <= threshold:
                self.stats['auto_skipped_ops'] = self.stats.get('auto_skipped_ops', 0) + 1
                return max(1, len(feats))
        budget = self.memory_budget_bytes
        if budget is None:
            if feats.is_cuda:
                free, _ = torch.cuda.mem_get_info(feats.device)
                budget = min(8 * 1024**3, int(free * .20))
            else:
                budget = 1024**3
        # Include fp32 norm/accumulators, split-K workspaces and halo overhead.
        # The safety factor is deliberately conservative; max_points is an
        # upper bound, not a promise to fit arbitrary backends in this budget.
        bytes_per_row = max(channels, 1) * max(feats.element_size(), 4) * 8
        rows = max(1, min(self.max_points, budget // bytes_per_row))
        return max(1, rows // 256 * 256) if rows >= 256 else rows

    def record(self, name, rows):
        self.stats[name + '_chunks'] = self.stats.get(name + '_chunks', 0) + 1
        self.stats[name + '_max_rows'] = max(self.stats.get(name + '_max_rows', 0), rows)


def active_policy():
    return _ACTIVE.get()


@contextmanager
def inference_chunks(max_points=1_000_000, memory_budget_mb=None, *, threshold_mb=None):
    """Enable chunked operations inside no_grad/inference_mode; thread-local.

    ``max_points`` includes the convolution halo. ``memory_budget_mb`` sets
    the estimated temporary workspace budget; global input/output/model memory is
    additional. With None it adapts to free CUDA memory (20%, capped at 8 GiB).
    """
    if torch.is_grad_enabled():
        raise RuntimeError('Chunked inference requires torch.no_grad() or torch.inference_mode()')
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points <= 0:
        raise ValueError('max_points must be a positive integer')
    if memory_budget_mb is not None and (not math.isfinite(memory_budget_mb) or memory_budget_mb <= 0):
        raise ValueError('memory_budget_mb must be finite and positive')
    if threshold_mb is not None and (not math.isfinite(threshold_mb) or threshold_mb <= 0):
        raise ValueError('threshold_mb must be finite and positive')
    policy = ChunkPolicy(max_points,
                         None if memory_budget_mb is None else int(memory_budget_mb * 1024**2),
                         None if threshold_mb is None else int(threshold_mb * 1024**2))
    if threshold_mb is not None:
        policy.stats.update(auto=True, auto_threshold_mib=threshold_mb)
    token = _ACTIVE.set(policy)
    try:
        yield policy
    finally:
        _ACTIVE.reset(token)


def pointwise(fn, feats, *, channels=None):
    """Run an independent per-row operation without retaining its workspaces."""
    policy = active_policy()
    if policy is None:
        return fn(feats)
    if torch.is_grad_enabled():
        raise RuntimeError('Chunked pointwise operations cannot run with gradients enabled')
    rows = policy.rows(feats, channels or feats.shape[-1])
    if len(feats) <= rows:
        return fn(feats)
    output = None
    for start in range(0, len(feats), rows):
        value = fn(feats[start:start + rows])
        if output is None:
            output = value.new_empty((len(feats), *value.shape[1:]))
        output[start:start + len(value)] = value
        policy.record('pointwise', len(value))
    return output


def _spatial_plan(coords, limit, radius):
    """KD partition of output cores with overlapping input bounding boxes."""
    plans = []
    radius = coords.new_tensor(radius)
    for batch in torch.unique(coords[:, 0]).tolist():
        indices = torch.nonzero(coords[:, 0] == batch).flatten()
        xyz = coords[indices, 1:]
        lo, hi = xyz.amin(0), xyz.amax(0) + 1
        stack = [(indices, indices, lo, hi)]
        while stack:
            core, inputs, lower, upper = stack.pop()
            if len(core) == 1 and len(inputs) > limit:
                raise ValueError('Chunk budget is too small to contain one voxel and its convolution halo')
            if len(inputs) <= limit or len(core) == 1:
                local = torch.nonzero(((coords[inputs, 1:] >= lower) &
                                       (coords[inputs, 1:] < upper)).all(1)).flatten()
                plans.append((inputs, local, inputs[local]))
                continue
            values = coords[core, 1:]
            span = values.amax(0) - values.amin(0)
            axis = int(span.argmax())
            if int(span[axis]) == 0:
                raise ValueError('Sparse coordinates must be unique within each batch')
            # Coordinate midpoint avoids splitting identical spatial slices.
            split = (values[:, axis].min() + values[:, axis].max() + 1) // 2
            left = values[:, axis] < split
            for child, is_left in ((core[left], True), (core[~left], False)):
                child_lo, child_hi = lower.clone(), upper.clone()
                if is_left:
                    child_hi[axis] = split
                else:
                    child_lo[axis] = split
                in_halo = ((coords[inputs, 1:] >= child_lo - radius) &
                           (coords[inputs, 1:] < child_hi + radius)).all(1)
                stack.append((child, inputs[in_halo], child_lo, child_hi))
    return plans


def convolution(module, x, unchunked):
    """Apply a stride-one odd-kernel submanifold conv with complete halos."""
    from .basic import SparseTensor
    from . import config

    policy = active_policy()
    if policy is None:
        return unchunked(module, x)
    if torch.is_grad_enabled():
        raise RuntimeError('Chunked convolution cannot run with gradients enabled')
    if config.CONV != 'flex_gemm':
        raise NotImplementedError('Chunked sparse convolution currently requires flex_gemm')
    if any(s != 1 for s in module.stride) or any(k % 2 != 1 for k in module.kernel_size):
        raise NotImplementedError('Chunking requires stride-one odd-kernel submanifold convolution')
    rows = policy.rows(x.feats, max(module.in_channels, module.out_channels), convolution=True)
    if len(x.feats) <= rows:
        return unchunked(module, x)
    radius = tuple((k // 2) * d for k, d in zip(module.kernel_size, module.dilation))
    key = (x.coords.data_ptr(), len(x.coords), rows, radius)
    # Plan indices are reused across repeated blocks at one resolution. Never
    # attach per-chunk convolution neighbor maps to the full sparse tensor.
    entry = policy.plans.get(key)
    if entry is None:
        plan = _spatial_plan(x.coords, rows, radius)
        policy.plans[key] = (x.coords, plan)
    else:
        plan = entry[1]
    output = x.feats.new_empty((len(x.feats), module.out_channels))
    shape = torch.Size([x.shape[0], x.feats.shape[1]])
    spatial_shape = x.spatial_shape
    for inputs, local, targets in plan:
        part = SparseTensor(x.feats[inputs], x.coords[inputs], shape=shape, scale=x._scale)
        part.register_spatial_cache('shape', spatial_shape)
        result = unchunked(module, part)
        output[targets] = result.feats[local]
        policy.record('convolution', len(inputs))
        policy.stats['halo_rows'] = policy.stats.get('halo_rows', 0) + len(inputs) - len(targets)
        del part, result
    return x.replace(output)
