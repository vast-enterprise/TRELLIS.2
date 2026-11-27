"""Inference-only VAE execution with host feature storage and a CUDA budget.

Features are spilled losslessly to CPU between operations. Convolutions compute
one layer at a time with complete halos; linear projections retain their full
matrix shapes. This trades PCIe traffic and host RAM for bounded GPU residency.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import gc
import math
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import SparseTensor, VarLenTensor
from .inference_chunk import ChunkPolicy, _spatial_plan, inference_chunks

_LOCK = threading.RLock()
_BUDGET = ContextVar('trellis_vae_cuda_budget', default=None)


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    return value


def _transfer(x, device):
    out = SparseTensor(x.feats.to(device), x.coords.to(device), scale=x._scale)
    # Neighbor maps are implementation workspaces, not encoder/decoder guides.
    out._spatial_cache = _move({scale: {k: v for k, v in cache.items()
        if not k.startswith('SubMConv')} for scale, cache in x._spatial_cache.items()}, device)
    return out


@contextmanager
def cuda_budget(device, max_memory_gb):
    """Cap the allocator after reserving observed external CUDA memory.

    The allocator limit is process-wide, so bounded forwards are serialized.
    Existing models and caller-owned tensors count towards the same limit.
    CUDA allocations made outside PyTorch cannot be enforced by this allocator.
    """
    if not math.isfinite(max_memory_gb) or max_memory_gb <= 1.5:
        raise ValueError('max_memory_gb must be finite and greater than 1.5 GiB')
    if torch.is_grad_enabled():
        raise RuntimeError('Bounded VAE inference requires no_grad or inference_mode')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Bounded VAE inference requires a CUDA model')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    with _LOCK, torch.cuda.device(device):
        parent = _BUDGET.get()
        if parent is not None and parent[:2] == (device.index, max_memory_gb):
            # FDG's public forward also wraps input preparation/mesh extraction.
            # Keep the same allowance through nested base-model forwards.
            yield parent[2]
            return
        previous = torch.cuda.get_per_process_memory_fraction(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.empty_cache()
        try:
            used = torch.cuda.device_memory_used(device)
        except (ImportError, ModuleNotFoundError, RuntimeError):
            free, _ = torch.cuda.mem_get_info(device)
            used = total - free
        external = max(0, used - torch.cuda.memory_reserved(device))
        reserve = max(int(1.5 * 1024**3), external + 256 * 1024**2)
        limit = min(int(max_memory_gb * 1024**3) - reserve, int(total * previous))
        if torch.cuda.memory_allocated(device) >= limit:
            raise torch.cuda.OutOfMemoryError('Caller-owned tensors and models already exceed the VAE CUDA budget')
        torch.cuda.set_per_process_memory_fraction(limit / total, device)
        token = _BUDGET.set((device.index, max_memory_gb, limit))
        try:
            yield limit
        finally:
            _BUDGET.reset(token)
            torch.cuda.set_per_process_memory_fraction(previous, device)


class HostVae:
    def __init__(self, model, limit, max_points=1_000_000, memory_mb=None, threshold_mb=6144):
        self.model = model
        self.device = model.device
        self.limit = limit
        self.threshold_mb = threshold_mb
        self.policy = ChunkPolicy(max_points=max_points,
            memory_budget_bytes=int((memory_mb or 8192) * 1024**2), preserve_linear_shapes=True)
        self.policy.stats.update(host_offload=True, preserve_linear_shapes=True,
                                 allocator_limit_gib=limit / 1024**3)
        self.plans = {}
        self.staging = {}
        self.resident_blocks = True

    def rows(self, channels, element_size=4):
        available = self.limit - torch.cuda.memory_allocated(self.device)
        budget = min(self.policy.memory_budget_bytes, max(1, int(available * .5)))
        rows = min(self.policy.max_points, max(1, budget // (max(channels, 1) * max(element_size, 4) * 8)))
        return max(1, rows // 256 * 256) if rows >= 256 else rows

    def readback(self, value, destination):
        key = value.dtype
        count = value.numel()
        slab = self.staging.get(key)
        if slab is None or slab.numel() < count:
            slab = torch.empty(count, dtype=value.dtype, pin_memory=True)
            self.staging[key] = slab
        view = slab[:count].view(value.shape)
        view.copy_(value, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        destination.copy_(view)

    def pointwise(self, fn, feats, channels=None):
        rows = self.rows(channels or feats.shape[-1], feats.element_size())
        if len(feats) == 0:
            return fn(feats.to(self.device)).cpu()
        output = None
        for start in range(0, len(feats), rows):
            value = fn(feats[start:start + rows].to(self.device))
            if output is None:
                output = torch.empty((len(feats), *value.shape[1:]), dtype=value.dtype)
            self.readback(value, output[start:start + len(value)])
            self.policy.record('host_pointwise', len(value))
            del value
        return output

    def linear(self, layer, feats):
        # Full GEMM shapes are essential: cuBLAS can change TF32/FP16 precision
        # with row count. Only its input and output are resident for this op.
        # Large GEMMs need contiguous allocations. Cached blocks from the
        # previous layer can otherwise consume the allocator cap as fragments.
        required = len(feats) * (layer.in_features + layer.out_features) * feats.element_size()
        if torch.cuda.memory_reserved(self.device) + required > self.limit:
            torch.cuda.empty_cache()
        inputs = feats.to(self.device)
        result = F.linear(inputs, layer.weight, layer.bias)
        output = result.cpu()
        self.policy.record('host_full_linear', len(feats))
        del inputs, result
        return output

    def conv(self, layer, x):
        from .conv.conv_flex_gemm import sparse_conv3d_forward
        from . import config
        if config.CONV != 'flex_gemm':
            raise NotImplementedError('Bounded VAE convolutions require flex_gemm')
        if any(s != 1 for s in layer.stride) or any(k % 2 != 1 for k in layer.kernel_size):
            raise NotImplementedError('Bounded VAE requires stride-one odd-kernel submanifold convolutions')
        rows = self.rows(max(layer.in_channels, layer.out_channels), x.feats.element_size())
        radius = tuple(k // 2 * d for k, d in zip(layer.kernel_size, layer.dilation))
        key = (x.coords.data_ptr(), len(x.coords), rows, radius)
        cached = self.plans.get(key)
        if cached is None:
            if len(x.coords) <= rows:
                indices = torch.arange(len(x.coords))
                plan = [(indices, indices, indices)]
            else:
                coords = x.coords.to(self.device)
                plan = [tuple(t.cpu() for t in entry) for entry in _spatial_plan(coords, rows, radius)]
                del coords
            self.plans[key] = (x.coords, plan)
        else:
            plan = cached[1]
        output = torch.empty((len(x.feats), layer.out_channels), dtype=x.feats.dtype)
        shape = x.shape
        spatial_shape = x.spatial_shape
        for inputs, local, targets in plan:
            part = SparseTensor(x.feats[inputs].to(self.device), x.coords[inputs].to(self.device),
                                shape=shape, scale=x._scale)
            part.register_spatial_cache('shape', spatial_shape)
            value = sparse_conv3d_forward(layer, part)
            output[targets] = value.feats[local.to(self.device)].cpu()
            self.policy.record('host_convolution', len(inputs))
            self.policy.stats['halo_rows'] = self.policy.stats.get('halo_rows', 0) + len(inputs) - len(targets)
            del part, value
        return x.replace(output)

    def down(self, layer, x):
        if layer.__class__.__name__ != 'SparseSpatial2Channel':
            raise NotImplementedError('Bounded encoder currently requires spatial-to-channel downsampling')
        key = f'spatial2channel_{layer.factor}'
        cache = x.get_spatial_cache(key)
        if cache is None:
            # Compute exactly the original global coordinate ordering with the
            # CUDA implementation, but with zero feature channels.
            meta = SparseTensor(torch.empty((len(x.coords), 0), device=self.device),
                                x.coords.to(self.device), scale=x._scale)
            out = layer(meta)
            mapping = _move(out._spatial_cache, 'cpu')
            for scale, entries in mapping.items():
                x._spatial_cache.setdefault(scale, {}).update(entries)
            cache = x.get_spatial_cache(key)
            del meta, out
        coords, idx, subidx = cache
        slots = layer.factor ** (x.coords.shape[-1] - 1)
        features = torch.zeros((len(coords) * slots, x.feats.shape[-1]), dtype=x.feats.dtype)
        rows = self.policy.max_points
        for start in range(0, len(idx), rows):
            end = min(start + rows, len(idx))
            features[idx[start:end] * slots + subidx[start:end]] = x.feats[start:end]
        out = x.replace(features.view(len(coords), -1), coords)
        out._scale = tuple(s * layer.factor for s in x._scale)
        return out

    def up(self, layer, x, subdivision):
        if layer.__class__.__name__ != 'SparseChannel2Spatial':
            raise NotImplementedError('Bounded decoder currently requires channel-to-spatial upsampling')
        cache = x.get_spatial_cache(f'channel2spatial_{layer.factor}')
        if cache is None:
            if subdivision is None:
                raise ValueError('Upsampling needs subdivision guidance or an encoder cache')
            idx, subidx = torch.nonzero(subdivision.feats, as_tuple=True)
            coords = x.coords[idx].clone()
            coords[:, 1:] *= layer.factor
            for axis in range(coords.shape[-1] - 1):
                coords[:, axis + 1] += subidx // layer.factor**axis % layer.factor
        else:
            coords, idx, subidx = cache
        slots = layer.factor ** (x.coords.shape[-1] - 1)
        values = x.feats.reshape(len(x.coords) * slots, -1)
        output = torch.empty((len(coords), values.shape[-1]), dtype=values.dtype)
        for start in range(0, len(coords), self.policy.max_points):
            end = start + self.policy.max_points
            output[start:end] = values[idx[start:end] * slots + subidx[start:end]]
        out = x.replace(output, coords)
        out._scale = tuple(s / layer.factor for s in x._scale)
        if cache is None:
            out._spatial_cache = {}
        return out

    def norm_act(self, norm, x):
        return x.replace(self.pointwise(lambda f: F.silu(norm(f)), x.feats))

    def residual(self, h, x, skip):
        projected = (x.replace(self.linear(skip, x.feats)) if isinstance(skip, nn.Linear) else x)
        rows = self.rows(max(h.feats.shape[-1], x.feats.shape[-1]))
        for start in range(0, len(h.feats), rows):
            end = start + rows
            a = h.feats[start:end].to(self.device)
            b = projected.feats[start:end].to(self.device)
            if not isinstance(skip, nn.Linear):
                b = skip(VarLenTensor(b)).feats
            a.add_(b)
            self.readback(a, h.feats[start:end])
            del a, b
        return h

    def block(self, block, x, subdiv=None):
        # Coarser scales fit comfortably on the GPU. Execute a whole block
        # there instead of crossing PCIe at every norm/activation/convolution.
        width = x.feats.shape[-1]
        if hasattr(block, 'conv1'):
            width = max(width, block.conv1.out_channels * 2)
        estimate = len(x.feats) * width * max(x.feats.element_size(), 4) * 12
        if self.resident_blocks and estimate + torch.cuda.memory_allocated(self.device) < self.limit * .75:
            try:
                gpu = SparseTensor(x.feats.to(self.device), x.coords.to(self.device), scale=x._scale)
                # A ConvNeXt block does not consume mappings from earlier
                # scales. Avoid transferring those multi-GB guide caches at
                # every block; preserve them on the host and merge new maps.
                needed = ['shape']
                if hasattr(block, 'updown'):
                    factor = block.updown.factor
                    needed.append(('spatial2channel_' if block.__class__.__name__ == 'SparseResBlockS2C3d'
                                   else 'channel2spatial_') + str(factor))
                for key in needed:
                    cached = x.get_spatial_cache(key)
                    if cached is not None:
                        gpu.register_spatial_cache(key, _move(cached, self.device))
                with inference_chunks(self.policy.max_points,
                                      self.policy.memory_budget_bytes / 1024**2,
                                      threshold_mb=self.threshold_mb, preserve_linear_shapes=True):
                    if block.__class__.__name__ == 'SparseResBlockC2S3d':
                        value = block(gpu, subdiv=_transfer(subdiv, self.device) if subdiv is not None else None)
                    else:
                        value = block(gpu)
                if isinstance(value, tuple):
                    output = tuple(_transfer(v, 'cpu') for v in value)
                else:
                    output = _transfer(value, 'cpu')
                for item in output if isinstance(output, tuple) else (output,):
                    merged = {k: dict(v) for k, v in x._spatial_cache.items()}
                    for scale, cache in item._spatial_cache.items():
                        merged.setdefault(scale, {}).update(cache)
                    item._spatial_cache = merged
                self.policy.record('resident_block', len(x.feats))
                return output
            except torch.cuda.OutOfMemoryError:
                pass
            # Any tensors assigned before OOM must be released before the host
            # path. The exception traceback has already left scope here.
            gpu = value = None
            gc.collect()
            torch.cuda.empty_cache()
        kind = block.__class__.__name__
        if kind == 'SparseConvNeXtBlock3d':
            h = self.conv(block.conv, x)
            value = self.pointwise(block.norm, h.feats)
            value = self.linear(block.mlp[0], value)
            value = self.pointwise(block.mlp[1], value)
            h = h.replace(self.linear(block.mlp[2], value))
            del value
            return self.residual(h, x, nn.Identity())
        if kind not in ('SparseResBlockS2C3d', 'SparseResBlockC2S3d'):
            raise NotImplementedError(f'Bounded VAE does not support {kind}')
        up = kind == 'SparseResBlockC2S3d'
        predicted = up and block.pred_subdiv
        if predicted:
            subdiv = x.replace(self.linear(block.to_subdiv, x.feats))
        h = self.norm_act(block.norm1, x)
        h = self.conv(block.conv1, h)
        if up:
            mask = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
            h, x = self.up(block.updown, h, mask), self.up(block.updown, x, mask)
        else:
            h, x = self.down(block.updown, h), self.down(block.updown, x)
        h = self.conv(block.conv2, self.norm_act(block.norm2, h))
        h = self.residual(h, x, block.skip_connection)
        return (h, subdiv) if predicted else h

    def encode(self, x, sample_posterior=False, return_raw=False):
        x = _transfer(x, 'cpu')
        h = x.replace(self.linear(self.model.input_layer, x.feats))
        h = h.replace(self.pointwise(lambda f: f.to(self.model.dtype), h.feats))
        for stage in self.model.blocks:
            for block in stage:
                h = self.block(block, h)
        h = h.replace(self.pointwise(lambda f: F.layer_norm(f.to(x.dtype), f.shape[-1:]), h.feats))
        h = h.replace(self.linear(self.model.to_latent, h.feats))
        h = _transfer(h, self.device)
        mean, logvar = h.feats.chunk(2, dim=-1)
        value = mean + torch.exp(.5 * logvar) * torch.randn_like(mean) if sample_posterior else mean
        z = h.replace(value)
        return (z, mean, logvar) if return_raw else z

    def decode(self, x, guide_subs=None, return_subs=False):
        if guide_subs is not None and self.model.pred_subdiv:
            raise ValueError('Predicted-subdivision decoder cannot accept guide_subs')
        if return_subs and not self.model.pred_subdiv:
            raise ValueError('return_subs requires predicted subdivisions')
        x = _transfer(x, 'cpu')
        guides = [_transfer(s, 'cpu') for s in guide_subs] if guide_subs is not None else None
        h = x.replace(self.linear(self.model.from_latent, x.feats))
        h = h.replace(self.pointwise(lambda f: f.to(self.model.dtype), h.feats))
        subs = []
        for i, stage in enumerate(self.model.blocks):
            for j, block in enumerate(stage):
                if i < len(self.model.blocks) - 1 and j == len(stage) - 1:
                    if self.model.pred_subdiv:
                        h, sub = self.block(block, h)
                        subs.append(sub)
                    else:
                        h = self.block(block, h, guides[i] if guides is not None else None)
                else:
                    h = self.block(block, h)
        h = h.replace(self.pointwise(lambda f: F.layer_norm(f.to(x.dtype), f.shape[-1:]), h.feats))
        h = h.replace(self.linear(self.model.output_layer, h.feats))
        h = _transfer(h, self.device)
        return (h, [_transfer(s, self.device) for s in subs]) if return_subs else h


def bounded_vae(model, x, max_memory_gb, operation, *, chunk_size='auto',
                chunk_memory_mb=None, chunk_threshold_mb=6144, **kwargs):
    """Budgeted VAE entry point; full features live on CPU between CUDA ops."""
    if model.training:
        raise RuntimeError('Bounded VAE inference requires model.eval()')
    points = 1_000_000 if chunk_size in (None, 'auto') else chunk_size
    if not isinstance(points, int) or isinstance(points, bool) or points <= 0:
        raise ValueError('chunk_size must be auto or a positive integer')
    if chunk_memory_mb is not None and (not math.isfinite(chunk_memory_mb) or chunk_memory_mb <= 0):
        raise ValueError('chunk_memory_mb must be finite and positive')
    if not math.isfinite(chunk_threshold_mb) or chunk_threshold_mb <= 0:
        raise ValueError('chunk_threshold_mb must be finite and positive')
    threshold = chunk_threshold_mb if chunk_size in (None, 'auto') else None
    with cuda_budget(model.device, max_memory_gb) as limit:
        # Leave the current GPU path in place for inputs that fit. For large
        # encoders the expanded first stage already predicts the need to spill.
        estimate = len(x.feats) * model.model_channels[0] * 4 * 4
        retry = 0
        if x.device.type == 'cuda' and (operation == 'decode' or
                torch.cuda.memory_allocated(model.device) + estimate < limit):
            fresh = policy = None
            try:
                fresh = SparseTensor(x.feats, x.coords, scale=x._scale)
                fresh._spatial_cache = {k: dict(v) for k, v in x._spatial_cache.items()}
                with inference_chunks(points, chunk_memory_mb, threshold_mb=threshold,
                                      preserve_linear_shapes=True) as policy:
                    result = getattr(model, '_' + operation)(fresh, **kwargs)
                model.last_chunk_stats = dict(policy.stats, host_offload=False,
                    max_memory_gb=max_memory_gb, allocator_limit_gib=limit / 1024**3)
                return result
            except torch.cuda.OutOfMemoryError:
                # Exit the except block before retrying: its traceback holds
                # references to intermediates from the failed forward.
                retry = 1
            del fresh, policy
            gc.collect()
            torch.cuda.empty_cache()
        executor = HostVae(model, limit, points, chunk_memory_mb, threshold)
        result = getattr(executor, operation)(x, **kwargs)
        model.last_chunk_stats = dict(executor.policy.stats, max_memory_gb=max_memory_gb,
                                      gpu_oom_fallbacks=retry)
        return result
