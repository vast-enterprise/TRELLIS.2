from typing import *
import torch
import torch.nn as nn
from .. import SparseTensor


class SparseSpatial2Channel(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as rearranging its features from spatial to channel.
    """
    def __init__(self, factor: int = 2):
        super(SparseSpatial2Channel, self).__init__()
        self.factor = factor

    def forward(self, x: SparseTensor) -> SparseTensor:
        DIM = x.coords.shape[-1] - 1
        cache = x.get_spatial_cache(f'spatial2channel_{self.factor}')
        if cache is None:
            coord = list(x.coords.unbind(dim=-1))
            for i in range(DIM):
                coord[i+1] = coord[i+1] // self.factor
            subidx = x.coords[:, 1:] % self.factor
            subidx = sum([subidx[..., i] * self.factor ** i for i in range(DIM)])

            MAX = [(s + self.factor - 1) // self.factor for s in x.spatial_shape]
            OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
            code = sum([c * o for c, o in zip(coord, OFFSET)])
            code, idx = code.unique(return_inverse=True)

            new_coords = torch.stack(
                [code // OFFSET[0]] +
                [(code // OFFSET[i+1]) % MAX[i] for i in range(DIM)],
                dim=-1
            )
        else:
            new_coords, idx, subidx = cache
            
        new_feats = torch.zeros(new_coords.shape[0] * self.factor ** DIM, x.feats.shape[1], device=x.feats.device, dtype=x.feats.dtype)
        from ..inference_chunk import active_policy
        policy = active_policy()
        if (cache is None and policy is not None and policy.memory_efficient
                and len(new_coords) * self.factor ** DIM < 2**31):
            idx = idx.int()
            subidx = subidx.to(torch.uint8 if self.factor ** DIM <= 256 else torch.int32)
        if policy is not None:
            if torch.is_grad_enabled():
                raise RuntimeError('Chunked spatial rearrangement cannot run with gradients enabled')
            rows = policy.rows(x.feats, x.feats.shape[1] * self.factor ** DIM)
        else:
            rows = max(1, len(x.feats))
        if len(x.feats) <= rows:
            new_feats[idx * self.factor ** DIM + subidx] = x.feats
        else:
            # Keep the global parent/child mapping and write disjoint slots.
            # Children of the same parent may cross a chunk boundary: no sum
            # or averaging is involved, so splitting them is exactly equivalent.
            for start in range(0, len(x.feats), rows):
                end = min(start + rows, len(x.feats))
                target = idx[start:end] * self.factor ** DIM + subidx[start:end]
                new_feats[target] = x.feats[start:end]
                policy.record('spatial2channel', end - start)

        out = SparseTensor(new_feats.reshape(new_coords.shape[0], -1), new_coords, None if x._shape is None else torch.Size([x._shape[0], x._shape[1] * self.factor ** DIM]))
        out._scale = tuple([s * self.factor for s in x._scale])
        out._spatial_cache = x._spatial_cache
        
        if cache is None:
            x.register_spatial_cache(f'spatial2channel_{self.factor}', (new_coords, idx, subidx))
            out.register_spatial_cache(f'channel2spatial_{self.factor}', (x.coords, idx, subidx))
            out.register_spatial_cache(f'shape', torch.Size(MAX))
            if self.training:
                subdivision = torch.zeros((new_coords.shape[0], self.factor ** DIM), device=x.device, dtype=torch.bool)
                subdivision[idx, subidx] = True
                out.register_spatial_cache(f'subdivision', subdivision)
                
        return out

    def reduced(self, x: SparseTensor, out_channels: int) -> SparseTensor:
        """S2C followed by the residual channel mean without padded features.

        Each mean group must fit within one child's channels. Missing children
        remain zero; no scatter reduction or change of addition order is used.
        The main branch must have already populated the global mapping.
        """
        from ..inference_chunk import active_policy
        slots = self.factor ** (x.coords.shape[-1] - 1)
        channels = x.feats.shape[1]
        if out_channels < slots or out_channels % slots or channels % (out_channels // slots):
            raise ValueError('Reduced S2C requires whole channel groups per child')
        coords, idx, subidx = x.get_spatial_cache(f'spatial2channel_{self.factor}')
        child_channels = out_channels // slots
        output = x.feats.new_zeros((len(coords) * slots, child_channels))
        policy = active_policy()
        rows = policy.rows(x.feats, channels)
        for start in range(0, len(x.feats), rows):
            end = start + rows
            value = x.feats[start:end].reshape(-1, child_channels, channels // child_channels).mean(-1)
            output[idx[start:end] * slots + subidx[start:end]] = value
            policy.record('reduced_spatial2channel', len(value))
        out = x.replace(output.view(len(coords), out_channels), coords)
        out._scale = tuple(s * self.factor for s in x._scale)
        return out


class SparseChannel2Spatial(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as rearranging its features from channel to spatial.
    """
    def __init__(self, factor: int = 2):
        super(SparseChannel2Spatial, self).__init__()
        self.factor = factor

    def indices(self, x, subdivision):
        cache = x.get_spatial_cache(f'channel2spatial_{self.factor}')
        if cache is not None:
            return cache
        if subdivision is None:
            raise ValueError('Upsampling requires subdivision or an encoder cache')
        from ..inference_chunk import active_policy
        idx, subidx = torch.nonzero(subdivision.feats, as_tuple=True)
        slots = self.factor ** (x.coords.shape[-1] - 1)
        if len(x.coords) * slots < 2**31:
            idx = idx.int()
            subidx = subidx.to(torch.uint8 if slots <= 256 else torch.int32)
        coords = x.coords.new_empty((len(idx), x.coords.shape[-1]))
        rows = active_policy().max_points
        for start in range(0, len(idx), rows):
            end = start + rows
            value = x.coords[idx[start:end]].clone()
            value[:, 1:] *= self.factor
            for axis in range(x.coords.shape[-1] - 1):
                value[:, axis + 1] += subidx[start:end] // self.factor**axis % self.factor
            coords[start:end] = value
        return coords, idx, subidx

    def forward(self, x: SparseTensor, subdivision: Optional[SparseTensor] = None) -> SparseTensor:
        DIM = x.coords.shape[-1] - 1

        cache = x.get_spatial_cache(f'channel2spatial_{self.factor}')
        if cache is None:
            if subdivision is None:
                raise ValueError('Cache not found. Provide subdivision tensor or pair SparseChannel2Spatial with SparseSpatial2Channel.')
            else:
                sub = subdivision.feats         # [N, self.factor ** DIM]
                N_leaf = sub.sum(dim=-1)        # [N]
                subidx = sub.nonzero()[:, -1]
                new_coords = x.coords.clone().detach()
                new_coords[:, 1:] *= self.factor
                new_coords = torch.repeat_interleave(new_coords, N_leaf, dim=0, output_size=subidx.shape[0])
                for i in range(DIM):
                    new_coords[:, i+1] += subidx // self.factor ** i % self.factor
                idx = torch.repeat_interleave(torch.arange(x.coords.shape[0], device=x.device), N_leaf, dim=0, output_size=subidx.shape[0])
        else:
            new_coords, idx, subidx = cache

        x_feats = x.feats.reshape(x.feats.shape[0] * self.factor ** DIM, -1)
        new_feats = x_feats[idx * self.factor ** DIM + subidx]
        out = SparseTensor(new_feats, new_coords, None if x._shape is None else torch.Size([x._shape[0], x._shape[1] // self.factor ** DIM]))
        out._scale = tuple([s / self.factor for s in x._scale])
        if cache is not None:           # only keep cache when subdiv following it
            out._spatial_cache = x._spatial_cache
        return out
