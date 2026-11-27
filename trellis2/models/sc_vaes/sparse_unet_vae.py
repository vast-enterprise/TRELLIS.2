from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from ...modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module
from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from ...modules.sparse.inference_chunk import pointwise, inference_chunks, active_policy


def _norm_act(norm, feats):
    return pointwise(lambda f: F.silu(norm(f)), feats)


def _residual(h, x, skip):
    policy = active_policy()
    if policy is None:
        return h + skip(x)
    rows = policy.rows(h.feats, max(h.feats.shape[-1], x.feats.shape[-1]))
    if policy.threshold_bytes is not None and len(h.feats) <= rows:
        return h + skip(x)
    projected = skip(x) if policy.preserve_linear_shapes and isinstance(skip, sp.SparseLinear) else None
    for start in range(0, len(h.feats), rows):
        end = start + rows
        # All supported skip connections are per-row identity/linear/channel
        # expansion. No convolution or spatial cache is consumed here.
        value = (projected.feats[start:end] if projected is not None else
                 skip(sp.VarLenTensor(x.feats[start:end])).feats)
        h.feats[start:end].add_(value)
        policy.record('residual', len(value))
    return h


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: Literal['nearest', 'spatial2channel'] = 'nearest',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        self.use_checkpoint = use_checkpoint
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        if resample_mode == 'nearest':
            self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        elif resample_mode =='spatial2channel' and self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        if resample_mode == 'nearest':
            self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        elif resample_mode =='spatial2channel' and self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        self.updown = None
        if self.downsample:
            if resample_mode == 'nearest':
                self.updown = sp.SparseDownsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseSpatial2Channel(2)
        elif self.upsample:
            self.to_subdiv = sp.SparseLinear(channels, 8)
            if resample_mode == 'nearest':
                self.updown = sp.SparseUpsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseChannel2Spatial(2)

    def _updown(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > 0))
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        subdiv = None
        if self.upsample:
            subdiv = self.to_subdiv(x)
        h = x.replace(_norm_act(self.norm1, x.feats))
        if self.resample_mode == 'spatial2channel':
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == 'nearest':
            h = self.conv1(h)
        h = h.replace(_norm_act(self.norm2, h.feats))
        h = self.conv2(h)
        h = _residual(h, x, self.skip_connection)
        if self.upsample:
            return h, subdiv
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = sp.SparseDownsample(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(_norm_act(self.norm1, x.feats))
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        h = h.replace(_norm_act(self.norm2, h.feats))
        h = self.conv2(h)
        h = _residual(h, x, self.skip_connection)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        if self.pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseUpsample(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(_norm_act(self.norm1, x.feats))
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = self.conv1(h)
        h = h.replace(_norm_act(self.norm2, h.feats))
        h = self.conv2(h)
        h = _residual(h, x, self.skip_connection)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = sp.SparseSpatial2Channel(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(_norm_act(self.norm1, x.feats))
        h = self.conv1(h)
        h = self.updown(h)
        policy = active_policy()
        if policy is not None and policy.memory_efficient:
            skip = self.updown.reduced(x, self.out_channels)
            h = h.replace(pointwise(lambda f: F.silu(self.norm2(f)), h.feats, inplace=True))
            h = self.conv2(h)
            return _residual(h, skip, nn.Identity())
        x = self.updown(x)
        h = h.replace(_norm_act(self.norm2, h.feats))
        h = self.conv2(h)
        h = _residual(h, x, self.skip_connection)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseChannel2Spatial(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(_norm_act(self.norm1, x.feats))
        policy = active_policy()
        if policy is not None and policy.memory_efficient:
            from ...modules.sparse.inference_chunk import convolution_to_children
            mask = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
            h = convolution_to_children(self.conv1, h, self.updown, mask)
            skip = self.updown(x, mask)
            # Both branches use the same subdivision/order. Keep one global
            # coordinate buffer instead of two high-resolution copies.
            skip.coords = h.coords
            h = h.replace(pointwise(lambda f: F.silu(self.norm2(f)), h.feats, inplace=True))
            h = self.conv2(h)
            h = _residual(h, skip, self.skip_connection)
            return (h, subdiv) if self.pred_subdiv else h
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = h.replace(_norm_act(self.norm2, h.feats))
        h = self.conv2(h)
        h = _residual(h, x, self.skip_connection)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        else:
            return self._forward(x, subdiv)
        
    
class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = sp.SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            zero_module(nn.Linear(int(channels * mlp_ratio), channels)),
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        policy = active_policy()
        if policy is not None and policy.preserve_linear_shapes:
            # Keep GEMM shapes: changing row counts can change FP16/TF32 rounding.
            efficient = policy.memory_efficient
            value = pointwise(self.norm, h.feats, inplace=efficient)
            if efficient:
                meta = h.replace(h.feats.new_empty((len(h.feats), 0)))
                del h
            value = self.mlp[0](value)
            value = pointwise(self.mlp[1], value, inplace=efficient)
            h = (meta if efficient else h).replace(self.mlp[2](value))
            del value
        else:
            h = h.replace(pointwise(lambda f: self.mlp(self.norm(f)), h.feats,
                                    channels=self.mlp[0].out_features,
                                    inplace=bool(policy and policy.memory_efficient)))
        return _residual(h, x, nn.Identity())
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseUnetVaeEncoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])
        self.to_latent = sp.SparseLinear(model_channels[-1], 2 * latent_channels)
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        **block_args[i],
                    )
                )
                
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, sample_posterior=False, return_raw=False, *,
                chunk_size: Union[int, Literal['auto'], None] = None,
                chunk_memory_mb: Optional[float] = None,
                chunk_threshold_mb: float = 6144,
                max_memory_gb: Optional[float] = None, allow_host_offload: bool = False):
        """Encode with optional inference-time adaptive chunking.

        Chunking is applied to pointwise layers, stride-one convolutions and
        spatial-to-channel rearrangements. Global parent/child indices and
        complete per-convolution halos preserve the full-pass receptive field.
        Linear projections keep their original matrix shapes to avoid cuBLAS
        precision changes; full feature tensors still reside on the device.
        Use only with ``eval()`` and ``no_grad()``/``inference_mode()``.
        Compare deterministic latents with ``sample_posterior=False``; sampling
        requires the same RNG state. Floating-point kernels can round differently
        at different chunk sizes; bitwise equality is not guaranteed.
        ``max_memory_gb=23`` enables a 23 GiB deployment budget, reserving
        at least 1.5 GiB (or observed external CUDA usage plus 256 MiB).
        This path reuses GPU intermediates and fuses resampling writes. CPU
        staging is disabled unless ``allow_host_offload=True``. Set
        ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` before CUDA startup
        to reduce fragmentation. External CUDA allocations can still grow;
        the cap enforces PyTorch allocations, not other processes.
        """
        self.last_chunk_stats = {}
        if max_memory_gb is not None:
            from ...modules.sparse.bounded_vae import bounded_vae
            return bounded_vae(self, x, max_memory_gb, 'encode',
                               chunk_size=chunk_size, chunk_memory_mb=chunk_memory_mb,
                               chunk_threshold_mb=chunk_threshold_mb,
                               allow_host_offload=allow_host_offload,
                               sample_posterior=sample_posterior, return_raw=return_raw)
        if chunk_size is not None:
            if self.training:
                raise RuntimeError('Chunked encode requires model.eval()')
            automatic = chunk_size == 'auto'
            with inference_chunks(1_000_000 if automatic else chunk_size, chunk_memory_mb,
                                  threshold_mb=chunk_threshold_mb if automatic else None,
                                  preserve_linear_shapes=True) as policy:
                result = self._encode(x, sample_posterior, return_raw)
            self.last_chunk_stats = dict(policy.stats)
            return result
        if chunk_memory_mb is not None:
            raise ValueError('chunk_memory_mb requires chunk_size')
        return self._encode(x, sample_posterior, return_raw)

    def _encode(self, x, sample_posterior=False, return_raw=False):
        h = self.input_layer(x)
        h = h.type(self.dtype)
        pending = [h]
        del h
        return self._encode_features(pending, x.dtype, sample_posterior, return_raw)

    def _encode_features(self, pending, input_dtype, sample_posterior=False, return_raw=False):
        # Transfer ownership so the caller does not retain the first feature
        # grid (potentially many GiB) through all subsequent encoder blocks.
        h = pending.pop()
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
        h = h.type(input_dtype)
        h = h.replace(pointwise(lambda f: F.layer_norm(f, f.shape[-1:]), h.feats,
                               inplace=bool(active_policy() and active_policy().memory_efficient)))
        h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z
    
    
class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.low_vram = False
        
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)
        self.from_latent = sp.SparseLinear(latent_channels, model_channels[0])
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[up_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
                    
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
            
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None,
                return_subs: bool = False, *, chunk_size: Union[int, Literal['auto'], None] = None,
                chunk_memory_mb: Optional[float] = None,
                chunk_threshold_mb: float = 6144,
                max_memory_gb: Optional[float] = None, allow_host_offload: bool = False) -> sp.SparseTensor:
        """Decode with optional convolution halos and bounded workspaces.

        ``chunk_size='auto'`` counts actual voxels at every layer and skips
        chunking when its estimated peak is below ``chunk_threshold_mb`` (MiB).
        An integer instead caps rows, including halos, regardless of threshold.
        Both modes require eval() and no_grad()/inference_mode(). Full tensors
        remain resident; the estimate is a heuristic, not an OOM guarantee.
        ``max_memory_gb=23`` caps the allocator after reserving at least
        1.5 GiB for external CUDA allocations. Intermediates stay on GPU;
        host storage is available only with ``allow_host_offload=True``.
        Use ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` at startup to
        reduce allocator fragmentation with large full-shape GEMMs.
        """
        self.last_chunk_stats = {}
        if max_memory_gb is not None:
            from ...modules.sparse.bounded_vae import bounded_vae
            return bounded_vae(self, x, max_memory_gb, 'decode',
                               chunk_size=chunk_size, chunk_memory_mb=chunk_memory_mb,
                               chunk_threshold_mb=chunk_threshold_mb,
                               allow_host_offload=allow_host_offload,
                               guide_subs=guide_subs, return_subs=return_subs)
        if chunk_size is not None:
            if self.training:
                raise RuntimeError('Chunked decode requires model.eval()')
            automatic = chunk_size == 'auto'
            with inference_chunks(1_000_000 if automatic else chunk_size, chunk_memory_mb,
                                  threshold_mb=chunk_threshold_mb if automatic else None) as policy:
                result = self._decode(x, guide_subs, return_subs)
            self.last_chunk_stats = dict(policy.stats)
            return result
        if chunk_memory_mb is not None:
            raise ValueError('chunk_memory_mb requires chunk_size')
        return self._decode(x, guide_subs, return_subs)

    def _decode(self, x, guide_subs=None, return_subs=False):
        assert guide_subs is None or self.pred_subdiv == False, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
        policy = active_policy()
        if policy is not None and policy.memory_efficient:
            policy.plans.clear()
        h = h.type(x.dtype)
        h = h.replace(pointwise(lambda f: F.layer_norm(f, f.shape[-1:]), h.feats,
                               inplace=bool(active_policy() and active_policy().memory_efficient)))
        h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        else:
            if return_subs:
                return h, subs
            else:
                return h
    
    def upsample(self, x: sp.SparseTensor, upsample_times: int) -> torch.Tensor:
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    h, sub = block(h)
                else:
                    h = block(h)
