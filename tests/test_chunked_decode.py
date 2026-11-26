"""Inference chunk halos, subdivision alignment and autograd isolation."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'o-voxel'))
from trellis2.modules import sparse as sp
from trellis2.modules.sparse.inference_chunk import inference_chunks, active_policy
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def precise_small_fixture_convolution():
    # These tiny FP32 fixtures test halo correctness, not flex_gemm autotuning.
    # A fixed explicit algorithm avoids shape-dependent TF32/split-K roundoff.
    from trellis2.modules.sparse.conv import config
    old = config.FLEX_GEMM_ALGO
    config.FLEX_GEMM_ALGO = 'explicit_gemm'
    try:
        yield
    finally:
        config.FLEX_GEMM_ALGO = old


def grid(n=9, channels=8, batches=1):
    xyz = torch.cartesian_prod(*[torch.arange(n)]*3).int()
    coords = torch.cat([torch.cat((torch.full((len(xyz), 1), b), xyz), 1) for b in range(batches)]).cuda().int()
    return sp.SparseTensor(torch.randn(len(coords), channels, device='cuda'), coords)


@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_convolution_overlap_and_batch(dilation, context):
    torch.manual_seed(42)
    x = grid(batches=2)
    conv = sp.SparseConv3d(8, 8, 3, dilation=dilation).cuda().eval()
    with context():
        ref = conv(x)
        fresh = sp.SparseTensor(x.feats, x.coords)
        with inference_chunks(256) as policy:
            out = conv(fresh)
        assert torch.equal(out.coords, ref.coords)
        torch.testing.assert_close(out.feats, ref.feats, atol=1e-5, rtol=1e-5)
        assert policy.stats['halo_rows'] > 0
        assert policy.stats['convolution_chunks'] > 1
        # Full convolution neighbor caches must not leak out of local chunks.
        assert not any(str(k).startswith('SubMConv') for k in fresh.get_spatial_cache())


def decoder(pred=True):
    model = SparseUnetVaeDecoder(out_channels=6, model_channels=[16, 8], latent_channels=8,
        num_blocks=[2, 1], block_type=['SparseConvNeXtBlock3d']*2,
        up_block_type=['SparseResBlockC2S3d'], block_args=[{}, {}], pred_subdiv=pred).cuda().eval()
    # Use nonzero residual outputs; constructor zero initialization would hide
    # missing halos from all but the upsampling layer.
    for name, p in model.named_parameters():
        if 'mlp.2.weight' in name or 'conv2.weight' in name:
            torch.nn.init.normal_(p, std=.03)
    if pred:
        model.blocks[0][-1].to_subdiv.weight.data.zero_()
        model.blocks[0][-1].to_subdiv.bias.data.fill_(2)
    return model


@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_multilayer_decode_and_texture_guidance(context):
    torch.manual_seed(17)
    shape, texture = decoder(), decoder(False)
    x = grid(n=5)
    with context():
        ref, ref_subs = shape(x, return_subs=True)
        out, subs = shape(sp.SparseTensor(x.feats, x.coords), return_subs=True, chunk_size=256)
        assert torch.equal(out.coords, ref.coords)
        torch.testing.assert_close(out.feats, ref.feats, atol=2e-5, rtol=2e-5)
        for a, b in zip(subs, ref_subs):
            assert torch.equal(a.coords, b.coords)
            torch.testing.assert_close(a.feats, b.feats)
        tex = texture(sp.SparseTensor(x.feats, x.coords), guide_subs=subs, chunk_size=256)
        tex_ref = texture(sp.SparseTensor(x.feats, x.coords), guide_subs=ref_subs)
        assert torch.equal(tex.coords, out.coords)
        torch.testing.assert_close(tex.feats, tex_ref.feats, atol=2e-5, rtol=2e-5)


def test_requires_eval_and_disabled_grad_and_restores_context():
    model = decoder()
    x = grid(n=2)
    with pytest.raises(RuntimeError, match='no_grad'):
        model(x, chunk_size=256)
    with torch.no_grad():
        model.train()
        with pytest.raises(RuntimeError, match='eval'):
            model(x, chunk_size=256)
        with pytest.raises(ValueError):
            with inference_chunks(0): pass
        with pytest.raises(ValueError):
            with inference_chunks(100):
                raise ValueError('injected failure')
        assert active_policy() is None
    # The ordinary training path still builds gradients.
    conv = sp.SparseConv3d(8, 8, 3).cuda()
    conv(x).feats.sum().backward()
    assert conv.weight.grad is not None


def test_budget_empty_tensor_and_cache_lifetime():
    from trellis2.modules.sparse.inference_chunk import pointwise
    x = grid(n=5)
    conv = sp.SparseConv3d(8, 8, 3).cuda().eval()
    with torch.inference_mode():
        with inference_chunks(1_000_000, memory_budget_mb=.125) as policy:
            assert policy.rows(x.feats, 1024) < policy.rows(x.feats, 8)
            empty = pointwise(lambda f: f + 1, x.feats[:0])
            assert empty.shape == x.feats[:0].shape
        with inference_chunks(100) as policy:
            out = conv(x)
            assert policy.plans and policy.stats['convolution_chunks'] > 1
        assert not any(str(k).startswith('inference_chunk') for k in x.get_spatial_cache())
        assert not any(str(k).startswith('inference_chunk') for k in out.get_spatial_cache())
        with inference_chunks(1):
            with pytest.raises(ValueError, match='halo'):
                conv(x)


def test_cached_texture_upsampling_and_public_geometry_decoder():
    from trellis2.modules.sparse.spatial.spatial2channel import SparseSpatial2Channel
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
    torch.manual_seed(31)
    x = grid(n=8, channels=1)
    with torch.no_grad():
        z = SparseSpatial2Channel(2)(x)
        texture = decoder(False)
        ref = texture(z)
        out = texture(z, chunk_size=256)
        assert torch.equal(out.coords, x.coords)
        torch.testing.assert_close(ref.feats, out.feats, atol=2e-5, rtol=2e-5)
        geometry = FlexiDualGridVaeDecoder(resolution=16, model_channels=[16, 8],
            latent_channels=8, num_blocks=[1, 0], block_type=['SparseConvNeXtBlock3d']*2,
            up_block_type=['SparseResBlockC2S3d'], block_args=[{}, {}]).cuda().eval()
        meshes, subs = geometry(sp.SparseTensor(z.feats, z.coords), chunk_size=256, return_subs=True)
        assert len(meshes) == len(subs) == 1
        assert torch.isfinite(meshes[0].vertices).all()


def test_auto_threshold_uses_actual_layer_rows_and_keeps_small_ops_whole():
    from trellis2.modules.sparse.inference_chunk import pointwise
    torch.manual_seed(71)
    model = decoder()
    x = grid(n=5)
    with torch.inference_mode():
        reference = model(sp.SparseTensor(x.feats, x.coords))
        whole = model(sp.SparseTensor(x.feats, x.coords), chunk_size='auto', chunk_threshold_mb=100_000)
        torch.testing.assert_close(reference.feats, whole.feats, atol=2e-5, rtol=2e-5)
        assert model.last_chunk_stats['auto_max_voxels'] == 1000
        assert model.last_chunk_stats['auto_skipped_ops'] > 0
        assert model.last_chunk_stats.get('convolution_chunks', 0) == 0
        # Exercise both decisions in the same context without a large fixture.
        with inference_chunks(256, threshold_mb=100_000) as policy:
            pointwise(lambda f: f * 2, x.feats)
            assert policy.stats['auto_skipped_ops'] == 1
            policy.threshold_bytes = 1
            f = x.feats.repeat(4, 1)
            out = pointwise(lambda f: f * 2, f)
            assert policy.stats['pointwise_chunks'] == 2
            torch.testing.assert_close(out, f * 2)
        for threshold in [0, -1, float('nan')]:
            with pytest.raises(ValueError, match='threshold'):
                model(x, chunk_size='auto', chunk_threshold_mb=threshold)
        model(x)
        assert model.last_chunk_stats == {}
    with pytest.raises(RuntimeError, match='no_grad'):
        model(x, chunk_size='auto')
