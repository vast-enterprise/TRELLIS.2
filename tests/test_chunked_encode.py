"""Encoder halo equivalence, spatial reduction mapping and posterior RNG order."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'o-voxel'))
from trellis2.modules import sparse as sp
from trellis2.modules.sparse.inference_chunk import active_policy, inference_chunks
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder
from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def precise_convolution():
    import flex_gemm  # Import first: flex_gemm enables TF32 during initialization.
    from trellis2.modules.sparse.conv import config
    old = config.FLEX_GEMM_ALGO
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    config.FLEX_GEMM_ALGO = 'explicit_gemm'
    try:
        yield
    finally:
        config.FLEX_GEMM_ALGO = old
        torch.set_float32_matmul_precision(precision)


def inputs():
    # Irregular, permuted coordinates, batch separation and partial parents
    # exercise boundaries more thoroughly than a filled cube in XYZ order.
    torch.manual_seed(84)
    xyz = torch.cartesian_prod(*[torch.arange(11)] * 3).int()
    xyz = xyz[torch.rand(len(xyz)) > .3]
    xyz = xyz[torch.randperm(len(xyz))]
    coords = torch.cat([torch.cat((torch.full((len(xyz), 1), b), xyz), 1)
                        for b in range(2)]).int().cuda()
    feats = torch.randn(len(coords), 6, device='cuda')
    return sp.SparseTensor(feats, coords)


def encoder(geometry=False, down='SparseResBlockS2C3d'):
    kwargs = dict(model_channels=[8, 16, 32], latent_channels=4,
                  num_blocks=[1, 1, 1], block_type=['SparseConvNeXtBlock3d'] * 3,
                  down_block_type=[down] * 2, block_args=[{}, {}, {}])
    model = (FlexiDualGridVaeEncoder(**kwargs) if geometry else
             SparseUnetVaeEncoder(in_channels=6, **kwargs)).cuda().eval()
    for name, p in model.named_parameters():
        if 'mlp.2.weight' in name or 'conv2.weight' in name:
            torch.nn.init.normal_(p, std=.03)
    return model


def run(model, x, geometry=False, **kwargs):
    fresh = sp.SparseTensor(x.feats, x.coords)
    if geometry:
        return model(fresh.replace(x.feats[:, :3]), fresh.replace(x.feats[:, 3:] > 0), **kwargs)
    return model(fresh, **kwargs)


@pytest.mark.parametrize('geometry', [False, True])
@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize('down', ['SparseResBlockS2C3d', 'SparseResBlockDownsample3d'])
def test_encoder_posterior_and_multiscale_halos(geometry, context, down):
    x = inputs()
    model = encoder(geometry, down)
    with context():
        ref, mean, logvar = run(model, x, geometry, return_raw=True)
        got, cm, cl = run(model, x, geometry, return_raw=True, chunk_size=256)
        assert torch.equal(ref.coords, got.coords)
        assert torch.equal(got.feats, cm)
        for a, b in [(got.feats, ref.feats), (cm, mean), (cl, logvar)]:
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
        assert model.last_chunk_stats['convolution_chunks'] > 1
        assert model.last_chunk_stats['halo_rows'] > 0
        if down == 'SparseResBlockS2C3d':
            assert model.last_chunk_stats['spatial2channel_chunks'] > 1
        # Downsample caches remain global and can reconstruct the input grid.
        up = sp.SparseChannel2Spatial(2) if down == 'SparseResBlockS2C3d' else sp.SparseUpsample(2)
        if down == 'SparseResBlockS2C3d':
            got = got.replace(torch.ones(len(got.coords), 64, device='cuda'))
        reconstructed = up(up(got))
        assert torch.equal(reconstructed.coords, x.coords)
        rng = torch.cuda.get_rng_state()
        ref_sample = run(model, x, geometry, sample_posterior=True)
        rng_after = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
        sample = run(model, x, geometry, sample_posterior=True, chunk_size=256)
        assert torch.equal(torch.cuda.get_rng_state(), rng_after)
        torch.testing.assert_close(sample.feats, ref_sample.feats, atol=5e-5, rtol=5e-5)
        # Input feature buffers must not be modified by in-place residuals.
        again = run(model, x, geometry)
        torch.testing.assert_close(again.feats, ref.feats, atol=3e-5, rtol=3e-5)


def test_spatial2channel_is_bitwise_exact_across_split_parents():
    x = inputs()
    op = sp.SparseSpatial2Channel(2).cuda().eval()
    with torch.inference_mode():
        reference_input = sp.SparseTensor(x.feats, x.coords)
        full = op(reference_input)
        fresh = sp.SparseTensor(x.feats, x.coords)
        with inference_chunks(113) as policy:
            got = op(fresh)
        assert torch.equal(full.coords, got.coords)
        assert torch.equal(full.feats, got.feats)
        assert policy.stats['spatial2channel_max_rows'] <= 113
        for key in ['spatial2channel_2']:
            for a, b in zip(reference_input.get_spatial_cache(key), fresh.get_spatial_cache(key)):
                assert torch.equal(a, b)
        recovered = sp.SparseChannel2Spatial(2)(got)
        assert torch.equal(recovered.coords, x.coords)
        assert torch.equal(recovered.feats, x.feats)


def test_encoder_auto_decisions_and_guards():
    x = inputs()
    model = encoder()
    with pytest.raises(RuntimeError, match='no_grad'):
        run(model, x, chunk_size='auto')
    with torch.no_grad():
        model.train()
        with pytest.raises(RuntimeError, match='eval'):
            run(model, x, chunk_size='auto')
        model.eval()
        ref = run(model, x)
        whole = run(model, x, chunk_size='auto', chunk_threshold_mb=100_000)
        torch.testing.assert_close(whole.feats, ref.feats, atol=3e-5, rtol=3e-5)
        assert model.last_chunk_stats['auto_skipped_ops'] > 0
        assert model.last_chunk_stats.get('convolution_chunks', 0) == 0
        chunk = run(model, x, chunk_size='auto', chunk_threshold_mb=1, chunk_memory_mb=.125)
        torch.testing.assert_close(chunk.feats, ref.feats, atol=3e-5, rtol=3e-5)
        assert model.last_chunk_stats['auto_max_voxels'] == len(x.feats)
        assert model.last_chunk_stats['convolution_chunks'] > 1
        for bad in [0, -1, float('nan')]:
            with pytest.raises(ValueError, match='threshold'):
                run(model, x, chunk_size='auto', chunk_threshold_mb=bad)
        with pytest.raises(ValueError, match='requires chunk_size'):
            run(model, x, chunk_memory_mb=64)
        assert active_policy() is None
        run(model, x)
        assert model.last_chunk_stats == {}
    # Existing non-chunk training remains differentiable.
    model.train()
    run(model, x).feats.square().mean().backward()
    assert model.input_layer.weight.grad is not None


def test_encoder_preserves_gemm_rows_with_tf32_enabled():
    # Guard against chunking the input/output projections or the ConvNeXt MLP:
    # changing GEMM row counts selects different TF32/FP16 algorithms on A100.
    torch.set_float32_matmul_precision('high')
    x = inputs()
    model = encoder()
    seen = {}
    handles = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            def capture(module, args, name=name):
                value = args[0]
                feats = value.feats if isinstance(value, sp.VarLenTensor) else value
                seen.setdefault(name, []).append(len(feats))
            handles.append(layer.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            run(model, x)
            reference_rows = {key: list(rows) for key, rows in seen.items()}
            seen.clear()
            run(model, x, chunk_size=256)
        assert seen == reference_rows
        assert model.last_chunk_stats['preserve_linear_shapes']
        assert model.last_chunk_stats['convolution_chunks'] > 1
    finally:
        for handle in handles:
            handle.remove()
