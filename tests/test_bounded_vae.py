"""Budget fallback, full GEMM shapes, spatial halos and allocator restoration."""
import pytest
import torch

from test_chunked_encode import inputs, encoder, run
from test_chunked_decode import decoder, grid
from trellis2.modules import sparse as sp
from trellis2.modules.sparse.bounded_vae import HostVae, bounded_vae, cuda_budget
from trellis2.modules.sparse.inference_chunk import active_policy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def precise():
    import flex_gemm
    from trellis2.modules.sparse.conv import config
    old, precision = config.FLEX_GEMM_ALGO, torch.get_float32_matmul_precision()
    config.FLEX_GEMM_ALGO = 'explicit_gemm'
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        config.FLEX_GEMM_ALGO = old
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_host_multilayer_encode_decode_and_guidance(context):
    with context():
        x = inputs()
        enc = encoder()
        ref, mean, logvar = run(enc, x, return_raw=True)
        executor = HostVae(enc, 22 * 1024**3, max_points=256)
        executor.resident_blocks = False
        got, gm, gl = executor.encode(sp.SparseTensor(x.feats, x.coords), return_raw=True)
        assert torch.equal(ref.coords, got.coords)
        torch.testing.assert_close(gm, mean, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(gl, logvar, atol=3e-5, rtol=3e-5)
        assert executor.policy.stats['halo_rows'] > 0
        assert executor.policy.stats['host_full_linear_max_rows'] == len(x.feats)
        shape, texture, z = decoder(), decoder(False), grid(n=5)
        ref, subs = shape(z, return_subs=True)
        executor = HostVae(shape, 22 * 1024**3, max_points=256)
        executor.resident_blocks = False
        got, gs = executor.decode(sp.SparseTensor(z.feats, z.coords), return_subs=True)
        assert torch.equal(ref.coords, got.coords)
        torch.testing.assert_close(got.feats, ref.feats, atol=3e-5, rtol=3e-5)
        for a, b in zip(gs, subs):
            assert torch.equal(a.coords, b.coords)
            torch.testing.assert_close(a.feats, b.feats, atol=3e-5, rtol=3e-5)
        ref_tex = texture(sp.SparseTensor(z.feats, z.coords), guide_subs=subs)
        got_tex = HostVae(texture, 22 * 1024**3, max_points=256).decode(
            sp.SparseTensor(z.feats, z.coords), guide_subs=gs)
        assert torch.equal(got_tex.coords, ref_tex.coords)
        torch.testing.assert_close(got_tex.feats, ref_tex.feats, atol=3e-5, rtol=3e-5)


def test_oom_fallback_and_allocator_restore(monkeypatch):
    model, x = encoder(), inputs()
    fraction = torch.cuda.get_per_process_memory_fraction()
    with torch.no_grad():
        reference = run(model, x)
        def fail(*args, **kwargs):
            raise torch.cuda.OutOfMemoryError('injected GPU workspace exhaustion')
        monkeypatch.setattr(model, '_encode', fail)
        got = run(model, x, max_memory_gb=23)
        torch.testing.assert_close(got.feats, reference.feats, atol=3e-5, rtol=3e-5)
        assert model.last_chunk_stats['host_offload']
        assert model.last_chunk_stats['gpu_oom_fallbacks'] == 1
        assert active_policy() is None
        assert torch.cuda.get_per_process_memory_fraction() == fraction
        with pytest.raises(ValueError, match='injected'):
            with cuda_budget('cuda', 23):
                assert torch.cuda.get_per_process_memory_fraction() <= fraction
                raise ValueError('injected')
        assert torch.cuda.get_per_process_memory_fraction() == fraction
        for value in [0, 1, float('nan')]:
            with pytest.raises(ValueError, match='max_memory_gb'):
                run(model, x, max_memory_gb=value)
    with pytest.raises(RuntimeError, match='no_grad'):
        run(model, x, max_memory_gb=23)


def test_mesh_chunks_keep_cross_boundary_faces_and_order():
    from o_voxel.convert import flexible_dual_grid_to_mesh
    torch.manual_seed(91)
    coords = grid(n=7).coords[:, 1:]
    vertices = torch.rand(len(coords), 3, device='cuda')
    flags = torch.rand(len(coords), 3, device='cuda') > .3
    weights = torch.rand(len(coords), 1, device='cuda')
    kwargs = dict(aabb=[[-.5]*3, [.5]*3], grid_size=16)
    with torch.inference_mode():
        reference = flexible_dual_grid_to_mesh(coords, vertices, flags, weights, **kwargs)
        for rows in (1, 17, 500):
            got = flexible_dual_grid_to_mesh(coords, vertices, flags, weights, chunk_size=rows, **kwargs)
            assert all(torch.equal(a, b) for a, b in zip(reference, got))


def test_public_geometry_decoder_budget_includes_mesh(monkeypatch):
    from trellis2.models.sc_vaes import fdg_vae
    model = fdg_vae.FlexiDualGridVaeDecoder(resolution=16, model_channels=[16, 8],
        latent_channels=8, num_blocks=[1, 0], block_type=['SparseConvNeXtBlock3d']*2,
        up_block_type=['SparseResBlockC2S3d'], block_args=[{}, {}]).cuda().eval()
    original = fdg_vae.flexible_dual_grid_to_mesh
    fractions = []
    def extract(*args, **kwargs):
        fractions.append(torch.cuda.get_per_process_memory_fraction())
        return original(*args, **kwargs)
    monkeypatch.setattr(fdg_vae, 'flexible_dual_grid_to_mesh', extract)
    x = grid(n=3)
    prior = torch.cuda.get_per_process_memory_fraction()
    with torch.inference_mode():
        ref = model(sp.SparseTensor(x.feats, x.coords))
        got = model(sp.SparseTensor(x.feats, x.coords), max_memory_gb=23)
        assert torch.equal(got[0].vertices, ref[0].vertices)
        assert torch.equal(got[0].faces, ref[0].faces)
    assert fractions[1] <= 22 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    assert torch.cuda.get_per_process_memory_fraction() == prior


def test_external_memory_reserve_and_nested_budget(monkeypatch):
    prior = torch.cuda.get_per_process_memory_fraction()
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, 'device_memory_used',
                        lambda device: torch.cuda.memory_reserved(device) + 3 * gib)
    with torch.no_grad():
        with cuda_budget('cuda', 23) as outer:
            assert outer <= int(19.75 * gib)
            with cuda_budget('cuda', 23) as inner:
                assert inner == outer
            total = torch.cuda.get_device_properties(0).total_memory
            assert torch.cuda.get_per_process_memory_fraction() == outer / total
    assert torch.cuda.get_per_process_memory_fraction() == prior


def test_host_decoder_preserves_encoder_upsampling_cache():
    from trellis2.modules.sparse.spatial.spatial2channel import SparseSpatial2Channel
    with torch.no_grad():
        fine = grid(n=8, channels=1)
        z = SparseSpatial2Channel(2)(fine)
        model = decoder(False)
        reference = model(z)
        executor = HostVae(model, 21 * 1024**3, max_points=256)
        executor.resident_blocks = False
        got = executor.decode(z)
        assert torch.equal(got.coords, fine.coords)
        torch.testing.assert_close(got.feats, reference.feats, atol=3e-5, rtol=3e-5)
