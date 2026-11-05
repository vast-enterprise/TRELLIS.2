"""Local-input, rotation and cleanup contracts; no Blender/GPU required."""
import contextlib
import io
import pickle
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import create_geo_pbr as pipeline
import create_geo_pbr_local


def fixture_dump():
    return {'objects': [{
        'vertices': np.asarray([[0.2, 0, 0], [0, 0.3, 0], [0, 0, 0.4]], dtype=np.float32),
        'faces': np.asarray([[0, 1, 2]], dtype=np.int32),
        'normals': np.asarray([[[1, 0, 0]] * 3], dtype=np.float32),
    }]}


def fake_dump(_blender, _glb, destination):
    with destination.open('wb') as stream:
        pickle.dump(fixture_dump(), stream)
    destination.with_name(destination.name + '_error.txt').write_text('debug')


def test_local_pickle_rotates_vertices_and_normals_without_bos():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        glb = root / 'input.glb'
        glb.write_bytes(b'original input')
        with mock.patch.object(pipeline, '_make_bos_client', side_effect=AssertionError('BOS called')), \
                mock.patch.object(pipeline, 'download_glb', side_effect=AssertionError('download called')), \
                mock.patch.object(pipeline, 'run_blender_dump', side_effect=fake_dump), \
                mock.patch.object(pipeline, '_convert_pbr_attributes', side_effect=AssertionError('voxelization called')):
            assert create_geo_pbr_local.main([
                'asset', '--glb', str(glb), '--output-dir', str(root),
                '--rotation-degrees', '0', '0', '90', '--pkl-only',
            ]) == 0
        with (root / 'asset.pbr.pkl').open('rb') as stream:
            dump = pickle.load(stream)
        np.testing.assert_allclose(dump['objects'][0]['vertices'],
                                   [[0, .2, 0], [-.3, 0, 0], [0, 0, .4]], atol=1e-7)
        np.testing.assert_allclose(dump['objects'][0]['normals'], [[[0, 1, 0]] * 3], atol=1e-7)
        with (root / '.cache/asset.pbr.pkl').open('rb') as stream:
            cached = pickle.load(stream)
        np.testing.assert_array_equal(cached['objects'][0]['vertices'], dump['objects'][0]['vertices'])
        assert glb.read_bytes() == b'original input'
        assert not (root / 'asset.geometry.vxz').exists()
        assert not (root / '.cache/asset.pbr.pkl_error.txt').exists()


def test_local_conversion_and_upload_cleanup_preserve_input():
    for upload, failure in [(False, False), (True, False), (True, True)]:
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            # Include the collision with the traditional downloaded cache name.
            cache = root / '.cache'
            cache.mkdir()
            glb = cache / 'asset.glb'
            glb.write_bytes(b'original input')
            client = stack.enter_context(mock.patch.object(pipeline, '_make_bos_client', return_value=object()))
            stack.enter_context(mock.patch.object(pipeline, '_download_input_glb', side_effect=AssertionError('download called')))
            stack.enter_context(mock.patch.object(pipeline, 'run_blender_dump', side_effect=fake_dump))
            coord = torch.tensor([[1, 2, 3]], dtype=torch.int32)
            attr = {'base_color': torch.tensor([[10, 20, 30]], dtype=torch.uint8)}
            stack.enter_context(mock.patch.object(pipeline, '_convert_pbr_attributes', return_value=(coord, attr)))
            stack.enter_context(mock.patch.object(pipeline, 'convert_geometry', return_value=(coord, attr)))
            for name in ('_write_pbr', '_write_geometry'):
                stack.enter_context(mock.patch.object(pipeline, name, side_effect=lambda path, *_: path.write_bytes(b'voxel')))
            uploader = stack.enter_context(mock.patch.object(pipeline, 'upload_file',
                                           side_effect=RuntimeError('upload failed') if failure else None))
            argv = ['asset', '--glb', str(glb), '--output-dir', str(root)]
            if upload:
                argv.append('--upload')
            if failure:
                try:
                    create_geo_pbr_local.main(argv)
                except RuntimeError as error:
                    assert str(error) == 'upload failed'
                else:
                    raise AssertionError('upload failure was swallowed')
            else:
                assert create_geo_pbr_local.main(argv) == 0
            assert client.call_count == int(upload)
            assert uploader.call_count == (1 if failure else 2 if upload else 0)
            assert glb.read_bytes() == b'original input'
            for name in ('asset.vxz', 'asset.geometry.vxz', '.cache/asset.pbr.pkl'):
                assert (root / name).exists() == (not upload)
            assert not (root / '.cache/asset.pbr.pkl_error.txt').exists()


def test_local_parser_rejects_invalid_requests():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        glb = root / 'input.glb'
        glb.write_bytes(b'input')
        base = ['asset', '--glb', str(glb), '--output-dir', str(root)]
        invalid = [
            ['asset'],
            ['asset', '--glb', str(root / 'missing.glb')],
            base + ['--rotation-degrees', 'nan', '0', '0'],
            base + ['--flip-front-back', '--rotation', '0', '0', '90'],
            base + ['--pkl-only', '--upload'],
            base + ['--pkl-only', '--store-mesh'],
            base + ['--store-mesh'],
            base + ['--bos-uri', 'another.glb'],
        ]
        collision = root / 'asset.pbr.pkl'
        collision.write_bytes(b'input')
        invalid.append(['asset', '--glb', str(collision), '--cache-dir', str(root),
                        '--output-dir', str(root / 'out')])
        for argv in invalid:
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    pipeline._parse_args(pipeline.build_parser(local=True), argv)
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError(f'invalid arguments accepted: {argv}')
        args = pipeline._parse_args(pipeline.build_parser(local=True),
                                    base + ['--output-format', 'vxzm', '--store-mesh', '--flip-front-back'])
        assert args.rotation_degrees == (0, 0, 180)
        assert args.store_mesh


if __name__ == '__main__':
    test_local_pickle_rotates_vertices_and_normals_without_bos()
    test_local_conversion_and_upload_cleanup_preserve_input()
    test_local_parser_rejects_invalid_requests()
    print('Local geometry/PBR tests passed')
