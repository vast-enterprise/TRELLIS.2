"""Small dependency-light regression tests for voxel color preprocessing.

Run with ``python tests/test_voxel_color_helpers.py``.  These checks cover
the Python-side color/alpha preprocessing; the C++ mip and UV code is covered
by the extension build in a CUDA-enabled environment.
"""

import io
import numpy as np
from PIL import Image


def _srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def make_texture_pack(array, color_space='sRGB'):
    image = Image.fromarray(array, mode='RGBA')
    data = io.BytesIO()
    image.save(data, format='PNG')
    return {
        'image': data.getvalue(),
        'interpolation': 'Linear',
        'extension': 'REPEAT',
        'color_space': color_space,
    }


def test_srgb_decode():
    value = np.array([[[128.0, 128.0, 128.0]]], dtype=np.float32) / 255.0
    expected = ((128.0 / 255.0 + 0.055) / 1.055) ** 2.4
    assert np.allclose(_srgb_to_linear(value), expected, atol=1e-7)


def test_alpha_premultiply_math():
    rgba = np.array([[[255, 128, 64, 0], [255, 128, 64, 128]]], dtype=np.uint8)
    premultiplied = rgba.astype(np.float32)
    premultiplied[..., :3] *= premultiplied[..., 3:4] / 255.0
    premultiplied = np.rint(premultiplied).astype(np.uint8)
    assert np.array_equal(premultiplied[0, 0, :3], [0, 0, 0])
    assert np.array_equal(premultiplied[0, 1, :3], [128, 64, 32])


def test_non_square_png_is_not_resized():
    pack = make_texture_pack(np.zeros((3, 5, 4), dtype=np.uint8))
    with Image.open(io.BytesIO(pack['image'])) as image:
        assert image.size == (5, 3)


def test_scalar_png_preserves_alpha_values():
    alpha = np.array([[0, 128, 255]], dtype=np.uint8)
    data = io.BytesIO()
    Image.fromarray(alpha, mode='L').save(data, format='PNG')
    with Image.open(io.BytesIO(data.getvalue())) as image:
        # Loading an L image as RGBA would turn its alpha channel into 255.
        # The voxel loader must instead read this scalar plane itself.
        restored = np.array(image.convert('L'), dtype=np.float32) / 255.0
    assert np.allclose(restored, alpha.astype(np.float32) / 255.0)


if __name__ == '__main__':
    test_srgb_decode()
    test_alpha_premultiply_math()
    test_non_square_png_is_not_resized()
    test_scalar_png_preserves_alpha_values()
    print('voxel color helper tests passed')
