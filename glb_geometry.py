"""Read static GLB triangle geometry without Blender or material decoding.

The returned arrays own their storage (the mmap is closed before returning).
Vertices use Blender's world Z-up frame and, by default, a centered unit cube.
Unsupported features raise a distinct exception so callers can use Blender.
"""
from __future__ import annotations

import json
import mmap
from pathlib import Path
import struct

import numpy as np


class UnsupportedGeometryFeature(ValueError):
    """A valid glTF feature requires the Blender import path."""


# These affect only fields that geometry-only deliberately does not export.
_MATERIAL_EXTENSIONS = {
    'KHR_materials_unlit', 'KHR_materials_pbrSpecularGlossiness',
    'KHR_materials_clearcoat', 'KHR_materials_sheen', 'KHR_materials_transmission',
    'KHR_materials_volume', 'KHR_materials_ior', 'KHR_materials_specular',
    'KHR_materials_emissive_strength', 'KHR_materials_iridescence',
    'KHR_materials_anisotropy', 'KHR_materials_variants', 'KHR_texture_transform',
    'KHR_texture_basisu', 'EXT_texture_webp', 'EXT_texture_avif',
    'KHR_lights_punctual',
}


def _entry(items, index, label):
    if not isinstance(index, int) or not 0 <= index < len(items):
        raise ValueError(f'Invalid {label} index: {index!r}')
    return items[index]


def _integer(value, label, minimum=0):
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f'Invalid {label}: {value!r}')
    return value


def _header(data):
    if len(data) < 20:
        raise ValueError('Truncated GLB header')
    magic, version, length = struct.unpack_from('<4sII', data)
    if magic != b'glTF' or version != 2 or length != len(data):
        raise ValueError('Expected a GLB 2.0 file with a valid byte length')
    document, binary = None, None
    offset = 12
    while offset < length:
        if offset + 8 > length:
            raise ValueError('Truncated GLB chunk header')
        size, kind = struct.unpack_from('<II', data, offset)
        start, end = offset + 8, offset + 8 + size
        if end > length or size % 4:
            raise ValueError('Invalid GLB chunk length')
        if offset == 12 and kind != 0x4E4F534A:
            raise ValueError('GLB first chunk must be JSON')
        if kind == 0x4E4F534A:
            if document is not None:
                raise ValueError('Duplicate GLB JSON chunk')
            document = json.loads(data[start:end])
        elif kind == 0x004E4942:
            if binary is not None:
                raise ValueError('Duplicate GLB BIN chunk')
            binary = (start, size)
        offset = end
    if document is None or binary is None:
        raise UnsupportedGeometryFeature('GLB requires JSON and an embedded BIN chunk')
    if document.get('asset', {}).get('version') != '2.0':
        raise UnsupportedGeometryFeature('Only glTF 2.0 is supported')
    extensions = set(document.get('extensionsUsed', [])) | set(document.get('extensionsRequired', []))
    unsupported = extensions - _MATERIAL_EXTENSIONS
    if unsupported:
        raise UnsupportedGeometryFeature(f'glTF extensions: {sorted(unsupported)}')
    if document.get('animations') or document.get('skins'):
        raise UnsupportedGeometryFeature('Animated or skinned geometry')
    buffers = document.get('buffers', [])
    if len(buffers) != 1 or 'uri' in buffers[0]:
        raise UnsupportedGeometryFeature('External or multiple geometry buffers')
    size = _integer(buffers[0].get('byteLength'), 'buffer byteLength')
    if not size <= binary[1] <= size + 3:
        raise ValueError('BIN chunk does not match buffer byteLength')
    return document, binary[0], size


def _accessor(doc, data, binary_start, binary_size, index, *, positions):
    a = _entry(doc.get('accessors', []), index, 'accessor')
    if 'sparse' in a:
        raise UnsupportedGeometryFeature('Sparse geometry accessors')
    if a.get('extensions'):
        raise UnsupportedGeometryFeature('Geometry accessor extensions')
    kind = a.get('componentType')
    columns = 3 if positions else 1
    if positions:
        if a.get('type') != 'VEC3' or kind != 5126 or a.get('normalized', False):
            raise UnsupportedGeometryFeature('POSITION must be unnormalized float32 VEC3')
        dtype = np.dtype('<f4')
    else:
        if a.get('type') != 'SCALAR' or kind not in (5121, 5123, 5125) or a.get('normalized', False):
            raise ValueError('Indices must be unsigned integer SCALAR')
        dtype = np.dtype({5121: 'u1', 5123: '<u2', 5125: '<u4'}[kind])
    count = _integer(a.get('count'), 'accessor count', 1)
    view = _entry(doc.get('bufferViews', []), a.get('bufferView'), 'bufferView')
    if view.get('extensions'):
        raise UnsupportedGeometryFeature('Geometry bufferView extensions')
    if view.get('buffer') != 0:
        raise UnsupportedGeometryFeature('External geometry buffer')
    view_start = _integer(view.get('byteOffset', 0), 'bufferView byteOffset')
    view_size = _integer(view.get('byteLength'), 'bufferView byteLength')
    start = _integer(a.get('byteOffset', 0), 'accessor byteOffset')
    width = columns * dtype.itemsize
    stride = _integer(view.get('byteStride', width), 'bufferView byteStride', width)
    if stride % dtype.itemsize or start % dtype.itemsize or (view_start + start) % dtype.itemsize:
        raise ValueError('Misaligned geometry accessor')
    if 'byteStride' in view and (not positions or stride > 252 or stride % 4):
        raise ValueError('Invalid geometry byteStride')
    if view_start + view_size > binary_size or start + (count - 1) * stride + width > view_size:
        raise ValueError('Geometry accessor exceeds its bufferView')
    # Copy only geometry. Embedded texture bytes are never decoded or copied.
    return np.ndarray((count, columns), dtype=dtype, buffer=data,
                      offset=binary_start + view_start + start,
                      strides=(stride, dtype.itemsize)).copy()


def _node_matrix(node):
    if 'matrix' in node:
        if any(key in node for key in ('translation', 'rotation', 'scale')):
            raise ValueError('Node cannot specify both matrix and TRS')
        matrix = np.asarray(node['matrix'], dtype=np.float32)
        if matrix.shape != (16,):
            raise ValueError('Node matrix must contain 16 values')
        matrix = matrix.reshape(4, 4).T.copy()  # glTF column-major
        if not np.array_equal(matrix[3], [0, 0, 0, 1]):
            raise ValueError('Node matrix must be affine')
    else:
        t = np.asarray(node.get('translation', [0, 0, 0]), dtype=np.float32)
        s = np.asarray(node.get('scale', [1, 1, 1]), dtype=np.float32)
        q = np.asarray(node.get('rotation', [0, 0, 0, 1]), dtype=np.float64)
        if t.shape != (3,) or s.shape != (3,) or q.shape != (4,):
            raise ValueError('Invalid node TRS shape')
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid node rotation quaternion')
        x, y, z, w = q / norm
        r = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = r * s
        matrix[:3, 3] = t
    if not np.isfinite(matrix).all():
        raise ValueError('Non-finite node transform')
    return matrix


def _mesh_primitives(doc, data, binary_start, binary_size, mesh_index):
    mesh = _entry(doc.get('meshes', []), mesh_index, 'mesh')
    if mesh.get('weights') or mesh.get('extensions'):
        raise UnsupportedGeometryFeature('Mesh weights or extensions')
    result = []
    for prim in mesh.get('primitives', []):
        if prim.get('mode', 4) != 4:
            raise UnsupportedGeometryFeature('Only TRIANGLES primitives are supported')
        if prim.get('targets'):
            raise UnsupportedGeometryFeature('Morph targets')
        if set(prim.get('extensions', {})) - {'KHR_materials_variants'}:
            raise UnsupportedGeometryFeature('Geometry primitive extensions')
        v = _accessor(doc, data, binary_start, binary_size,
                      prim.get('attributes', {}).get('POSITION'), positions=True)
        if not np.isfinite(v).all():
            raise ValueError('Non-finite POSITION values')
        if len(v) > np.iinfo(np.int32).max:
            raise UnsupportedGeometryFeature('More than int32 vertices in a primitive')
        indices = (_accessor(doc, data, binary_start, binary_size, prim['indices'], positions=False).ravel()
                   if 'indices' in prim else np.arange(len(v), dtype=np.uint32))
        if len(indices) % 3 or int(indices.max()) >= len(v):
            raise ValueError('Invalid triangle indices')
        # Blender drops unused vertices before computing scene bounds. Use a
        # linear-time bitmap instead of sorting every triangle corner.
        used = np.zeros(len(v), dtype=bool)
        used[indices] = True
        if not used.all():
            remap = np.cumsum(used, dtype=np.int32) - 1
            indices = remap[indices]
            v = v[used]
        result.append((v, indices.reshape(-1, 3).astype(np.int32, copy=False)))
    return result


def load_glb_geometry(path: str | Path, *, normalize: bool = True) -> dict:
    """Return geometry-only dump data for the default scene of a static GLB.

    Supports multiple primitives, mesh instances, nested matrix/TRS nodes,
    non-indexed triangles and interleaved/accessor-offset buffer layouts.
    Does not weld vertices, fix normals or read UVs/materials. Sparse,
    compressed, animated, skinned and morphed geometry requires Blender.

    Unit-cube normalization uses two float32 stages like the Blender path;
    ``convert_geometry`` performs the second, strict-interior normalization.
    Arbitrary transformed scenes are geometrically equivalent, but are not
    promised bit-identical to Blender's scene evaluation/rounding.
    """
    with Path(path).open('rb') as stream:
        if stream.seek(0, 2) == 0:
            raise ValueError('Empty GLB file')
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            doc, binary_start, binary_size = _header(data)
            nodes = doc.get('nodes', [])
            scene = _entry(doc.get('scenes', []), doc.get('scene', 0), 'scene')
            stack = [(index, np.eye(4, dtype=np.float32)) for index in reversed(scene.get('nodes', []))]
            visited, cache, objects = set(), {}, []
            while stack:
                index, parent = stack.pop()
                node = _entry(nodes, index, 'node')
                if index in visited:
                    raise ValueError('Scene contains a cycle or a node with multiple parents')
                visited.add(index)
                if 'skin' in node or node.get('weights') or set(node.get('extensions', {})) - {'KHR_lights_punctual'}:
                    raise UnsupportedGeometryFeature('Node skin, weights or geometry extensions')
                world = parent @ _node_matrix(node)
                if 'mesh' in node:
                    mesh_index = node['mesh']
                    if mesh_index not in cache:
                        cache[mesh_index] = _mesh_primitives(doc, data, binary_start, binary_size, mesh_index)
                    for v, f in cache[mesh_index]:
                        v = v @ world[:3, :3].T + world[:3, 3]
                        v = v[:, [0, 2, 1]].copy(order='C')
                        v[:, 1] *= -1  # glTF Y-up -> Blender Z-up
                        objects.append({'vertices': v, 'faces': f, 'uvs': None, 'matIDs': None})
                stack.extend((child, world) for child in reversed(node.get('children', [])))
    if not objects:
        raise ValueError('GLB default scene contains no triangle meshes')
    if normalize:
        lo = np.minimum.reduce([obj['vertices'].min(0) for obj in objects])
        hi = np.maximum.reduce([obj['vertices'].max(0) for obj in objects])
        extent = float((hi - lo).max())
        if not np.isfinite(extent) or extent <= 0:
            raise ValueError('Degenerate or non-finite scene bounds')
        scale = np.float32(1 / extent)
        offset = -(lo * scale + hi * scale) * np.float32(.5)
        for obj in objects:
            obj['vertices'] *= scale
            obj['vertices'] += offset
    return {'objects': objects, 'geometry_only': True,
            'coordinate_system': 'blender_world_z_up', 'geometry_loader': 'glb'}
