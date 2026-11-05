"""Collect the exact scene mesh already transformed for voxel sampling."""
import numpy as np


def mesh_from_dump(dump):
    """Concatenate nonempty objects without welding, reorienting or rescaling.

    Call after scene normalization. Keeping object-local vertices (including
    coincident vertices) and winding preserves open/non-manifold surfaces.
    """
    from ..io.vxzm import _mesh_arrays

    vertices, faces, offset = [], [], 0
    for obj in dump.get('objects', []):
        v, f = np.asarray(obj.get('vertices')), np.asarray(obj.get('faces'))
        if v.size == 0 or f.size == 0:
            continue
        v, f = _mesh_arrays(v, f)
        vertices.append(v)
        faces.append(f.astype(np.int64) + offset)
        offset += len(v)
    if not vertices:
        raise ValueError('PBR dump contains no non-empty mesh objects')
    return _mesh_arrays(np.concatenate(vertices), np.concatenate(faces))
