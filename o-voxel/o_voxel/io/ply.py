from typing import *
import io
import torch
import numpy as np


__all__ = [
    "read_ply",
    "write_ply",
]


DTYPE_MAP = {
    torch.uint8: 'u1',
    torch.uint16: 'u2',
    torch.uint32: 'u4',
    torch.int8: 'i1',
    torch.int16: 'i2',
    torch.int32: 'i4',
    torch.float32: 'f4',
    torch.float64: 'f8'
}


def read_ply(file) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Read a PLY file containing voxels.

    Args:
        file: Path or file-like object of the PLY file.
        
    Returns:
        torch.Tensor: the coordinates of the voxels.
        Dict[str, torch.Tensor]: the attributes of the voxels.
    """
    # Keep PLY optional for the core voxelizer.  The GLB smoke test uses
    # Open3D for visualization output, so importing ``o_voxel`` must not
    # force-install the separate plyfile package.
    import plyfile

    plydata = plyfile.PlyData.read(file)
    xyz = np.stack([plydata.elements[0][k] for k in ['x', 'y', 'z']], axis=1)
    coord = np.round(xyz).astype(int)
    coord = torch.from_numpy(coord)
    field_names = plydata.elements[0].data.dtype.names
    attr_keys = [k for k in field_names if k not in ['x', 'y', 'z']]

    # ``red``/``green``/``blue`` is the conventional PLY color layout and is
    # recognized by viewers such as MeshLab and CloudCompare.  Map it back to
    # our native attribute name on read so a PLY written for visualization can
    # still be consumed by o-voxel.
    attr = {}
    rgb_names = ('red', 'green', 'blue')
    if all(k in field_names for k in rgb_names):
        attr['base_color'] = np.stack([plydata.elements[0][k] for k in rgb_names], axis=1)
        attr_keys = [k for k in attr_keys if k not in rgb_names]

    # Group the legacy ``<attribute>_<channel>`` fields once per attribute.
    # The old implementation kept duplicate names here, which was harmless
    # but unnecessarily repeated work and made mixed PLY layouts brittle.
    attr_names = []
    for key in attr_keys:
        if '_' not in key:
            continue
        name, channel = key.rsplit('_', 1)
        if channel.isdigit() and name not in attr_names:
            attr_names.append(name)
    for name in attr_names:
        channel_keys = [k for k in attr_keys
                        if k.startswith(f'{name}_') and k[len(name) + 1:].isdigit()]
        channel_keys.sort(key=lambda k: int(k[len(name) + 1:]))
        if channel_keys:
            attr[name] = np.stack([plydata.elements[0][k] for k in channel_keys], axis=1)
    attr = {k: torch.from_numpy(v) for k, v in attr.items()}
    
    return coord, attr


def write_ply(file, coord: torch.Tensor, attr: Dict[str, torch.Tensor]):
    """
    Write a PLY file containing voxels.
    
    Args:
        file: Path or file-like object of the PLY file.
        coord: the coordinates of the voxels.
        attr: the attributes of the voxels.
    """    
    import plyfile

    dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    for k, v in attr.items():
        assert v.ndim == 2 and v.shape[0] == len(coord), \
            f"Attribute {k} must have shape [num_voxels, channels]"
        assert v.dtype in DTYPE_MAP, f"Unsupported data type {v.dtype} for attribute {k}"
        # Emit the conventional property names for the material color.  The
        # previous generic ``base_color_0/1/2`` fields are valid PLY but most
        # visualization tools do not recognize them as color data.
        if k == 'base_color' and v.shape[-1] == 3:
            dtypes.extend([('red', DTYPE_MAP[v.dtype]),
                           ('green', DTYPE_MAP[v.dtype]),
                           ('blue', DTYPE_MAP[v.dtype])])
            continue
        for j in range(v.shape[-1]):
            dtypes.append((f'{k}_{j}', DTYPE_MAP[v.dtype]))
    data = np.empty(len(coord), dtype=dtypes)
    all_chs = np.concatenate([coord.cpu().numpy().astype(np.float32)] + [v.cpu().numpy() for v in attr.values()], axis=1)
    data[:] = list(map(tuple, all_chs))
    plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(file)
