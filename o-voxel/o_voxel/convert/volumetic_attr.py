from typing import *
import io
import importlib.util
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import trimesh
import trimesh.visual

from .. import _C

__all__ = [
    "textured_mesh_to_volumetric_attr",
    "blender_dump_to_volumetric_attr"
]


ALPHA_MODE_ENUM = {
    "OPAQUE": 0,
    "MASK": 1,
    "BLEND": 2,
}


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode an IEC sRGB image into Blender shader linear RGB."""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def textured_mesh_to_volumetric_attr(
    mesh: Union[trimesh.Scene, trimesh.Trimesh, str],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    aabb: Union[list, tuple, np.ndarray, torch.Tensor] = None,
    mip_level_offset: float = 0.0,
    verbose: bool = False,
    timing: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Voxelize a mesh into a sparse voxel grid with PBR properties.
    
    Args:
        mesh (trimesh.Scene, trimesh.Trimesh, str): The input mesh.
            If a string is provided, it will be loaded as a mesh using trimesh.load().
        voxel_size (float, list, tuple, np.ndarray, torch.Tensor): The size of each voxel.
        grid_size (int, list, tuple, np.ndarray, torch.Tensor): The size of the grid.
            NOTE: One of voxel_size and grid_size must be provided.
        aabb (list, tuple, np.ndarray, torch.Tensor): The axis-aligned bounding box of the mesh.
            If not provided, it will be computed automatically.
        tile_size (int): The size of the tiles used for each individual voxelization.
        mip_level_offset (float): The mip level offset for texture mip level selection.
        verbose (bool): Whether to print the settings.
        timing (bool): Whether to print the timing information.
        
    Returns:
        torch.Tensor: The indices of the voxels that are occupied by the mesh.
        Dict[str, torch.Tensor]: A dictionary containing the following keys:
        - "base_color": The base color of the occupied voxels.
        - "metallic": The metallic value of the occupied voxels.
        - "roughness": The roughness value of the occupied voxels.
        - "emissive": The emissive value of the occupied voxels.
        - "alpha": The alpha value of the occupied voxels.
        - "normal": The normal of the occupied voxels.
    """
    
    # Load mesh
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh)
    if isinstance(mesh, trimesh.Scene):
        groups = mesh.dump()
    if isinstance(mesh, trimesh.Trimesh):
        groups = [mesh]
    scene = trimesh.Scene(groups)
        
    # Voxelize settings
    assert voxel_size is not None or grid_size is not None, "Either voxel_size or grid_size must be provided"

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
        assert voxel_size.dim() == 1, f"voxel_size must be a 1D tensor, but got {voxel_size.shape}"
        assert voxel_size.size(0) == 3, f"voxel_size must have 3 elements, but got {voxel_size.size(0)}"

    if grid_size is not None:
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32)
        assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
        assert grid_size.dim() == 1, f"grid_size must be a 1D tensor, but got {grid_size.shape}"
        assert grid_size.size(0) == 3, f"grid_size must have 3 elements, but got {grid_size.size(0)}"

    if aabb is not None:
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
        assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
        assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
        assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Auto adjust aabb
    if aabb is None:
        aabb = scene.bounds
        min_xyz = aabb[0]
        max_xyz = aabb[1]

        if voxel_size is not None:
            padding = torch.ceil((max_xyz - min_xyz) / voxel_size) * voxel_size - (max_xyz - min_xyz)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5
        if grid_size is not None:
            padding = (max_xyz - min_xyz) / (grid_size - 1)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5

        aabb = torch.stack([min_xyz, max_xyz], dim=0).float()

    # Fill voxel size or grid size
    if voxel_size is None:
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    if grid_size is None:
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    
    grid_range = torch.stack([torch.zeros_like(grid_size), grid_size], dim=0).int()
        
    # Print settings
    if verbose:
        print(f"Voxelize settings:")
        print(f"  Voxel size: {voxel_size}")
        print(f"  Grid size: {grid_size}")
        print(f"  AABB: {aabb}")
        
    # Load Scene
    scene_buffers = {
        'triangles': [],
        'normals': [],
        'uvs': [],
        'material_ids': [],
        'base_color_factor': [],
        'base_color_texture': [],
        'metallic_factor': [],
        'metallic_texture': [],
        'roughness_factor': [],
        'roughness_texture': [],
        'emissive_factor': [],
        'emissive_texture': [],
        'alpha_mode': [],
        'alpha_cutoff': [],
        'alpha_factor': [],
        'alpha_texture': [],
        'normal_texture': [],
    }
    for sid, (name, g) in tqdm(enumerate(scene.geometry.items()), total=len(scene.geometry), desc="Loading Scene", disable=not verbose):
        if verbose:
            print(f"Geometry: {name}")
            print(f"  Visual: {g.visual}")
            print(f"  Triangles: {g.triangles.shape[0]}")
            print(f"  Vertices: {g.vertices.shape[0]}")
            print(f"  Normals: {g.vertex_normals.shape[0]}")
            if g.visual.material.baseColorFactor is not None:
                print(f"  Base color factor: {g.visual.material.baseColorFactor}")
            if g.visual.material.baseColorTexture is not None:
                print(f"  Base color texture: {g.visual.material.baseColorTexture.size} {g.visual.material.baseColorTexture.mode}")
            if g.visual.material.metallicFactor is not None:
                print(f"  Metallic factor: {g.visual.material.metallicFactor}")
            if g.visual.material.roughnessFactor is not None:
                print(f"  Roughness factor: {g.visual.material.roughnessFactor}")
            if g.visual.material.metallicRoughnessTexture is not None:
                print(f"  Metallic roughness texture: {g.visual.material.metallicRoughnessTexture.size} {g.visual.material.metallicRoughnessTexture.mode}")
            if g.visual.material.emissiveFactor is not None:
                print(f"  Emissive factor: {g.visual.material.emissiveFactor}")
            if g.visual.material.emissiveTexture is not None:
                print(f"  Emissive texture: {g.visual.material.emissiveTexture.size} {g.visual.material.emissiveTexture.mode}")
            if g.visual.material.alphaMode is not None:
                print(f"  Alpha mode: {g.visual.material.alphaMode}")
            if g.visual.material.alphaCutoff is not None:
                print(f"  Alpha cutoff: {g.visual.material.alphaCutoff}")
            if g.visual.material.normalTexture is not None:
                print(f"  Normal texture: {g.visual.material.normalTexture.size} {g.visual.material.normalTexture.mode}")
        
        assert isinstance(g, trimesh.Trimesh), f"Only trimesh.Trimesh is supported, but got {type(g)}"
        assert isinstance(g.visual, trimesh.visual.TextureVisuals), f"Only trimesh.visual.TextureVisuals is supported, but got {type(g.visual)}"
        assert isinstance(g.visual.material, trimesh.visual.material.PBRMaterial), f"Only trimesh.visual.material.PBRMaterial is supported, but got {type(g.visual.material)}"
        triangles = torch.tensor(g.triangles, dtype=torch.float32) - aabb[0].reshape(1, 1, 3)                                                            # [N, 3, 3]
        normals = torch.tensor(g.vertex_normals[g.faces], dtype=torch.float32)                                                                           # [N, 3, 3]
        uvs = torch.tensor(g.visual.uv[g.faces], dtype=torch.float32) if g.visual.uv is not None \
                else torch.zeros(g.triangles.shape[0], 3, 2, dtype=torch.float32)                                                                        # [N, 3, 2]
        baseColorFactor = torch.tensor(g.visual.material.baseColorFactor / 255, dtype=torch.float32) if g.visual.material.baseColorFactor is not None \
                          else torch.ones(3, dtype=torch.float32)                                                                                        # [3]
        baseColorTexture = torch.tensor(np.array(g.visual.material.baseColorTexture.convert('RGBA'))[..., :3], dtype=torch.float32) / 255.0 if g.visual.material.baseColorTexture is not None \
                           else torch.tensor([])                                                                                                                # [H, W, 3]
        metallicFactor = g.visual.material.metallicFactor if g.visual.material.metallicFactor is not None else 1.0
        metallicTexture = torch.tensor(np.array(g.visual.material.metallicRoughnessTexture.convert('RGB'))[..., 2], dtype=torch.float32) / 255.0 if g.visual.material.metallicRoughnessTexture is not None \
                          else torch.tensor([])                                                                                                                 # [H, W]
        roughnessFactor = g.visual.material.roughnessFactor if g.visual.material.roughnessFactor is not None else 1.0
        roughnessTexture = torch.tensor(np.array(g.visual.material.metallicRoughnessTexture.convert('RGB'))[..., 1], dtype=torch.float32) / 255.0 if g.visual.material.metallicRoughnessTexture is not None \
                           else torch.tensor([])                                                                                                                # [H, W]
        emissiveFactor = torch.tensor(g.visual.material.emissiveFactor, dtype=torch.float32) if g.visual.material.emissiveFactor is not None \
                         else torch.zeros(3, dtype=torch.float32)                                                                                        # [3]
        emissiveTexture = torch.tensor(np.array(g.visual.material.emissiveTexture.convert('RGB'))[..., :3], dtype=torch.float32) / 255.0 if g.visual.material.emissiveTexture is not None \
                          else torch.tensor([])                                                                                                                 # [H, W, 3]
        alphaMode = ALPHA_MODE_ENUM[g.visual.material.alphaMode] if g.visual.material.alphaMode in ALPHA_MODE_ENUM else 0
        alphaCutoff = g.visual.material.alphaCutoff if g.visual.material.alphaCutoff is not None else 0.5
        alphaFactor = g.visual.material.baseColorFactor[3] / 255 if g.visual.material.baseColorFactor is not None else 1.0
        alphaTexture = torch.tensor(np.array(g.visual.material.baseColorTexture.convert('RGBA'))[..., 3], dtype=torch.float32) / 255.0 if g.visual.material.baseColorTexture is not None and alphaMode != 0 \
                       else torch.tensor([])                                                                                                                    # [H, W]
        normalTexture = torch.tensor(np.array(g.visual.material.normalTexture.convert('RGB'))[..., :3], dtype=torch.float32) / 255.0 if g.visual.material.normalTexture is not None \
                        else torch.tensor([])                                                                                                                   # [H, W, 3]
        
        scene_buffers['triangles'].append(triangles)
        scene_buffers['normals'].append(normals)
        scene_buffers['uvs'].append(uvs)
        scene_buffers['material_ids'].append(torch.full((triangles.shape[0],), sid, dtype=torch.int32))
        scene_buffers['base_color_factor'].append(baseColorFactor)
        scene_buffers['base_color_texture'].append(baseColorTexture)
        scene_buffers['metallic_factor'].append(metallicFactor)
        scene_buffers['metallic_texture'].append(metallicTexture)
        scene_buffers['roughness_factor'].append(roughnessFactor)
        scene_buffers['roughness_texture'].append(roughnessTexture)
        scene_buffers['emissive_factor'].append(emissiveFactor)
        scene_buffers['emissive_texture'].append(emissiveTexture)
        scene_buffers['alpha_mode'].append(alphaMode)
        scene_buffers['alpha_cutoff'].append(alphaCutoff)
        scene_buffers['alpha_factor'].append(alphaFactor)
        scene_buffers['alpha_texture'].append(alphaTexture)
        scene_buffers['normal_texture'].append(normalTexture)
        
    scene_buffers['triangles'] = torch.cat(scene_buffers['triangles'], dim=0)   # [N, 3, 3]
    scene_buffers['normals'] = torch.cat(scene_buffers['normals'], dim=0)       # [N, 3, 3]
    scene_buffers['uvs'] = torch.cat(scene_buffers['uvs'], dim=0)               # [N, 3, 2]
    scene_buffers['material_ids'] = torch.cat(scene_buffers['material_ids'], dim=0)  # [N]
            
    # Voxelize
    out_tuple = _C.textured_mesh_to_volumetric_attr_cpu(
        voxel_size,
        grid_range,
        scene_buffers["triangles"],
        scene_buffers["normals"],
        scene_buffers["uvs"],
        scene_buffers["material_ids"],
        scene_buffers["base_color_factor"],
        scene_buffers["base_color_texture"],
        [1] * len(scene_buffers["base_color_texture"]),
        [0] * len(scene_buffers["base_color_texture"]),
        scene_buffers["metallic_factor"],
        scene_buffers["metallic_texture"],
        [1] * len(scene_buffers["metallic_texture"]),
        [0] * len(scene_buffers["metallic_texture"]),
        scene_buffers["roughness_factor"],
        scene_buffers["roughness_texture"],
        [1] * len(scene_buffers["roughness_texture"]),
        [0] * len(scene_buffers["roughness_texture"]),
        scene_buffers["emissive_factor"],
        scene_buffers["emissive_texture"],
        [1] * len(scene_buffers["emissive_texture"]),
        [0] * len(scene_buffers["emissive_texture"]),
        scene_buffers["alpha_mode"],
        scene_buffers["alpha_cutoff"],
        scene_buffers["alpha_factor"],
        scene_buffers["alpha_texture"],
        [1] * len(scene_buffers["alpha_texture"]),
        [0] * len(scene_buffers["alpha_texture"]),
        scene_buffers["normal_texture"],
        [1] * len(scene_buffers["normal_texture"]),
        [0] * len(scene_buffers["normal_texture"]),
        mip_level_offset,
        timing,
        False,
    )
    
    # Post process
    coord = out_tuple[0]
    attr = {
        "base_color": torch.clamp(out_tuple[1] * 255, 0, 255).byte().reshape(-1, 3),
        "metallic": torch.clamp(out_tuple[2] * 255, 0, 255).byte().reshape(-1, 1),
        "roughness": torch.clamp(out_tuple[3] * 255, 0, 255).byte().reshape(-1, 1),
        "emissive": torch.clamp(out_tuple[4] * 255, 0, 255).byte().reshape(-1, 3),
        "alpha": torch.clamp(out_tuple[5] * 255, 0, 255).byte().reshape(-1, 1),
        "normal": torch.clamp((out_tuple[6] * 0.5 + 0.5) * 255, 0, 255).byte().reshape(-1, 3),
    }
    
    return coord, attr


def blender_dump_to_volumetric_attr(
    dump: Dict[str, Any],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    aabb: Union[list, tuple, np.ndarray, torch.Tensor] = None,
    mip_level_offset: float = 0.0,
    verbose: bool = False,
    timing: bool = False,
    add_emission: bool = True,
    color_space: str = 'agx',
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Voxelize a mesh into a sparse voxel grid with PBR properties.
    
    Args:
        dump (Dict[str, Any]): Dumped data from a blender scene.
        voxel_size (float, list, tuple, np.ndarray, torch.Tensor): The size of each voxel.
        grid_size (int, list, tuple, np.ndarray, torch.Tensor): The size of the grid.
            NOTE: One of voxel_size and grid_size must be provided.
        aabb (list, tuple, np.ndarray, torch.Tensor): The axis-aligned bounding box of the mesh.
            If not provided, it will be computed automatically.
        mip_level_offset (float): The mip level offset for texture mip level selection.
        verbose (bool): Whether to print the settings.
        timing (bool): Whether to print the timing information.
        
    Returns:
        torch.Tensor: The indices of the voxels that are occupied by the mesh.
        Dict[str, torch.Tensor]: A dictionary containing the following keys:
        - "base_color": The base color of the occupied voxels.
        - "metallic": The metallic value of the occupied voxels.
        - "roughness": The roughness value of the occupied voxels.
        - "emissive": The emissive value of the occupied voxels.
        - "alpha": The alpha value of the occupied voxels.
        - "normal": The normal of the occupied voxels.
    """        
    if color_space not in ('linear', 'srgb', 'agx'):
        raise ValueError("color_space must be one of: linear, srgb, agx")
    # Voxelize settings
    assert voxel_size is not None or grid_size is not None, "Either voxel_size or grid_size must be provided"

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
        assert voxel_size.dim() == 1, f"voxel_size must be a 1D tensor, but got {voxel_size.shape}"
        assert voxel_size.size(0) == 3, f"voxel_size must have 3 elements, but got {voxel_size.size(0)}"

    if grid_size is not None:
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32)
        assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
        assert grid_size.dim() == 1, f"grid_size must be a 1D tensor, but got {grid_size.shape}"
        assert grid_size.size(0) == 3, f"grid_size must have 3 elements, but got {grid_size.size(0)}"

    if aabb is not None:
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
        assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
        assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
        assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Auto adjust aabb
    if aabb is None:
        min_xyz = np.min([
            object['vertices'].min(axis=0)
            for object in dump['objects']
        ], axis=0)
        max_xyz = np.max([
            object['vertices'].max(axis=0)
            for object in dump['objects']
        ], axis=0)

        if voxel_size is not None:
            padding = torch.ceil((max_xyz - min_xyz) / voxel_size) * voxel_size - (max_xyz - min_xyz)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5
        if grid_size is not None:
            padding = (max_xyz - min_xyz) / (grid_size - 1)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5

        aabb = torch.stack([min_xyz, max_xyz], dim=0).float()

    # Fill voxel size or grid size
    if voxel_size is None:
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    if grid_size is None:
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    
    grid_range = torch.stack([torch.zeros_like(grid_size), grid_size], dim=0).int()
        
    # Print settings
    if verbose:
        print(f"Voxelize settings:")
        print(f"  Voxel size: {voxel_size}")
        print(f"  Grid size: {grid_size}")
        print(f"  AABB: {aabb}")
        
    # Load Scene
    scene_buffers = {
        'triangles': [],
        'normals': [],
        'uvs': [],
        'material_ids': [],
        'base_color_factor': [],
        'base_color_texture': [],
        'base_color_texture_filter': [],
        'base_color_texture_wrap': [],
        'metallic_factor': [],
        'metallic_texture': [],
        'metallic_texture_filter': [],
        'metallic_texture_wrap': [],
        'roughness_factor': [],
        'roughness_texture': [],
        'roughness_texture_filter': [],
        'roughness_texture_wrap': [],
        'emissive_factor': [],
        'emissive_texture': [],
        'emissive_texture_filter': [],
        'emissive_texture_wrap': [],
        'alpha_mode': [],
        'alpha_cutoff': [],
        'alpha_factor': [],
        'alpha_texture': [],
        'alpha_texture_filter': [],
        'alpha_texture_wrap': [],
    }

    def load_texture(pack, channels=None, premultiply_alpha=False, decode_srgb=False):
        png_bytes = pack['image']
        with Image.open(io.BytesIO(png_bytes)) as image:
            # ``dump_pbr.py`` encodes scalar sockets (roughness, metallic and
            # alpha) as L PNGs.  Converting L to RGBA makes the requested
            # alpha channel opaque (255), so retain the scalar plane in that
            # case.  RGB/RGBA material images still use the full RGBA path.
            is_scalar_png = image.mode in ('1', 'L', 'I', 'I;16', 'F')
            if is_scalar_png:
                scalar_arr = np.array(image.convert('L'), dtype=np.float32) / 255.0
            rgba_arr = np.array(image.convert('RGBA'), dtype=np.float32) / 255.0
        source_color_space = pack.get('color_space', 'sRGB')
        if is_scalar_png:
            if premultiply_alpha or decode_srgb or channels is None or not isinstance(channels, int):
                raise ValueError("Scalar PNG can only be loaded as one unencoded channel")
            # The integer is a logical source-channel selector (R/G/B/A)
            # retained by the caller; after ``extract_image`` the PNG itself
            # is already a single L plane, so any integer selector maps to it.
            arr = scalar_arr
        else:
            arr = rgba_arr
            if decode_srgb:
                if source_color_space == 'sRGB':
                    # Decode before filtering and retain float32 precision
                    # through mip generation. This avoids the former raw-sRGB
                    # sampling error and an unnecessary linear->8-bit->linear
                    # round trip.
                    arr[..., :3] = _srgb_to_linear(arr[..., :3])
                elif source_color_space != 'Non-Color':
                    raise ValueError(
                        f"Color texture must use Blender sRGB or Non-Color data, got {source_color_space!r}"
                    )
            if premultiply_alpha:
                arr[..., :3] *= arr[..., 3:4]
            if channels is not None:
                arr = arr[..., channels]
        texture = torch.tensor(arr, dtype=torch.float32)
        filter_mode = {
            'Linear': 1,
            'Closest': 0,
            'Cubic': 1,
            'Smart': 1,
        }[pack['interpolation']]
        wrap_mode = {
            'REPEAT': 0,
            'EXTEND': 1,
            'CLIP': 1,
            'MIRROR': 2,
        }[pack['extension']]
        return texture, filter_mode, wrap_mode

    for material in dump['materials']:
        baseColorFactor = torch.tensor(material['baseColorFactor'][:3], dtype=torch.float32)
        is_emission_shader = material.get('shaderType') == 'Emission'
        if material['baseColorTexture'] is not None:
            baseColorTexture, baseColorTextureFilter, baseColorTextureWrap = \
                load_texture(material['baseColorTexture'], channels=[0, 1, 2],
                             premultiply_alpha=not is_emission_shader, decode_srgb=True)
            assert baseColorTexture.shape[2] == 3, f"Base color texture must have 3 channels, but got {baseColorTexture.shape[2]}"
        else:
            baseColorTexture = torch.tensor([])
            baseColorTextureFilter = 0
            baseColorTextureWrap = 0
        if is_emission_shader and baseColorTexture.numel() > 0:
            # A pure Emission shader has no separate emission channel in the
            # dump. Apply Strength to its base-color texture without applying
            # it a second time in the C++ add-emission path.  Do not clamp:
            # Emission Strength can produce HDR values which AgX must receive
            # before final output quantization.
            baseColorTexture = baseColorTexture * float(material.get('emissionStrength', 1.0))
        scene_buffers['base_color_factor'].append(baseColorFactor)
        scene_buffers['base_color_texture'].append(baseColorTexture)
        scene_buffers['base_color_texture_filter'].append(baseColorTextureFilter)
        scene_buffers['base_color_texture_wrap'].append(baseColorTextureWrap)

        metallicFactor = material['metallicFactor']
        if material['metallicTexture'] is not None:
            metallicTexture, metallicTextureFilter, metallicTextureWrap = \
                load_texture(material['metallicTexture'], channels=2)
            assert metallicTexture.dim() == 2, f"Metallic roughness texture must have 2 dimensions, but got {metallicTexture.dim()}"
        else:
            metallicTexture = torch.tensor([])
            metallicTextureFilter = 0
            metallicTextureWrap = 0
        scene_buffers['metallic_factor'].append(metallicFactor)
        scene_buffers['metallic_texture'].append(metallicTexture)
        scene_buffers['metallic_texture_filter'].append(metallicTextureFilter)
        scene_buffers['metallic_texture_wrap'].append(metallicTextureWrap)

        roughnessFactor = material['roughnessFactor']
        if material['roughnessTexture'] is not None:
            roughnessTexture, roughnessTextureFilter, roughnessTextureWrap = \
                load_texture(material['roughnessTexture'], channels=1)
            assert roughnessTexture.dim() == 2, f"Metallic roughness texture must have 2 dimensions, but got {roughnessTexture.dim()}"
        else:
            roughnessTexture = torch.tensor([])
            roughnessTextureFilter = 0
            roughnessTextureWrap = 0
        scene_buffers['roughness_factor'].append(roughnessFactor)
        scene_buffers['roughness_texture'].append(roughnessTexture)
        scene_buffers['roughness_texture_filter'].append(roughnessTextureFilter)
        scene_buffers['roughness_texture_wrap'].append(roughnessTextureWrap)

        emissiveFactor = torch.tensor(material.get('emissiveFactor', [0.0, 0.0, 0.0])[:3], dtype=torch.float32)
        if material.get('emissiveTexture') is not None:
            emissiveTexture, emissiveTextureFilter, emissiveTextureWrap = \
                load_texture(material['emissiveTexture'], channels=[0, 1, 2], premultiply_alpha=False, decode_srgb=True)
        else:
            emissiveTexture = torch.tensor([])
            emissiveTextureFilter = 0
            emissiveTextureWrap = 0
        strength = float(material.get('emissionStrength', 1.0))
        emissiveFactor = emissiveFactor * strength
        if is_emission_shader:
            emissiveFactor = torch.zeros(3, dtype=torch.float32)
            emissiveTexture = torch.tensor([])
        scene_buffers['emissive_factor'].append(emissiveFactor)
        scene_buffers['emissive_texture'].append(emissiveTexture)
        scene_buffers['emissive_texture_filter'].append(emissiveTextureFilter)
        scene_buffers['emissive_texture_wrap'].append(emissiveTextureWrap)

        alphaMode = ALPHA_MODE_ENUM[material['alphaMode']]
        alphaCutoff = material['alphaCutoff']
        alphaFactor = material['alphaFactor']
        if material['alphaTexture'] is not None:
            alphaTexture, alphaTextureFilter, alphaTextureWrap = \
                load_texture(material['alphaTexture'], channels=3)
            assert alphaTexture.dim() == 2, f"Alpha texture must have 2 dimensions, but got {alphaTexture.dim()}"
        else:
            alphaTexture = torch.tensor([])
            alphaTextureFilter = 0
            alphaTextureWrap = 0
        scene_buffers['alpha_mode'].append(alphaMode)
        scene_buffers['alpha_cutoff'].append(alphaCutoff)
        scene_buffers['alpha_factor'].append(alphaFactor)
        scene_buffers['alpha_texture'].append(alphaTexture)
        scene_buffers['alpha_texture_filter'].append(alphaTextureFilter)
        scene_buffers['alpha_texture_wrap'].append(alphaTextureWrap)
    
    for object in dump['objects']:
        triangles = torch.tensor(object['vertices'][object['faces']], dtype=torch.float32).reshape(-1, 3, 3) - aabb[0].reshape(1, 1, 3)
        normails = torch.tensor(object['normals'], dtype=torch.float32)
        uvs = torch.tensor(object['uvs'], dtype=torch.float32) if object['uvs'] is not None else torch.zeros(triangles.shape[0], 3, 2, dtype=torch.float32)
        material_id = torch.tensor(object['mat_ids'], dtype=torch.int32)
        scene_buffers['triangles'].append(triangles)
        scene_buffers['normals'].append(normails)
        scene_buffers['uvs'].append(uvs)
        scene_buffers['material_ids'].append(material_id)
        
    scene_buffers['triangles'] = torch.cat(scene_buffers['triangles'], dim=0)   # [N, 3, 3]
    scene_buffers['normals'] = torch.cat(scene_buffers['normals'], dim=0)       # [N, 3, 3]
    scene_buffers['uvs'] = torch.cat(scene_buffers['uvs'], dim=0)               # [N, 3, 2]
    scene_buffers['material_ids'] = torch.cat(scene_buffers['material_ids'], dim=0)  # [N]

    scene_buffers['uvs'][:, :, 1] = 1 - scene_buffers['uvs'][:, :, 1]  # Flip v coordinate

    # Validate all data before entering the native implementation.  Invalid
    # geometry/material indices otherwise become unchecked pointer arithmetic
    # in C++ and can terminate the process with SIGSEGV instead of a useful
    # Python exception.
    for name in ('triangles', 'normals', 'uvs'):
        value = scene_buffers[name]
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinite values")
    if scene_buffers['material_ids'].numel():
        mids = scene_buffers['material_ids']
        if int(mids.min()) < 0 or int(mids.max()) >= len(dump['materials']):
            raise ValueError(
                f"material_ids out of range [0, {len(dump['materials'])}): "
                f"min={int(mids.min())}, max={int(mids.max())}"
            )
    if scene_buffers['triangles'].shape[0] == 0:
        raise ValueError("Dump contains no triangles")
            
    # Voxelize
    out_tuple = _C.textured_mesh_to_volumetric_attr_cpu(
        voxel_size,
        grid_range,
        scene_buffers["triangles"],
        scene_buffers["normals"],
        scene_buffers["uvs"],
        scene_buffers["material_ids"],
        scene_buffers["base_color_factor"],
        scene_buffers["base_color_texture"],
        scene_buffers["base_color_texture_filter"],
        scene_buffers["base_color_texture_wrap"],
        scene_buffers["metallic_factor"],
        scene_buffers["metallic_texture"],
        scene_buffers["metallic_texture_filter"],
        scene_buffers["metallic_texture_wrap"],
        scene_buffers["roughness_factor"],
        scene_buffers["roughness_texture"],
        scene_buffers["roughness_texture_filter"],
        scene_buffers["roughness_texture_wrap"],
        scene_buffers["emissive_factor"],
        scene_buffers["emissive_texture"],
        scene_buffers["emissive_texture_filter"],
        scene_buffers["emissive_texture_wrap"],
        scene_buffers["alpha_mode"],
        scene_buffers["alpha_cutoff"],
        scene_buffers["alpha_factor"],
        scene_buffers["alpha_texture"],
        scene_buffers["alpha_texture_filter"],
        scene_buffers["alpha_texture_wrap"],
        [torch.tensor([]) for _ in range(len(scene_buffers["base_color_texture"]))],
        [0] * len(scene_buffers["base_color_texture"]),
        [0] * len(scene_buffers["base_color_texture"]),
        mip_level_offset,
        timing,
        add_emission,
    )
    
    # Post process
    coord = out_tuple[0]
    base_color = out_tuple[1].reshape(-1, 3)
    if color_space == 'srgb':
        nonnegative = torch.clamp(base_color, min=0)
        base_color = torch.where(nonnegative <= 0.0031308, 12.92 * nonnegative,
                                 1.055 * nonnegative.pow(1 / 2.4) - 0.055)
    elif color_space == 'agx':
        # Always use the exact OCIO module/config shipped with this checkout;
        # an unrelated module with the same import name must not silently
        # change the rendered colors.
        ocio_root = Path(__file__).resolve().parents[3] / "ocioutils"
        module_path = ocio_root / "color_conversion.py"
        try:
            if not module_path.is_file():
                raise FileNotFoundError(module_path)
            spec = importlib.util.spec_from_file_location("trellis2_color_conversion", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load bundled color conversion module: {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            agx = module.agx
        except (FileNotFoundError, ImportError) as e:
            raise ImportError(
                "AgX output requires the bundled ocioutils/color_conversion.py, "
                "ocioutils/config.ocio, LUT directories, and PyOpenColorIO"
            ) from e
        base_color = torch.from_numpy(agx.apply(
            base_color.detach().cpu().numpy()[None], dst_space='AgX Base sRGB'
        )[0])
    base_color = torch.clamp(base_color * 255, 0, 255).byte()
    attr = {
        "base_color": base_color,
        "metallic": torch.clamp(out_tuple[2] * 255, 0, 255).byte().reshape(-1, 1),
        "roughness": torch.clamp(out_tuple[3] * 255, 0, 255).byte().reshape(-1, 1),
        "emissive": torch.clamp(out_tuple[4] * 255, 0, 255).byte().reshape(-1, 3),
        "alpha": torch.clamp(out_tuple[5] * 255, 0, 255).byte().reshape(-1, 1),
        "normal": torch.clamp((out_tuple[6] * 0.5 + 0.5) * 255, 0, 255).byte().reshape(-1, 3),
    }
    
    return coord, attr
