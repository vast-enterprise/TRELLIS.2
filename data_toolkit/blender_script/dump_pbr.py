import argparse, sys, os, math, io, tempfile
from typing import *
import bpy
from mathutils import Vector
import numpy as np
from PIL import Image
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_normals import clean_geometry, triangle_normals, triangle_topology, NORMAL_SOURCE


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj if bpy.app.version[0] < 4 else bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl if bpy.app.version[0] < 4 else bpy.ops.wm.stl_import,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply if bpy.app.version[0] < 4 else bpy.ops.wm.ply_import,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str, merge_vertices: bool = True) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=merge_vertices, import_shading='NORMALS', bone_heuristic='TEMPERANCE')
    else:
        import_function(filepath=object_path)
        
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    to_remove = []
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            to_remove.append(obj)
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
      

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    
    return scale, offset


# =============== NODE TREE PARSING ===============

def extract_image(tex_node, channels):
        """Extract a texture through Blender's image encoder.

        Reading ``Image.pixels`` bypasses Blender's image/color-management
        handling (and is especially wrong for packed/generated images).  Save
        as PNG first, then read the encoded bytes just like Blender's render
        path does.  PNG is intentionally kept 8-bit because the on-disk PBR
        dump format and O-Voxel sampler use byte textures.
        """
        image = tex_node.image
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        old_format, old_path = image.file_format, image.filepath_raw
        try:
            image.file_format = 'PNG'
            image.filepath_raw = path
            image.save()
            with Image.open(path) as pil_image:
                pil_image = pil_image.convert('RGBA')
                # Blender's in-memory rows are bottom-to-top.  Preserve that
                # orientation after the PNG round-trip so the existing UV
                # convention in the O-Voxel sampler remains unchanged.
                rgba = np.array(pil_image, dtype=np.uint8)[::-1].copy()
        finally:
            image.file_format, image.filepath_raw = old_format, old_path
            if os.path.exists(path):
                os.remove(path)

        data = rgba[..., channels]

        if len(data.shape) == 2:  # Single channel
            pil_image = Image.fromarray(data, mode='L')
        elif data.shape[2] == 3:
            pil_image = Image.fromarray(data, mode='RGB')
        elif data.shape[2] == 4:
            pil_image = Image.fromarray(data, mode='RGBA')
        else:
            raise ValueError("Unsupported channel shape for image")

        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        png_bytes = buffer.getvalue()

        return {
            'image': png_bytes,
            'interpolation': tex_node.interpolation,
            'extension': tex_node.extension,
            'color_space': image.colorspace_settings.name,
        }


def try_extract_image(link, expected_channel='RGB'):
    """
    Tries to extract an image from a texture node link.
    Supported sub tree modes:
      - RGB:
        TEX_IMAGE ->
      - R, G, B:
        TEX_IMAGE -> SEPARATE_COLOR ->
      - A:
        TEX_IMAGE ->
    """
    assert expected_channel in ['RGB', 'R', 'G', 'B', 'A'], "Unsupported channel"

    if expected_channel == 'RGB':
        assert link.from_node.type == 'TEX_IMAGE', "Material is not supported"
        assert link.from_socket.name == 'Color', "Material is not supported"
        tex_node = link.from_node
        # Keep alpha in the dump for correct premultiplied filtering. Callers
        # that need RGB only explicitly select channels when loading.
        return extract_image(tex_node, [0, 1, 2, 3])

    if expected_channel in ['R', 'G', 'B']:
        socket_name = {
            'R': 'Red',
            'G': 'Green',
            'B': 'Blue',
        }[expected_channel]
        assert link.from_node.type == 'SEPARATE_COLOR' and link.from_node.mode == 'RGB', \
            f"Material is not supported, {link.from_node.type}, {link.from_node.mode}"
        assert link.from_socket.name == socket_name, "Material is not supported"
        sep_node = link.from_node
        assert sep_node.inputs[0].is_linked and sep_node.inputs[0].links[0].from_node.type == 'TEX_IMAGE', \
            "Material is not supported"
        assert sep_node.inputs[0].links[0].from_socket.name == 'Color', "Material is not supported"
        tex_node = sep_node.inputs[0].links[0].from_node
        channel_index = {
            'R': 0,
            'G': 1,
            'B': 2,
        }[expected_channel]
        return extract_image(tex_node, channel_index)

    if expected_channel == 'A':
        assert link.from_node.type == 'TEX_IMAGE', "Material is not supported"
        assert link.from_socket.name == 'Alpha', "Material is not supported"
        tex_node = link.from_node
        return extract_image(tex_node, 3)


def try_extract_factor(link, mode='color'):
    """
    Tries to extract a factor from a math node link.
    Supported sub tree modes:
      - color:
       ANY -> MIX(MULTIPLY) ->
      - scalar:
       ANY -> MATH(MULTIPLY) ->
    """
    assert mode in ['color','scalar'], "Unsupported mode"

    if mode == 'color':
        if link.from_node.type == 'MIX':
            mix_node = link.from_node
            assert mix_node.data_type == 'RGBA' and mix_node.blend_type == 'MULTIPLY', f"Material is not supported, {mix_node.data_type}, {mix_node.blend_type}"
            assert not mix_node.inputs['Factor'].is_linked and mix_node.inputs['Factor'].default_value == 1.0, \
                "Material is not supported"
            if mix_node.inputs['A'].is_linked:
                assert not mix_node.inputs['B'].is_linked, "Material is not supported"
                return (list(mix_node.inputs['B'].default_value)[:3], mix_node.inputs['A'].links[0])
            else:
                assert not mix_node.inputs['A'].is_linked, "Material is not supported"
                assert mix_node.inputs['B'].is_linked, "Material is not supported"
                return (list(mix_node.inputs['A'].default_value)[:3], mix_node.inputs['B'].links[0])
        return ([1.0, 1.0, 1.0], link)

    if mode =='scalar':
        if link.from_node.type == 'MATH':
            math_node = link.from_node
            assert math_node.operation == 'MULTIPLY', "Material is not supported"
            assert math_node.inputs[0].is_linked, "Material is not supported"
            assert not math_node.inputs[1].is_linked, "Material is not supported"
            return (math_node.inputs[1].default_value, math_node.inputs[0].links[0])
        return (1.0, link)


def try_extract_image_with_factor(link, expected_channel='RGB'):
    """
    Tries to extract an image and a factor from a texture node link.
    """
    factor, link = try_extract_factor(link, 'color' if expected_channel in ['RGB'] else 'scalar')
    image = try_extract_image(link, expected_channel)
    return (factor, image)


def _is_vertex_color_node(node):
    """Return whether *node* reads the glTF per-corner color attribute.

    Blender's glTF importer has used both ``Vertex Color`` and generic
    ``Attribute`` nodes over its supported versions.  Keep this check narrow
    enough that unrelated attributes are not silently treated as COLOR_0.
    """
    if node.type == 'VERTEX_COLOR':
        return True
    if node.type == 'ATTRIBUTE':
        name = getattr(node, 'attribute_name', '') or ''
        return name.upper() in {'COLOR_0', 'COLOR', 'COL'}
    return False


def _vertex_color_layer_name(node):
    """Return an explicitly requested color attribute layer, when present."""
    if node.type == 'ATTRIBUTE':
        value = getattr(node, 'attribute_name', '') or ''
    elif node.type == 'VERTEX_COLOR':
        value = getattr(node, 'layer_name', '') or ''
    else:
        value = ''
    value = str(value).strip()
    return value or None


def _base_color_source(link):
    """Extract constants, image texture and optional COLOR_0 from a color graph.

    The existing dump format stores material constants and image textures
    separately.  A vertex-color input is an additional per-corner multiplier,
    so it is returned as a flag and is sampled by the voxelizer.  This handles
    the common glTF graphs (direct image/color, and multiply image * vertex
    color) while leaving unsupported graphs on the historical error path.
    """
    def visit(current, seen):
        node = current.from_node
        if id(node) in seen:
            raise AssertionError("Material color graph contains a cycle")
        seen = seen | {id(node)}
        if _is_vertex_color_node(node):
            return [1.0, 1.0, 1.0], None, True, _vertex_color_layer_name(node)
        if node.type == 'TEX_IMAGE' and current.from_socket.name == 'Color':
            return [1.0, 1.0, 1.0], extract_image(node, [0, 1, 2, 3]), False, None
        if node.type == 'RGB':
            return list(node.outputs[0].default_value)[:3], None, False, None
        if node.type in {'MIX', 'MIX_RGB'}:
            if node.type == 'MIX':
                data_type = getattr(node, 'data_type', None)
                blend_type = getattr(node, 'blend_type', None)
                fac = node.inputs.get('Factor')
                a = node.inputs.get('A')
                b = node.inputs.get('B')
            else:
                data_type = 'RGBA'
                blend_type = getattr(node, 'blend_type', None)
                fac = node.inputs.get('Fac')
                a = node.inputs.get('Color1')
                b = node.inputs.get('Color2')
            if data_type not in (None, 'RGBA') or blend_type != 'MULTIPLY':
                raise AssertionError("Material color graph is not a multiply")
            if fac is not None and fac.is_linked:
                raise AssertionError("Material color graph has a linked mix factor")
            if fac is not None and abs(float(fac.default_value) - 1.0) > 1e-6:
                raise AssertionError("Material color graph has a partial mix factor")
            if a is None or b is None:
                raise AssertionError("Material color graph has incomplete multiply inputs")
            result = [1.0, 1.0, 1.0], None, False, None
            for socket in (a, b):
                if socket.is_linked:
                    part = visit(socket.links[0], seen)
                else:
                    value = list(socket.default_value)
                    part = (value[:3], None, False, None)
                if result[1] is not None and part[1] is not None:
                    raise AssertionError("Material color graph has multiple image textures")
                if result[2] and part[2]:
                    raise AssertionError("Material color graph has multiple vertex colors")
                if result[3] and part[3] and result[3].upper() != part[3].upper():
                    raise AssertionError("Material color graph has multiple vertex color layers")
                result = ([x * y for x, y in zip(result[0], part[0])],
                          result[1] or part[1], result[2] or part[2],
                          result[3] or part[3])
            return result
        if node.type == 'SEPRGB' and current.from_socket.name in {'Image', 'Color'}:
            if not node.inputs[0].is_linked:
                raise AssertionError("Material color graph has incomplete separator")
            return visit(node.inputs[0].links[0], seen)
        raise AssertionError(f"Unsupported material color node: {node.type}")

    return visit(link, set())


def ensure_default_materials():
    """Give mesh faces without a material Blender's default Principled BSDF."""
    default_material = None

    def get_default_material():
        nonlocal default_material
        if default_material is None:
            default_material = bpy.data.materials.new(name='DefaultPBR')
            default_material.use_nodes = True
        return default_material

    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        if not obj.material_slots:
            obj.data.materials.append(get_default_material())
        for slot in obj.material_slots:
            if slot.material is None:
                slot.material = get_default_material()

        # Invalid face indices can also occur in imported meshes. Preserve
        # valid assignments and send only unassigned faces to the fallback.
        missing_faces = [face for face in obj.data.polygons
                         if face.material_index >= len(obj.material_slots)]
        if missing_faces:
            fallback = get_default_material()
            fallback_index = next((i for i, slot in enumerate(obj.material_slots)
                                   if slot.material == fallback), None)
            if fallback_index is None:
                fallback_index = len(obj.material_slots)
                obj.data.materials.append(fallback)
            for face in missing_faces:
                face.material_index = fallback_index


def main(arg):
    arg.geometry_only = getattr(arg, "geometry_only", False)
    if arg.geometry_only:
        arg.renormal = False
        arg.merge_distance = 0.0
    # Initialize context
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object, merge_vertices=not arg.geometry_only)
    print('[INFO] Scene initialized.')

    # Material IDs are indices into bpy.data.materials, so create fallbacks
    # before serializing that collection or evaluating the mesh. Geometry-only
    # mode intentionally skips all material/normal/UV work for benchmarking.
    if not arg.geometry_only:
        ensure_default_materials()
    
    # Normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    # Start dumping
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    output = {
        'materials': [],
        'objects': [],
        'geometry_only': arg.geometry_only,
        # File-level provenance prevents geometry/authored caches being mixed.
        'surface_normal_source': NORMAL_SOURCE if arg.renormal else 'blender_authored_corner_world_v1',
        'normal_preprocessing': {'renormal': arg.renormal, 'merge_distance': arg.merge_distance,
                                 'max_edge_faces': arg.max_edge_faces,
                                 'algorithm': 'manifold_patches_v2'},
    }

    # Dumping materials
    for mat in ([] if arg.geometry_only else bpy.data.materials):
        assert mat.use_nodes == True, "Material is not supported"

        pack = {
            "baseColorFactor": [1.0, 1.0, 1.0],
            "alphaFactor": 1.0,
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
            "alphaMode": "OPAQUE",
            "alphaCutoff": 0.5,
            "baseColorTexture": None,
            "baseColorUseVertexColor": False,
            "alphaTexture": None,
            "metallicTexture": None,
            "roughnessTexture": None,
            "emissiveFactor": [0.0, 0.0, 0.0],
            "emissiveTexture": None,
            "emissionStrength": 0.0,
            "shaderType": "Principled",
        }

        try:
            principled_node = mat.node_tree.nodes.get('Principled BSDF')
            if principled_node is None:
                principled_node = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            emission_node = mat.node_tree.nodes.get('Emission')
            if emission_node is None:
                emission_node = next((n for n in mat.node_tree.nodes if n.type == 'EMISSION'), None)
            if principled_node is None and emission_node is None:
                raise RuntimeError("Material is not supported: no Principled or Emission shader")

            # Unshaded materials are represented as a pure emission shader.
            if principled_node is None:
                pack["shaderType"] = "Emission"
                pack["metallicFactor"] = 0.0
                pack["roughnessFactor"] = 1.0
                has_transparent_shader = any(n.type == 'BSDF_TRANSPARENT' for n in mat.node_tree.nodes)
                if has_transparent_shader:
                    pack["alphaMode"] = "BLEND"
                if emission_node.inputs['Color'].is_linked:
                    link = emission_node.inputs['Color'].links[0]
                    factor, image, uses_vertex_color, vertex_color_layer = _base_color_source(link)
                    pack['baseColorFactor'] = factor
                    pack['baseColorTexture'] = image
                    pack['baseColorUseVertexColor'] = uses_vertex_color
                    pack['baseColorVertexColorLayer'] = vertex_color_layer
                else:
                    pack['baseColorFactor'] = list(emission_node.inputs['Color'].default_value)
                pack['emissionStrength'] = float(emission_node.inputs['Strength'].default_value)
                # For a constant emission color bake strength into the
                # factor.  Texture-backed emission is scaled when loaded so
                # filtering happens on the original texels first.
                if pack['baseColorTexture'] is None:
                    pack['baseColorFactor'] = [float(v) * pack['emissionStrength'] for v in pack['baseColorFactor'][:3]] + [pack['baseColorFactor'][3] if len(pack['baseColorFactor']) > 3 else 1.0]
                elif has_transparent_shader:
                    # glTF unshaded alpha commonly arrives as an Emission +
                    # Transparent node graph.  The RGB dump retains alpha,
                    # so reuse it as the material alpha source.
                    pack['alphaTexture'] = pack['baseColorTexture']
                # The emission shader's color is already the final material
                # color; do not add it a second time in the voxelizer.
                pack['emissiveFactor'] = [0.0, 0.0, 0.0]
                principled_node = None

            # Handle base color
            if principled_node is not None and not principled_node.inputs['Base Color'].is_linked:
                pack["baseColorFactor"] = list(principled_node.inputs['Base Color'].default_value)
            elif principled_node is not None:
                link = principled_node.inputs['Base Color'].links[0]
                factor, image, uses_vertex_color, vertex_color_layer = _base_color_source(link)
                pack["baseColorFactor"] = factor
                pack["baseColorTexture"] = image
                pack["baseColorUseVertexColor"] = uses_vertex_color
                pack["baseColorVertexColorLayer"] = vertex_color_layer

            # Handle emission on Principled BSDF.  Blender 4.x exposes
            # ``Emission Color`` and ``Emission Strength``; older files may
            # still use ``Emission``.
            if principled_node is not None:
                emission_color_socket = principled_node.inputs.get('Emission Color')
                if emission_color_socket is None:
                    emission_color_socket = principled_node.inputs.get('Emission')
                strength_socket = principled_node.inputs.get('Emission Strength')
                if emission_color_socket is not None:
                    if emission_color_socket.is_linked:
                        link = emission_color_socket.links[0]
                        if link.from_node.type == 'RGB':
                            pack['emissiveFactor'] = list(link.from_node.outputs[0].default_value)[:3]
                        else:
                            factor, image = try_extract_image_with_factor(link, 'RGB')
                            pack['emissiveFactor'] = factor
                            pack['emissiveTexture'] = image
                    else:
                        pack['emissiveFactor'] = list(emission_color_socket.default_value)[:3]
                if strength_socket is not None:
                    pack['emissionStrength'] = float(strength_socket.default_value)

            # Handle alpha.  Pure emission materials are opaque unless a
            # transparent shader is explicitly used (their alpha is not
            # represented by a Principled socket).
            if principled_node is None:
                pack["alphaFactor"] = 1.0
            elif not principled_node.inputs['Alpha'].is_linked:
                pack["alphaFactor"] = principled_node.inputs['Alpha'].default_value
                if pack["alphaFactor"] < 1.0:
                    pack["alphaMode"] = "BLEND"
            else:
                link = principled_node.inputs['Alpha'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["alphaFactor"] = node.outputs[0].default_value
                    if pack["alphaFactor"] < 1.0:
                        pack["alphaMode"] = "BLEND"
                else:
                    pack["alphaMode"] = "BLEND"
                    if node.type == 'MATH':
                        if node.operation == 'ROUND':
                            assert node.inputs[0].is_linked, "Material is not supported"
                            pack["alphaMode"] = "MASK"
                            link = node.inputs[0].links[0]
                        elif node.operation == 'SUBTRACT':
                            assert node.inputs[0].default_value == 1.0 and \
                                node.inputs[1].is_linked and \
                                node.inputs[1].links[0].from_node.type == 'MATH' and \
                                node.inputs[1].links[0].from_node.operation == 'LESS_THAN', \
                                "Material is not supported"
                            assert node.inputs[1].links[0].from_node.inputs[0].is_linked, "Material is not supported"
                            pack["alphaMode"] = "MASK"
                            pack["alphaCutoff"] = node.inputs[1].links[0].from_node.inputs[1].default_value
                            link = node.inputs[1].links[0].from_node.inputs[0].links[0]
                    factor, image = try_extract_image_with_factor(link, 'A')
                    pack["alphaFactor"] = factor
                    pack["alphaTexture"] = image

            # Handle metallic/roughness (fixed defaults for unshaded).
            if principled_node is None:
                pack["metallicFactor"] = 0.0
                pack["roughnessFactor"] = 1.0
            elif not principled_node.inputs['Metallic'].is_linked:
                pack["metallicFactor"] = principled_node.inputs['Metallic'].default_value
            else:
                link = principled_node.inputs['Metallic'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["metallicFactor"] = node.outputs[0].default_value
                else:
                    factor, image = try_extract_image_with_factor(link, 'B')
                    pack["metallicFactor"] = factor
                    pack["metallicTexture"] = image

            # Handle roughness
            if principled_node is not None and not principled_node.inputs['Roughness'].is_linked:
                pack["roughnessFactor"] = principled_node.inputs['Roughness'].default_value
            elif principled_node is not None:
                link = principled_node.inputs['Roughness'].links[0]
                node = link.from_node
                if node.type == 'VALUE':
                    pack["roughnessFactor"] = node.outputs[0].default_value
                else:
                    factor, image = try_extract_image_with_factor(link, 'G')
                    pack["roughnessFactor"] = factor
                    pack["roughnessTexture"] = image

            output['materials'].append(pack)
        except:
            with open(arg.output_path + '_error.txt', 'w') as f:
                f.write(str([[n.name] for n in mat.node_tree.nodes]))
            raise RuntimeError("Material is not supported")

    # Dumping meshes
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        
        pack = {
            "vertices": None,
            "faces": None,
            "uvs": None,
            "matIDs": None,
        }
        
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        try:
            world_matrix = eval_obj.matrix_world.copy()
            if arg.renormal and not arg.geometry_only:
                stats = clean_geometry(eval_mesh, world_matrix, arg.merge_distance, arg.max_edge_faces)
                print(f"[INFO] Re-normal {obj.name}: {stats}", flush=True)
                pack['normal_preprocessing'] = stats
            eval_mesh.calc_loop_triangles()
            # foreach_get avoids millions of Python RNA wrapper allocations.
            triangles = eval_mesh.loop_triangles
            pack["vertices"] = np.empty((len(eval_mesh.vertices), 3), dtype=np.float32)
            eval_mesh.vertices.foreach_get('co', pack['vertices'].ravel())
            if not arg.renormal:
                matrix = np.asarray(world_matrix, dtype=np.float32)
                pack['vertices'] = pack['vertices'] @ matrix[:3, :3].T + matrix[:3, 3]
            pack["faces"] = np.empty((len(triangles), 3), dtype=np.int32)
            triangles.foreach_get('vertices', pack['faces'].ravel())
            if arg.geometry_only:
                output['objects'].append(pack)
                continue
            loops = np.empty((len(triangles), 3), dtype=np.int32)
            triangles.foreach_get('loops', loops.ravel())
            if arg.renormal:
                pack["normals"] = triangle_normals(pack["vertices"], pack["faces"])
                pack['topology'] = triangle_topology(eval_mesh, faces=pack['faces'])
            else:
                if hasattr(eval_mesh, 'calc_normals_split'):
                    eval_mesh.calc_normals_split()
                normals = np.empty((len(eval_mesh.loops), 3), dtype=np.float32)
                if hasattr(eval_mesh, 'corner_normals'):
                    eval_mesh.corner_normals.foreach_get('vector', normals.ravel())
                else:
                    eval_mesh.loops.foreach_get('normal', normals.ravel())
                matrix = np.asarray(world_matrix.to_3x3().inverted_safe().transposed(), dtype=np.float32)
                normals = normals @ matrix.T
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                np.divide(normals, lengths, out=normals, where=lengths > 1e-10)
                pack['normals'] = normals[loops]

            if eval_mesh.uv_layers.active is not None:
                uv_data = eval_mesh.uv_layers.active.data
                uvs = np.empty((len(uv_data), 2), dtype=np.float32)
                uv_data.foreach_get('uv', uvs.ravel())
                pack["uvs"] = uvs[loops]

            slot_ids = np.asarray([
                bpy.data.materials.find(slot.name) if slot.material is not None else -1
                for slot in obj.material_slots
            ], dtype=np.int32)
            material_ids = np.empty(len(triangles), dtype=np.int32)
            triangles.foreach_get('material_index', material_ids)
            pack['mat_ids'] = np.full(len(triangles), -1, dtype=np.int32)
            valid = (material_ids >= 0) & (material_ids < len(slot_ids))
            pack['mat_ids'][valid] = slot_ids[material_ids[valid]]

            # Preserve glTF COLOR_0 per-corner data.  Blender exposes imported
            # vertex colors through color_attributes; both CORNER (the usual
            # glTF representation) and POINT domains are supported.  Store
            # RGBA float values in triangle-loop order so the voxelizer can
            # interpolate them at the same surface query as UV/base color.
            object_material_ids = np.unique(pack['mat_ids'])
            object_requires_color = any(
                0 <= value < len(output['materials']) and
                output['materials'][value].get('baseColorUseVertexColor', False)
                for value in object_material_ids
            )
            if object_requires_color:
                color_attributes = list(getattr(eval_mesh, 'color_attributes', []))
                layers_by_name = {layer.name.upper(): layer for layer in color_attributes}
                semantic_layer = next((layer for layer in color_attributes
                                       if layer.name.upper() in {'COLOR_0', 'COLOR', 'COL'}), None)
                colors = np.ones((len(triangles), 3, 4), dtype=np.float32)
                layer_cache = {}
                for material_index in np.unique(pack['mat_ids']):
                    material = (output['materials'][material_index]
                                if 0 <= material_index < len(output['materials']) else None)
                    if material is None or not material.get('baseColorUseVertexColor', False):
                        continue
                    requested = material.get('baseColorVertexColorLayer')
                    color_layer = (layers_by_name.get(str(requested).upper())
                                   if requested else semantic_layer)
                    if color_layer is None and requested and str(requested).upper() in {'COLOR_0', 'COLOR', 'COL'}:
                        color_layer = semantic_layer
                    if color_layer is None:
                        raise RuntimeError(f'Vertex color layer {requested!r} is missing from mesh')
                    if color_layer.name not in layer_cache:
                        values = np.empty((len(color_layer.data), 4), dtype=np.float32)
                        color_layer.data.foreach_get('color', values.ravel())
                        layer_cache[color_layer.name] = values
                    mask = pack['mat_ids'] == material_index
                    indices = loops[mask] if color_layer.domain == 'CORNER' else pack['faces'][mask]
                    colors[mask] = layer_cache[color_layer.name][indices]
                pack['vertex_colors'] = colors
            else:
                pack['vertex_colors'] = None

            output['objects'].append(pack)
        finally:
            eval_obj.to_mesh_clear()

    # Save output
    os.makedirs(os.path.dirname(arg.output_path), exist_ok=True)
    with open(arg.output_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('[INFO] Output saved to {}.'.format(arg.output_path))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--geometry-only', action='store_true',
                        help='Export only normalized vertices/faces; skip weld, normals, UVs and materials')
    parser.add_argument('--renormal', action=argparse.BooleanOptionalAction, default=True,
                        help='Repair winding and export geometry normals (default: enabled)')
    parser.add_argument('--merge-distance', type=float, default=1e-7,
                        help='World-space weld distance after scene normalization; 0 disables welding')
    parser.add_argument('--max-edge-faces', type=int, default=0,
                        help='0 allows multi-face junctions; 2 is strict manifold; 3 allows three-face edges')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_path', type=str, default='/tmp', help='The path the output will be dumped to.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    if args.geometry_only:
        args.renormal = False
        args.merge_distance = 0.0

    # Failed runs must not leave a previous dump that a caller could reuse.
    if os.path.exists(args.output_path):
        os.unlink(args.output_path)
    try:
        main(args)
    except Exception as error:
        if os.path.exists(args.output_path):
            os.unlink(args.output_path)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path + '_error.txt', 'w') as stream:
            stream.write(str(error))
        raise
