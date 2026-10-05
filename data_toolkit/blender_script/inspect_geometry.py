"""Blender diagnostic: preserve an asset and mark edges by adjacent-face count.

blender -b --python-exit-code 1 -P data_toolkit/blender_script/inspect_geometry.py \
  -- --object model.glb --output-dir inspection

This does not run the rejection filter. The original imported mesh and a
separate post-weld diagnostic snapshot are retained for visual comparison.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_pbr import init_scene, load_object, normalize_scene
from geometry_normals import edge_statistics, clean_geometry


def collection(name, hidden=False):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    c.hide_viewport = hidden
    c.hide_render = hidden
    return c


def markers(name, edges, color, target):
    if not edges:
        return
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = 0.0008
    curve.bevel_resolution = 0
    for endpoints in edges:
        poly = curve.splines.new('POLY')
        poly.points.add(1)
        for point, xyz in zip(poly.points, endpoints):
            point.co = (*xyz, 1)
    obj = bpy.data.objects.new(name, curve)
    target.objects.link(obj)
    obj.color = (*color, 1)
    obj.show_in_front = True
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1)
    curve.materials.append(material)


def snapshot(bm):
    stats = edge_statistics(bm)
    problem = [e for e in bm.edges if len(e.link_faces) > 2]
    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    affected = {f for e in problem for f in e.link_faces}
    area = sum(f.calc_area() for f in bm.faces)
    stats.update(vertices=len(bm.verts), faces=len(bm.faces),
                 affected_faces=len(affected),
                 affected_area_fraction=sum(f.calc_area() for f in affected) / max(area, 1e-30),
                 max_edge_face_degree=max((len(e.link_faces) for e in bm.edges), default=0))
    stats['multi_face_segments'] = [[list(v.co) for v in e.verts] for e in problem]
    stats['boundary_segments'] = [[list(v.co) for v in e.verts] for e in boundary]
    return stats


def main(args):
    init_scene()
    load_object(str(args.object))
    normalize_scene()
    originals = list(o for o in bpy.context.scene.objects if o.type == 'MESH')
    red = collection('DIAGNOSTIC_01_RED_post_weld_edges_with_MORE_THAN_TWO_faces')
    pre_red = collection('DIAGNOSTIC_04_RED_before_weld_comparison', hidden=True)
    cyan = collection('DIAGNOSTIC_02_CYAN_open_boundary_ALLOWED', hidden=True)
    welded = collection('DIAGNOSTIC_03_POST_WELD_snapshot_toggle_for_comparison', hidden=True)
    report = {'asset': args.object.name, 'merge_distance': args.merge_distance,
              'boundary_policy': 'One adjacent face is an allowed open boundary.',
              'objects': []}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in originals:
        obj.color = (0.58, 0.62, 0.68, 1)
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.transform(evaluated.matrix_world)
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            before = snapshot(bm)
            markers(obj.name + '_RED_multi_face', before['multi_face_segments'], (1,.06,.025), pre_red)
            markers(obj.name + '_CYAN_boundary_ALLOWED', before['boundary_segments'], (.02,.75,1), cyan)
            # Match production's face-preserving weld and patch orientation.
            # If cleanup fails, still save the original for manual diagnosis.
            cleanup_error = None
            try:
                cleanup_stats = clean_geometry(mesh, evaluated.matrix_world, args.merge_distance)
                bm.clear()
                bm.from_mesh(mesh)
            except ValueError as error:
                cleanup_error = str(error)
                cleanup_stats = None
            after = snapshot(bm)
            markers(obj.name + '_post_weld_RED_overlay', after['multi_face_segments'], (1,.06,.025), red)
            after_mesh = bpy.data.meshes.new(obj.name + '_post_weld')
            bm.to_mesh(after_mesh)
            after_obj = bpy.data.objects.new(after_mesh.name, after_mesh)
            welded.objects.link(after_obj)
            after_obj.color = (.58,.62,.68,1)
            markers(obj.name + '_post_weld_RED', after['multi_face_segments'], (1,.06,.025), welded)
            record = {'name': obj.name, 'before_weld': before, 'after_weld': after,
                      'cleanup_stats': cleanup_stats, 'cleanup_error': cleanup_error}
            report['objects'].append(record)
            # Geometry for a standalone, reproducible projection preview.
            bm.verts.index_update()
            record['preview_vertices'] = [list(v.co) for v in bm.verts]
            record['preview_triangles'] = [[v.index for v in f.verts] for f in bm.faces]
        finally:
            bm.free()
            evaluated.to_mesh_clear()

    readme = bpy.data.texts.new('READ_ME__EDGE_INSPECTION')
    readme.write('Original imported meshes and materials are preserved; scene is normalized.\n'
                 'RED = post-cleanup edge has >2 adjacent faces; allowed by default.\n'
                 'Use --max-edge-faces 2 in the converter for strict filtering.\n'
                 'Before-weld RED comparison collection is hidden initially.\n'
                 'CYAN = open boundary with exactly 1 adjacent face; VALID and NOT rejected.\n'
                 'Cyan collection is hidden initially; enable it in the Outliner.\n'
                 'POST_WELD collection is a hidden comparison snapshot, not repaired training data.\n'
                 'Hide original meshes before enabling POST_WELD to avoid overlaps.\n'
                 'Red curves draw in front so occluded problem edges remain visible.\n'
                 'Use Material Preview to inspect original colors.\n')
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.shading.type = 'SOLID'
                space.shading.color_type = 'OBJECT'
                space.region_3d.view_distance = 1.8
                space.region_3d.view_location = (0,0,0)
                space.region_3d.view_rotation = Euler((1.15,0,0.7)).to_quaternion()
                space.clip_start = 0.001
    bpy.ops.object.select_all(action='DESELECT')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'edge_report.json').write_text(json.dumps(report, indent=2))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_dir / (args.object.stem + '.inspection.blend')))
    for row in report['objects']:
        print(row['name'], {phase: {k:v for k,v in row[phase].items() if not k.endswith('segments')}
                           for phase in ('before_weld', 'after_weld')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--merge-distance', type=float, default=1e-7)
    main(parser.parse_args(sys.argv[sys.argv.index('--')+1:]))
