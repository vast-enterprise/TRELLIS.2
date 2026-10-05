"""Fast geometry-only re-normalization of an evaluated Blender mesh.

Operate on a temporary world-space mesh, keeping UVs on face corners.
Open boundaries and multi-face junctions are valid surface topology.
"""
import math
import time

import bmesh
import numpy as np

NORMAL_SOURCE = 'blender_geometry_reoriented_face_world_v1'


def edge_statistics(bm):
    """Classify boundaries separately from multi-face junctions."""
    degrees = [len(e.link_faces) for e in bm.edges]
    return {
        'edges': len(degrees),
        'boundary_edges': sum(d == 1 for d in degrees),
        'wire_edges': sum(d == 0 for d in degrees),
        'two_face_edges': sum(d == 2 for d in degrees),
        'multi_face_edges': sum(d > 2 for d in degrees),
    }


def orient_manifold_patches(bm):
    """Repair relative winding without propagating across multi-face edges.

    On an open patch choose the orientation that changes the least original
    area. Closed patches use signed volume. No global inside/outside is
    invented for a branched surface.
    """
    parity = {}
    for seed in bm.faces:
        if seed in parity:
            continue
        parity[seed] = False
        stack, patch = [seed], []
        closed = True
        while stack:
            face = stack.pop()
            patch.append(face)
            for edge in face.edges:
                if len(edge.link_faces) != 2:
                    closed = False
                    continue
                other = next(f for f in edge.link_faces if f != face)
                flip = parity[face] ^ (not edge.is_contiguous)
                if other in parity:
                    if parity[other] != flip:
                        raise ValueError('discard asset: non-orientable two-face patch')
                else:
                    parity[other] = flip
                    stack.append(other)
        if closed:
            volume = sum(((-1 if parity[f] else 1) *
                          f.verts[0].co.dot(f.verts[1].co.cross(f.verts[2].co))) for f in patch)
            invert = volume < 0
        else:
            flipped_area = sum(f.calc_area() for f in patch if parity[f])
            invert = flipped_area > sum(f.calc_area() for f in patch) * .5
        for face in patch:
            if parity[face] ^ invert:
                face.normal_flip()


def triangle_topology(mesh):
    """Bit 0: source triangle touches an edge with >2 incident faces.

    This labels the whole contributing triangle, not a geometric distance
    band around its junction edge. It is provenance, not a validity score.
    """
    edge_count = {}
    for t in mesh.loop_triangles:
        a,b,c = t.vertices
        for x,y in ((a,b),(b,c),(c,a)):
            key = (min(x,y),max(x,y))
            edge_count[key] = edge_count.get(key,0) + 1
    return np.asarray([any(edge_count[(min(x,y),max(x,y))] > 2
                           for x,y in zip(t.vertices, (*t.vertices[1:],t.vertices[0])))
                       for t in mesh.loop_triangles], dtype=np.uint8)


def clean_geometry(mesh, world_matrix, merge_distance=1e-7, max_edge_faces=0):
    if not math.isfinite(merge_distance) or merge_distance < 0:
        raise ValueError('merge_distance must be finite and non-negative')
    if max_edge_faces < 0 or max_edge_faces == 1:
        raise ValueError('max_edge_faces must be 0 (unlimited) or >=2')
    start = time.perf_counter()
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.transform(world_matrix)
        if any(not all(math.isfinite(x) for x in v.co) for v in bm.verts):
            raise ValueError('discard asset: non-finite world-space vertices')
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        before = len(bm.verts)
        skipped_weld_vertices = 0
        # Do not weld across objects or use a voxel-sized threshold: thin
        # shells and UV/material seams must survive. BMesh keeps loop data.
        if merge_distance:
            targetmap = bmesh.ops.find_doubles(
                bm, verts=list(bm.verts), dist=merge_distance)['targetmap']
            # weld_verts/remove_doubles may silently remove duplicate faces.
            # Detect those using the exact weld map before mutating topology.
            bm.verts.index_update()
            seen = {}
            blocked = set()
            for face in bm.faces:
                key = tuple(sorted(targetmap.get(v, v).index for v in face.verts))
                if len(set(key)) < 3 or face.calc_area() <= 1e-16:
                    continue
                if key in seen:
                    blocked.update(key)
                seen[key] = face
            # Preserve coincident surfaces with different materials/normals.
            # Skip only weld groups that would collapse one onto another.
            safe_map = {v:t for v,t in targetmap.items() if t.index not in blocked}
            skipped_weld_vertices = len(targetmap) - len(safe_map)
            targetmap = safe_map
            bmesh.ops.weld_verts(bm, targetmap=targetmap)
        merged = before - len(bm.verts)
        # Only numerical degeneracies, not valid thin/small triangles.
        bad = [f for f in bm.faces if f.calc_area() <= 1e-16]
        if bad:
            bmesh.ops.delete(bm, geom=bad, context='FACES_ONLY')
        if not bm.faces:
            raise ValueError('discard asset: no nondegenerate faces')
        bm.verts.index_update()
        seen = set()
        for f in bm.faces:
            key = tuple(sorted(v.index for v in f.verts))
            if key in seen:
                raise ValueError('discard asset: duplicate/coincident triangles after welding')
            seen.add(key)
        edge_stats = edge_statistics(bm)
        max_degree = max((len(e.link_faces) for e in bm.edges), default=0)
        if max_edge_faces and max_degree > max_edge_faces:
            raise ValueError(
                f"discard asset: edge degree {max_degree} exceeds max_edge_faces={max_edge_faces}; "
                f"{edge_stats['boundary_edges']} open boundary edges are allowed")
        # Compiled Blender operator repairs relative winding and closed-shell
        # orientation. Open components have no guaranteed inside/outside.
        if edge_stats['multi_face_edges'] or edge_stats['boundary_edges']:
            orient_manifold_patches(bm)
        else:
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.normal_update()
        for e in bm.edges:
            if len(e.link_faces) == 2 and not e.is_contiguous:
                raise ValueError('discard asset: unresolved same-direction half-edge')
        # Keep both winding and face-corner UVs from this exact topology.
        bm.to_mesh(mesh)
        mesh.update()
        mesh.calc_loop_triangles()
        return {
            'merged_vertices': merged, 'removed_degenerate_faces': len(bad),
            'faces': len(bm.faces), 'merge_distance': merge_distance,
            'max_edge_face_degree': max_degree, 'max_edge_faces': max_edge_faces,
            'skipped_weld_vertices': skipped_weld_vertices,
            **edge_stats,
            'seconds': time.perf_counter() - start,
        }
    finally:
        bm.free()


def triangle_normals(vertices, faces):
    """World-space winding normals, independent of custom/split normals."""
    tri = np.asarray(vertices, dtype=np.float64)[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths <= 1e-20):
        raise ValueError('discard asset: invalid geometric triangle normal')
    return np.repeat((normals / lengths[:, None])[:, None, :], 3, axis=1).astype(np.float32)
