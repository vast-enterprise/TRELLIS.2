# VXZM versions 0, 1 and 2

VXZM stores multi-surface voxel material samples.  Unlike VXZ, two or more
records may have the same fine-grid coordinate when their surface normals are
in different clusters.  VXZ remains unchanged and is not decoded by this
reader.

## Container

- bytes 0..3: ASCII `VXZM`
- byte 4: version (`0`, `1` or `2`); must agree with the JSON header
- bytes 5..8: big-endian uint32 JSON-end offset
- bytes 9..offset: UTF-8 JSON header
- remaining bytes: independently compressed binary sections

The header records `grid_size`, `region_resolution`, `region_block_size`,
record counts/layout, compression, and every section's relative offset and
compressed length. Normal semantics are versioned:

| Version | `normal_source` | Attributes |
|---|---|---|
| 0 (legacy) | `surface_authored_with_winding_guard` | RGB + normal, optional PBR channels |
| 1 (legacy canonical) | `geometry_reoriented_face_v1` | `base_color[3]`, `normal[3]`, `confidence[1]`, optional `alpha[1]` |
| 2 (current) | `geometry_reoriented_face_v1` | RGB + normal + confidence + `topology[1]`, optional `alpha[1]` |

Readers support all three and reject mismatched source/version/schema. For backward
compatibility the generic `write_vxzm` API defaults to legacy v0; callers writing
canonical geometry normals **must** pass
`normal_source="geometry_reoriented_face_v1"`. The writer chooses v2 when the topology attribute is present, otherwise v1.
Current conversion scripts output v2 with alpha and require dumps with triangle
topology. Older readers that require the exact alpha-free layout must be updated
before reading these files; the updated reader still accepts alpha-free v1/v2.

The current writer also inherits the high-resolution VXZ acceleration: zstd
defaults to level 3 (deflate to level 1), large region unique/index construction
uses CUDA when available with a CPU fallback, and the existing sorted
region/local coordinates are reused to count unique fine voxels. These changes
only affect serialization speed; duplicate records are not merged.

## Sections

1. `region_svo`: a depth-8 SVO by default, containing occupied coarse regions
   in Morton order.
2. `region_counts`: little-endian uint32 record count per decoded SVO leaf.
3. `region_offsets`: little-endian uint64 byte offsets into `records`, including
   a final sentinel offset.
4. `records`: one row per sample, ordered first by coarse Morton region and
   then by local XYZ. Each row is `local_xyz[3]` followed by all uint8
attributes declared by `record_layout`.

Version 1 records occupy 10 uncompressed bytes without alpha: local XYZ (3), RGB (3),
normal (3), confidence (1). Version 2 adds one topology byte (11 bytes total);
an alpha channel adds one byte in either version.
No record ID is stored. The
original dump retains full PBR data for future modality conversion; v1/v2 files
contain the selected RGB target and optional straight alpha. Emission may be folded
into RGB according to `metadata.add_emission`; separate emission and MRA are not stored.

Normal encodes `(n * .5 + .5) * 255` as uint8. Decode with `2*u/255-1`
and **renormalize** before angular comparisons or a normal query. RGB uses the
header's `metadata.color_space`, not an assumed albedo color space. Defaults
remain AgX plus emission; use `--color-space srgb --no-add-emission` for a
base-color target in sRGB.

Confidence is a heuristic for collision ambiguity, not geometric validity.
Count final normal clusters per unique fine XYZ: `m=1,2,3,4,5,6,>=7` maps to
`q=1,1,.9,.7,.5,.3,.1`, stored as `255,255,230,179,128,77,26` (round half up).
All records at that XYZ get the same q. The conversion header records
`metadata.confidence_source="cluster_count_v1"`. Decode loss weights or query
confidence using `attr["confidence"].float()/255`, **not** the dataset's generic
`feats` representation (which scales every selected channel to [-1,1]).

`region_resolution` is a power of two in `[4, 1024]`; 256 is the default. Each
axis of `region_block_size` must be at most 256 so every local coordinate is
representable by its uint8 field.

For a fine coordinate `p`, region coordinate and local coordinate are:

```text
region = p // region_block_size
local  = p %  region_block_size
```

At 1024 resolution with the default 256 coarse-region resolution, the local
block is 4x4x4. Duplicate local positions are legal and preserve independent
normal clusters.

## Normal clustering

Clustering is performed before quantization. Normals use signed cosine
similarity (never `abs(dot)`), so opposite coincident faces remain separate.
A sample may join a cluster only if its angle to every existing member is at
most the configured threshold (15 degrees by default). VXZM uses positive,
winding-symmetric weights:

```text
u = (voxel_center - triangle_closest_point) / voxel_size
w = max(exp(-0.5 * dot(u,u)), 1e-8)
```

The closest point is shared with UV sampling. Cluster normal is normalized
after weighted averaging. This does not make the weights invariant to local
triangle subdivision. Legacy VXZ keeps its original weight implementation.

Versions 1/2 use reoriented **geometric face normals**, without smoothing or
normal maps. Blender dump operates on a temporary evaluated mesh in normalized
world space, preserves UV corners/materials and removes numerical degeneracies.
It welds vertices within each object (1e-7 by default), but skips weld groups
that would silently delete coincident faces. Other duplicate/invalid faces can
still reject the asset.

Open boundaries and multi-face junctions are **allowed by default**. Use
`--max-edge-faces 3` to permit up to three faces sharing an edge, or `2` for
strict edge-manifold filtering; `0` means unlimited. Degree-one boundary edges
are always allowed. A degree cap is a data policy, not a normal-quality test.
Normal orientation propagates only across degree-two edges. Open/branched
patches preserve the original area-majority sign; ordinary closed shells use
outward orientation. Non-orientable degree-two patches and non-finite geometry
are still rejected. No global self-intersection check or universal outward
orientation for open/branched surfaces is claimed.

Version 2 `topology` is a uint8 bitmask. Bit 0 means **at least one contributing
source triangle touches an edge with >2 incident faces after cleanup**. All
other bits are reserved and currently zero. Native clustering bitwise-ORs
provenance without changing sample coordinates, RGB, normal or confidence.
This labels the whole contributing triangle, not just voxels intersecting the
junction edge; long triangles can therefore label records far from that edge.
The header contains `topology_bits` and conversion metadata records
`topology_source="source_triangle_multi_face_edge_v1"`. Open boundaries do not
set bit 0; geometric self-intersections without shared topology are not detected.
Do not automatically downweight occupancy/RGB merely because bit 0 is set.

The dataset keeps `topology` in the raw `attr` mapping; it is deliberately
excluded from default neural input `feats`. Use `(attr['topology'] & 1) != 0`
for filtering, evaluation buckets or future targeted supervision.

Version 0 retains authored corner-normal semantics, including the geometric
opposite-winding guard. `create_vxzm.py --no-renormal` is a marked legacy
comparison path with RGB/normal/confidence channels. Old readers supporting
only older versions cannot read version 2; regenerate old data for a consistent
canonical training set.

`max_records_per_voxel` and `max_total_records` are optional safety limits.
Both default to zero (unlimited), and exceeding either raises an error rather
than silently dropping geometry. `max_total_records` also caps the raw
triangle/voxel sample buffer before clustering, so it is a conservative
memory-safety bound; several raw samples may later merge into one record.

The existing TRELLIS sparse PBR dataset continues to consume `.vxz` only,
because its sparse tensor representation assumes unique coordinates. Read
`.vxzm` through `o_voxel.io.read_vxzm` or
`MultiSurfaceVoxelPbrDataset`, which returns ordinary record tensors and
region offsets rather than silently feeding duplicates into a sparse backend.

Mesh storage is optional. `write_vxzm` accepts `mesh_vertices` and
`mesh_faces` only when a training/validation asset needs the source surface;
the default is mesh-free. When present, the normalized scene mesh is stored
in compressed `mesh_vertices` and `mesh_faces` sections and described by the
header `mesh` object. `read_vxzm_mesh` reads only the header and mesh sections for path inputs,
without decoding voxel records. `read_vxzm(..., return_mesh=True)` decodes
records as usual and appends the mesh tuple (or `None` for mesh-free files). Object boundaries, duplicate
vertices, open surfaces, face winding, and non-manifold connectivity are
preserved.
