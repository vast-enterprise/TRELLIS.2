# VXZM version 0

VXZM stores multi-surface voxel material samples.  Unlike VXZ, two or more
records may have the same fine-grid coordinate when their surface normals are
in different clusters.  VXZ remains unchanged and is not decoded by this
reader.

## Container

- bytes 0..3: ASCII `VXZM`
- byte 4: version (`0`)
- bytes 5..8: big-endian uint32 JSON-end offset
- bytes 9..offset: UTF-8 JSON header
- remaining bytes: independently compressed binary sections

The header records `grid_size`, `region_resolution`, `region_block_size`,
record counts/layout, compression, and every section's relative offset and
compressed length. `normal_source` is fixed to
`surface_authored_with_winding_guard`; a VXZM reader must reject other
semantics instead of silently mixing normal-map data into a surface-clustered
dataset.

## Sections

1. `region_svo`: a depth-8 SVO by default, containing occupied coarse regions
   in Morton order.
2. `region_counts`: little-endian uint32 record count per decoded SVO leaf.
3. `region_offsets`: little-endian uint64 byte offsets into `records`, including
   a final sentinel offset.
4. `records`: one row per sample, ordered first by coarse Morton region and
   then by local XYZ. Each row is `local_xyz[3]` followed by all uint8
attributes declared by `record_layout`.

`base_color` with three uint8 channels and `normal` with three uint8 channels
are mandatory record attributes. The TRELLIS PBR pipeline additionally writes
`metallic`, `roughness`, `emissive`, and `alpha`; generic writers may append
other uint8 PBR attributes through `record_layout`. With `add_emission=true`,
`base_color` already contains the emission contribution, while the separate
`emissive` field is retained for compatibility and debugging.

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
most the configured threshold (15 degrees by default). PBR values use the
legacy voxelizer's weighted average. The same normal representation used for
clustering is stored and normalized after weighted averaging. VXZM uses the
interpolated mesh-authored loop/vertex surface normal and deliberately ignores
normal maps. A signed geometric face-normal guard additionally separates
opposite-winding coincident faces when malformed GLBs reuse the same authored
normal on both sides. This preserves smooth-shaded surface semantics without
losing reversed overlapping geometry.

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
