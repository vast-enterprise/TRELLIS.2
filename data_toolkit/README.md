# Dataset Preparation Toolkit

This toolkit provides a comprehensive pipeline for preparing 3D datasets, including downloading, processing, voxelizing, and latent encoding for SC-VAE and Flow Model training.

### Step 1: Install Dependencies

Initialize the environment and install necessary dependencies:

```bash
. ./data_toolkit/setup.sh
```

### Step 2: Initialize Metadata

Before processing, load the dataset metadata.

```bash
python data_toolkit/build_metadata.py <SUBSET> --root <ROOT> [--source <SOURCE>]
```

**Arguments:**
- `SUBSET`: Target dataset subset. Options: `ObjaverseXL`, `ABO`, `HSSD`, `TexVerse` (Training sets); `SketchfabPicked`, `Toys4k` (Test sets).
- `ROOT`: Root directory to save the data.
- `SOURCE`: Data source (Required if `SUBSET` is `ObjaverseXL`). Options: `sketchfab`, `github`.

**Example:**
Load metadata for `ObjaverseXL` (sketchfab) and save to `datasets/ObjaverseXL_sketchfab`:
```bash
python data_toolkit/build_metadata.py ObjaverseXL --source sketchfab --root datasets/ObjaverseXL_sketchfab
```

### Step 3: Download Data

Download the 3D assets to the local storage.

```bash
python data_toolkit/download.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Arguments:**
- `RANK` / `WORLD_SIZE`: Parameters for multi-node distributed downloading.

**Example:**
To download the `ObjaverseXL` subset:

> **Note:** The example below sets a large `WORLD_SIZE` (160,000) for demonstration purposes, meaning only a tiny fraction of the dataset will be downloaded by this single process.

```bash
python data_toolkit/download.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --world_size 160000
```

*Attention: Some datasets may require an interactive Hugging Face login or manual steps. Please follow any on-screen instructions.*

**Update Metadata:**
After downloading, update the metadata registry:
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Step 4: Process Mesh and PBR Textures

Standardize 3D assets by dumping mesh and PBR textures.
*Note: This process utilizes the CPU.*

```bash
# Dump Meshes
python data_toolkit/dump_mesh.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]

# Dump PBR Textures
python data_toolkit/dump_pbr.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]

# Get statisitics of the asset
python asset_stats.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Example:**
```bash
python data_toolkit/dump_mesh.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/dump_pbr.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python asset_stats.py --root datasets/ObjaverseXL_sketchfab
```

**Update Metadata:**
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Step 5: Convert to O-Voxels

Convert the processed meshes and textures into O-Voxels format.
*Note: This process utilizes the CPU.*

```bash
python data_toolkit/dual_grid.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>]

python data_toolkit/voxelize_pbr.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>]
```

**Arguments:**
- `RESOLUTION`: Target resolutions for O-Voxels, comma-separated (e.g., `256,512,1024`). Default is `256`.

**Example:**
Convert `ObjaverseXL` to resolutions 256, 512, and 1024:
```bash
python data_toolkit/dual_grid.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256,512,1024
python data_toolkit/voxelize_pbr.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256,512,1024
```

`voxelize_pbr.py` writes the final material color to the existing
`base_color` voxel attribute.  For Blender-aligned colors the defaults are
AgX output and emission enabled; these can be selected explicitly or changed:

```bash
python data_toolkit/voxelize_pbr.py ObjaverseXL \
  --root datasets/ObjaverseXL_sketchfab --resolution 1024 \
  --color_space agx --add_emission
```

Supported `--color_space` values are `linear`, `srgb` (IEC sRGB OETF), and
`agx` (Blender AgX Base sRGB).  Use `--no-add-emission` to store only the
non-emissive base color.

Multi-surface material samples use the separate `.vxzm` format. It preserves
multiple records at one fine voxel when their signed surface-normal angle is
greater than 15 degrees. Existing `.vxz` output remains the default and is
unchanged. Generate either or both formats with:

VXZM requires PBR dumps produced by the current `dump_pbr.py`, which preserves
Blender-authored corner normals in world space. Legacy dumps without the
`surface_normal_source=blender_authored_corner_world_v1` marker are rejected;
re-run the PBR dump stage for those GLBs. This check applies only to VXZM, so
the existing VXZ path remains compatible with older dumps.

```bash
python data_toolkit/voxelize_pbr.py ObjaverseXL \
  --root datasets/ObjaverseXL_sketchfab --resolution 1024 \
  --output-format both --region-resolution 256 \
  --cluster-angle-degrees 15 \
  --max-records-per-voxel 0 --max-total-records 0
```

At 1024 resolution, each occupied leaf of the 256-resolution coarse SVO owns
a 4x4x4 local block. Its record count, byte offset, local positions, colors,
surface normals, metallic, roughness, emissive, and alpha attributes are
stored independently. See
`o-voxel/VXZM_FORMAT.md` for the binary layout. A standalone conversion and
PLY visualization test is available as:

```bash
PYTHONPATH=o-voxel:. python tests/test_glb_to_vxzm.py \
  --glb path/to/model.glb --output-dir /tmp/trellis2_vxzm_test \
  --resolution 1024 --region-resolution 256
```

For training-side experimentation, `MultiSurfaceVoxelPbrDataset` reads VXZM
as ordinary `coord`/`feats` tensors plus sample and region offsets. It does not
construct the existing sparse-convolution tensor, because duplicate
coordinates would otherwise be coalesced and lose their normal clusters.
VXZM cache metadata includes the complete record layout, so schema changes
such as adding a PBR field automatically invalidate and regenerate old files.

For a standalone GLB smoke test (including Blender dumping, VXZ round-trip,
and PLY visualization exports), run:

```bash
PYTHONPATH=o-voxel:. python tests/test_glb_to_vxz.py \
  --glb path/to/model.glb --output-dir /tmp/trellis2_glb_test \
  --blender blender --resolution 64 --color-space agx
```

The script writes `<model>.vxz`, coordinate-only `<model>_voxel.ply`, and
colored `<model>.ply` / `<model>_color.ply` files with standard uint8 `red`,
`green`, and `blue` fields. The
repository-local `ocioutils/` directory contains the exact OCIO conversion
module, configuration, and LUTs used for AgX (`ocioutils/color_conversion.py`,
`ocioutils/config.ocio`, `ocioutils/luts/`, and `ocioutils/filmic/`); no approximation is used when
`--color-space agx` is selected.

To configure a fresh Debian/Ubuntu CUDA Pod and build the extension, run
`bash set_ovoxel.sh` from the repository root. Use `SKIP_APT=1` when the
system packages are already installed, or `SKIP_BUILD=1` to only validate the
runtime dependencies.


### At this point, the dataset is ready for SC-VAE Training

### Step 6: Encode Latents

Encode sparse structures into latents to train the first-stage generator.

```bash
# 1. Encode Shape Latents
python data_toolkit/encode_shape_latent.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>]

# 2. Encode PBR Latents
python data_toolkit/encode_pbr_latent.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>]

# 3. Update Metadata (Required before next step)
python data_toolkit/build_metadata.py <SUBSET> --root <ROOT>

# 4. Encode Sparse Structure (SS) Latents
python data_toolkit/encode_ss_latent.py --root <ROOT> --shape_latent_name <SHAPE_LATENT_NAME> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <SS_RESOLUTION>] 
```

**Arguments:**
- `RESOLUTION`: Input O-Voxel resolution. Default is `1024`.
- `SS_RESOLUTION`: Resolution for sparse structures. Default is `64`.
- `SHAPE_LATENT_NAME`: The specific version name of the shape latent.

**Example:**
```bash
python data_toolkit/encode_shape_latent.py --root datasets/ObjaverseXL_sketchfab --resolution 512
python data_toolkit/encode_pbr_latent.py --root datasets/ObjaverseXL_sketchfab --resolution 512
python data_toolkit/encode_shape_latent.py --root datasets/ObjaverseXL_sketchfab --resolution 1024
python data_toolkit/encode_pbr_latent.py --root datasets/ObjaverseXL_sketchfab --resolution 1024

# Update metadata
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab

# Encode SS Latents
python data_toolkit/encode_ss_latent.py --root datasets/ObjaverseXL_sketchfab --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024 --resolution 64

# Final Metadata Update
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Caching PBR Latents from BOS

For VXZ objects stored at
`texture-surface-sample/sample_vxz/<uuid[:2]>/<uuid>.vxz`, use the dynamic
multi-GPU cacher below. Every GPU claims the next UUID only after finishing
its current one, so objects are not statically split by rank. The output is
written to
`texture-surface-sample/latentsvxz64/<uuid[:2]>/<uuid>.pt` in the format
`{"coords": <int32 N x 3>, "latents": <float32 N x 32>}` used by the
diffusion data readers.

```bash
export BOS_ACCESS_KEY_ID="..."
export BOS_SECRET_ACCESS_KEY="..."
python data_toolkit/cache_pbr_latent_bos.py \
    --uuid-list /path/to/uuid_list.txt \
    --start 0 --end 10000 \
    --enc-pretrained /path/to/tex_enc_next_dc_f16c32_fp16 \
    --num-gpus 8
```

The script skips existing BOS outputs by default, retries each failed UUID
once, and writes failures to `cache_pbr_latent_failed.txt`. Use
`--no-skip-existing` to overwrite existing outputs or `--gpus 0 2 3` to select
specific visible devices. `--start` is inclusive and `--end` is exclusive;
use these options to split the same list across machines. `--no-upload` runs
the complete download/encode pipeline but discards each resulting latent.
For a locally configured encoder checkpoint, pass
`--model-root`, `--enc-model`, and `--ckpt` instead of `--enc-pretrained`.

Before enabling uploads, the full BOS download and multi-GPU encoding path can
be exercised while discarding every latent:

```bash
python data_toolkit/cache_pbr_latent_bos.py \
    --uuid-list /path/to/uuid_list.txt \
    --start 0 --end 10 \
    --num-gpus 2 \
    --no-upload \
    --work-dir /tmp/cache_pbr_latent_smoke
```

To verify the encoder and decoder reconstruction on the first downloadable VXZ
in a list slice, run:

```bash
python data_toolkit/verify_pbr_vae_bos.py \
    --uuid-list /path/to/uuid_list.txt \
    --start 0 --end 10 \
    --device cuda:0 \
    --output-dir /tmp/pbr_vae_roundtrip
```

This prints per-attribute reconstruction metrics and writes input/decoded PLY
files for visual inspection.

### Step 7: Render Image Conditions

Render multi-view images to train the image-conditioned generator.
*Note: This process may utilize the CPU.*

```bash
python data_toolkit/render_cond.py <SUBSET> --root <ROOT> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Arguments:**
- `NUM_VIEWS`: Number of views to render per asset. Default is `16`.

**Example:**
```bash
python data_toolkit/render_cond.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

**Final Metadata Update:**
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```
