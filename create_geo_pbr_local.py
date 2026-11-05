#!/usr/bin/env python3
"""Create aligned geometry/PBR voxels from a local GLB without BOS credentials.

Example:
    python create_geo_pbr_local.py asset --glb model.glb --rotation-degrees 0 0 -90

Use --pkl-only to save only the transformed Blender dump. Geometry-normal
repair is enabled by the shared Blender helper. BOS credentials are required
only with --upload; the supplied GLB is always retained.
"""
from typing import Optional

import create_geo_pbr


def main(argv: Optional[list[str]] = None) -> int:
    return create_geo_pbr.main(argv, local=True)


if __name__ == '__main__':
    raise SystemExit(main())
