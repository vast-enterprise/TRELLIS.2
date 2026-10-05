#!/usr/bin/env python3
"""Create a pure RGB + geometry-normal + confidence + topology VXZM by GLB UUID.

Example:
    python create_vxzm.py UUID --resolution 1024 --output-dir vxzm_outputs

Re-normal is enabled by default. --no-renormal is a legacy authored-normal
comparison mode, marked VXZM v0. Only .vxzm is written; no VXZ/PLY output.
Download, upload, normalization, and cache lifecycle are shared with create_vxz.
"""
from create_vxz import build_parser as _build_parser, main as _main


def build_parser():
    return _build_parser(pure_vxzm=True)


def main(argv=None):
    return _main(argv, pure_vxzm=True)


if __name__ == '__main__':
    raise SystemExit(main())
