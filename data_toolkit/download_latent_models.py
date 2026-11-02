#!/usr/bin/env python3
"""Download the official TRELLIS.2 VAE weights into the repository's .hf_models."""
import argparse
import os
from pathlib import Path


def main():
    # This explicit download utility must work in inference pods configured offline.
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parents[1] / '.hf_models'))
    args = parser.parse_args()
    names = ('tex_enc_next_dc_f16c32_fp16', 'shape_enc_next_dc_f16c32_fp16',
             'tex_dec_next_dc_f16c32_fp16')
    snapshot_download('microsoft/TRELLIS.2-4B', local_dir=args.output_dir,
                      allow_patterns=[f'ckpts/{name}.{ext}' for name in names
                                      for ext in ('json', 'safetensors')])


if __name__ == '__main__':
    main()
