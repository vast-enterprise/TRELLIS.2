#!/usr/bin/env python3
"""Verify PBR VAE reconstruction on a VXZ object downloaded from BOS."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OVOXEL_ROOT = ROOT / "o-voxel"
if OVOXEL_ROOT.is_dir() and str(OVOXEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OVOXEL_ROOT))

import torch

from data_toolkit.cache_pbr_latent_bos import (
    DEFAULT_ATTRS,
    DEFAULT_BUCKET,
    DEFAULT_ENCODER,
    DEFAULT_VXZ_FOLDER,
    _env_first,
    _make_bos_client,
    bos_key,
    download_bos_file,
    load_encoder,
    read_uuid_list,
    vxz_to_sparse_tensor,
)


DEFAULT_DECODER = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"


def _coordinate_order(coords: torch.Tensor) -> torch.Tensor:
    """Produce a stable lexicographic order for [batch, x, y, z] coords."""
    coords = coords.to(torch.int64)
    max_xyz = max(1024, int(coords[:, 1:].max().item()) + 1)
    code = (((coords[:, 0] * max_xyz + coords[:, 1]) * max_xyz + coords[:, 2])
            * max_xyz + coords[:, 3])
    return torch.argsort(code)


def _write_visualizations(
    output_dir: Path,
    uuid: str,
    input_coords: torch.Tensor,
    input_features: torch.Tensor,
    decoded_features: torch.Tensor,
) -> None:
    import o_voxel

    output_dir.mkdir(parents=True, exist_ok=True)
    channel_slices = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    for label, features in (("input", input_features), ("decoded", decoded_features)):
        quantized = (features.clamp(0, 1) * 255).round().to(torch.uint8)
        attrs = {name: quantized[:, value] for name, value in channel_slices.items()}
        path = output_dir / f"{uuid}_{label}.ply"
        o_voxel.io.write_ply(str(path), input_coords[:, 1:].int(), attrs)
        print(f"Wrote {path}")


def _print_metrics(input_features: torch.Tensor, decoded_features: torch.Tensor) -> None:
    names = ("base_color", "metallic", "roughness", "alpha")
    slices = (slice(0, 3), slice(3, 4), slice(4, 5), slice(5, 6))
    print("Reconstruction metrics in material [0, 1] space (decoded values clipped for metrics):")
    for name, channel_slice in zip(names, slices):
        target = input_features[:, channel_slice]
        prediction = decoded_features[:, channel_slice].clamp(0, 1)
        difference = prediction - target
        mae = difference.abs().mean().item()
        rmse = difference.square().mean().sqrt().item()
        psnr = float("inf") if rmse == 0 else 20.0 * math.log10(1.0 / rmse)
        print(f"  {name:10s} MAE={mae:.6f} RMSE={rmse:.6f} PSNR={psnr:.2f} dB")

    rgb_target = input_features[:, :3]
    rgb_prediction = decoded_features[:, :3].clamp(0, 1)
    cosine = torch.nn.functional.cosine_similarity(rgb_target, rgb_prediction, dim=-1)
    print(f"  base_color cosine mean={cosine.mean().item():.6f}, min={cosine.min().item():.6f}")
    out_of_range = ((decoded_features < 0) | (decoded_features > 1)).float().mean().item()
    print(f"  decoded values outside [0, 1]: {out_of_range * 100:.4f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--uuid-list", required=True)
    parser.add_argument("--start", type=int, default=0, help="Inclusive UUID-list index")
    parser.add_argument("--end", type=int, default=None, help="Exclusive UUID-list index")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--vxz-folder", default=DEFAULT_VXZ_FOLDER)
    parser.add_argument("--attrs", nargs="+", default=list(DEFAULT_ATTRS))
    parser.add_argument("--enc-pretrained", default=DEFAULT_ENCODER)
    parser.add_argument("--dec-pretrained", default=DEFAULT_DECODER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pbr_vae_roundtrip"))
    parser.add_argument("--bos-endpoint", default=os.environ.get("BOS_ENDPOINT", "bj.bcebos.com"))
    parser.add_argument("--bos-access-key-id", default=_env_first("BOS_ACCESS_KEY_ID", "BOS_ACCESS_KEY"))
    parser.add_argument("--bos-secret-access-key", default=_env_first("BOS_SECRET_ACCESS_KEY", "BOS_SECRET_KEY"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start < 0 or (args.end is not None and args.end < args.start):
        raise ValueError("Expected 0 <= start <= end")
    uuids = read_uuid_list(args.uuid_list)[args.start:args.end]
    if not uuids:
        raise ValueError("Selected UUID slice is empty")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    client = _make_bos_client(
        args.bos_endpoint, args.bos_access_key_id, args.bos_secret_access_key
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    uuid = None
    vxz_path = None
    for candidate in uuids:
        candidate_path = args.output_dir / f"{candidate}.vxz"
        key = bos_key(args.vxz_folder, candidate, ".vxz")
        try:
            download_bos_file(client, args.bucket, key, candidate_path)
            uuid, vxz_path = candidate, candidate_path
            break
        except Exception as error:
            candidate_path.unlink(missing_ok=True)
            print(f"Skipping unavailable {candidate}: {error!r}", flush=True)
    if uuid is None or vxz_path is None:
        raise RuntimeError(f"No downloadable VXZ found in selected slice of {len(uuids)} UUIDs")

    try:
        print(f"Verifying {uuid} from bos://{args.bucket}/{bos_key(args.vxz_folder, uuid, '.vxz')}")
        x = vxz_to_sparse_tensor(vxz_path, args.attrs).to(device)
        import trellis2.models as models

        print("Loading encoder...", flush=True)
        encoder = load_encoder(args.enc_pretrained).to(device).eval()
        print("Loading decoder...", flush=True)
        decoder = models.from_pretrained(args.dec_pretrained).to(device).eval()
        with torch.inference_mode():
            z = encoder(x)
            reconstruction = decoder(z)
        torch.cuda.synchronize(device)

        input_order = _coordinate_order(x.coords)
        decoded_order = _coordinate_order(reconstruction.coords)
        input_coords = x.coords[input_order].cpu()
        decoded_coords = reconstruction.coords[decoded_order].cpu()
        coords_equal = torch.equal(input_coords, decoded_coords)
        print(
            f"Voxels input={len(input_coords)}, latent={len(z.coords)}, "
            f"decoded={len(decoded_coords)}, coordinate_match={coords_equal}"
        )
        if not coords_equal:
            raise ValueError("Decoded voxel coordinates do not match the input coordinates")

        input_features = (x.feats[input_order].float().cpu() * 0.5 + 0.5)
        decoded_features = (
            reconstruction.feats[decoded_order].float().cpu() * 0.5 + 0.5
        )
        _print_metrics(input_features, decoded_features)
        _write_visualizations(
            args.output_dir, uuid, input_coords, input_features, decoded_features
        )
        return 0
    finally:
        vxz_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
