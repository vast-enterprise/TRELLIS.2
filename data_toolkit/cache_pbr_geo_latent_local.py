#!/usr/bin/env python3
"""Cache geometry and PBR latents for matching local VXZ files.

For every object this utility reads two files::

    <texture-vxz-folder>/<uuid[:2]>/<uuid>.vxz
    <geometry-vxz-folder>/<uuid[:2]>/<uuid>.vxz

The output is one PyTorch file containing the format consumed by the joint
PBR/geometry readers::

    {"coords": ..., "latents_geo": ..., "latents": ...}

Metallic, roughness and alpha inputs are zeroed after normalization; only
base color is taken from the source PBR attributes.

``--no-upload`` is useful for validating a local sample.  In that mode the
completed ``.pt`` files are retained in ``--output-dir`` and no OSS command is
invoked.
"""

from __future__ import annotations

import argparse
import gc
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OVOXEL_ROOT = ROOT / "o-voxel"
if OVOXEL_ROOT.is_dir() and str(OVOXEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OVOXEL_ROOT))

import torch
import torch.multiprocessing as mp

from data_toolkit.cache_pbr_latent_bos import (
    DEFAULT_ATTRS,
    _clear_cuda_after_error,
    load_encoder,
    read_uuid_list,
)


DEFAULT_TEXTURE_FOLDER = "/mnt/oss/wl-texture-surface-sample/sample_vxz_2048/texture"
DEFAULT_GEOMETRY_FOLDER = "/mnt/oss/wl-texture-surface-sample/sample_vxz_2048/geometry"
DEFAULT_CHECKPOINT_DIR = ROOT / ".hf_models" / "ckpts"
DEFAULT_PBR_ENCODER = str(DEFAULT_CHECKPOINT_DIR / "tex_enc_next_dc_f16c32_fp16")
DEFAULT_GEO_ENCODER = str(DEFAULT_CHECKPOINT_DIR / "shape_enc_next_dc_f16c32_fp16")
DEFAULT_OSS_LOCATION = "oss://wl-texture-surface-samplexxx/default_test" # must be specified
DEFAULT_OSSUTIL_CONFIG = "/mnt/pfs/users/chenjiahao/.ossutilconfig"


def local_vxz_path(folder: str | os.PathLike[str], sample: str) -> Path:
    return Path(folder) / sample[:2] / f"{sample}.vxz"


def _read_vxz(path: Path, attrs: Sequence[str]):
    import o_voxel

    coords, data = o_voxel.io.read_vxz(str(path), num_threads=4)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Invalid VXZ coordinates shape in {path}: {tuple(coords.shape)}")
    missing = [name for name in attrs if name not in data]
    if missing:
        raise ValueError(f"{path} is missing VXZ attributes: {missing}")
    return coords, data


def _read_pbr(path: Path, attrs: Sequence[str]):
    from trellis2.modules import sparse as sp

    coords, data = _read_vxz(path, attrs)
    features = torch.cat([data[name] for name in attrs], dim=-1).float()
    features = features / 255.0 * 2.0 - 1.0
    # Cache base color only: zero material inputs AFTER normalization.
    # Resolve channels by attribute name so --attrs ordering stays supported.
    channel_offset = 0
    for name in attrs:
        channel_count = data[name].shape[1]
        if name in ("metallic", "roughness", "alpha"):
            features[:, channel_offset:channel_offset + channel_count] = 0.0
        channel_offset += channel_count
    batch = torch.zeros((coords.shape[0], 1), dtype=coords.dtype)
    return sp.SparseTensor(features, torch.cat([batch, coords], dim=-1))


def _read_geometry(path: Path):
    """Build the two sparse inputs expected by FlexiDualGridVaeEncoder."""
    from trellis2.modules import sparse as sp

    coords, data = _read_vxz(path, ("vertices", "intersected"))
    batch = torch.zeros((coords.shape[0], 1), dtype=coords.dtype)
    sparse_coords = torch.cat([batch, coords], dim=-1)
    vertices = sp.SparseTensor((data["vertices"] / 255.0).float(), sparse_coords)
    bits = data["intersected"]
    intersected = vertices.replace(
        torch.cat([bits % 2, bits // 2 % 2, bits // 4 % 2], dim=-1).bool()
    )
    return vertices, intersected


def _encode_pbr(encoder: Any, voxels: Any, device: torch.device):
    """Encode PBR input while dropping its temporary GPU sparse tensor early."""
    voxels_device = voxels.to(device)
    try:
        with torch.inference_mode():
            latent = encoder(voxels_device)
    finally:
        del voxels_device
    if latent.coords.ndim != 2 or latent.coords.shape[1] != 4:
        raise ValueError(f"PBR encoder returned invalid coordinates: {tuple(latent.coords.shape)}")
    if latent.feats.ndim != 2 or latent.feats.shape[0] != latent.coords.shape[0]:
        raise ValueError("PBR encoder returned incompatible coordinate/feature shapes")
    if not torch.isfinite(latent.feats).all():
        raise ValueError("PBR encoder returned non-finite latent features")
    pack = {
        "coords": latent.coords[:, 1:].detach().cpu(),
        "latents": latent.feats.detach().float().cpu(),
    }
    # The returned pack is CPU-only; release the GPU SparseTensor before the
    # next encoder is moved to CUDA.
    del latent
    return pack


def _encode_geometry(encoder: Any, vertices: Any, intersected: Any, device: torch.device):
    # Keep explicit references so the GPU copies are released immediately
    # after this call.  This matters because ``vertices`` and ``intersected``
    # contain millions of sparse entries for the production VXZs.
    vertices_device = vertices.to(device)
    # ``vertices`` and ``intersected`` describe the same sparse grid.  Reuse
    # the uploaded coordinates instead of transferring/storing an identical
    # N x 4 int32 coordinate tensor twice on CUDA.
    intersected_device = vertices_device.replace(intersected.feats.to(device))
    try:
        with torch.inference_mode():
            latent = encoder(vertices_device, intersected_device)
    finally:
        del vertices_device, intersected_device
    if latent.coords.ndim != 2 or latent.coords.shape[1] != 4:
        raise ValueError(f"geometry encoder returned invalid coordinates: {tuple(latent.coords.shape)}")
    if latent.feats.ndim != 2 or latent.feats.shape[0] != latent.coords.shape[0]:
        raise ValueError("geometry encoder returned incompatible coordinate/feature shapes")
    if not torch.isfinite(latent.feats).all():
        raise ValueError("geometry encoder returned non-finite latent features")
    pack = {
        "coords": latent.coords[:, 1:].detach().cpu(),
        "latents": latent.feats.detach().float().cpu(),
    }
    del latent
    return pack


def _release_cuda(device: torch.device) -> None:
    """Synchronize and release unreachable CUDA allocations after a phase."""
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _offload_encoder(encoder: Any, device: torch.device) -> None:
    """Move one encoder back to CPU before the other encoder is loaded on GPU."""
    try:
        encoder.cpu()
    finally:
        _release_cuda(device)


def upload_oss_file(save_path: Path, oss_location: str, sample: str, config: str) -> None:
    import subprocess

    if not save_path.is_file() or save_path.stat().st_size == 0:
        raise RuntimeError(f"Cannot upload missing/empty file: {save_path}")
    subprocess.run(
        ["ossutil", "cp", "-u", str(save_path),
         os.path.join(oss_location, sample[:2], sample + ".pt"),
         "--config-file", config],
        check=True, capture_output=True, text=True,
    )


def _worker(worker_id: int, gpu_id: int, task_queue: Any, result_queue: Any,
            options: Dict[str, Any]) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    # Keep both parameter sets in host memory, but by default only one model
    # on the GPU at a time.  The previous implementation called ``.to(device)``
    # on both here, making their parameter/storage footprints overlap even
    # though the encoders are invoked sequentially.
    pbr_encoder = load_encoder(options["enc_pretrained"], options["model_root"],
                               options["enc_model"], options["ckpt"])
    geo_encoder = load_encoder(options["geo_enc_pretrained"], options["geo_model_root"],
                               options["geo_enc_model"], options["geo_ckpt"])
    keep_encoders_on_gpu = not options["offload_encoders"]
    if keep_encoders_on_gpu:
        pbr_encoder.to(device)
        geo_encoder.to(device)
    work_dir = Path(options["work_dir"]) / f"worker_{worker_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(options["output_dir"])
    attrs = tuple(options["attrs"])
    while True:
        sample = task_queue.get()
        if sample is None:
            break
        texture_path = local_vxz_path(options["texture_vxz_folder"], sample)
        geometry_path = local_vxz_path(options["geometry_vxz_folder"], sample)
        output_path = output_dir / sample[:2] / f"{sample}.pt"
        errors: List[str] = []
        for attempt in range(options["retries"] + 1):
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            pbr_peak_allocated_mb = pbr_peak_reserved_mb = 0.0
            geo_peak_allocated_mb = geo_peak_reserved_mb = 0.0
            pbr = vertices = intersected = None
            pbr_pack = geo_pack = pack = None
            try:
                for path in (texture_path, geometry_path):
                    if not path.is_file():
                        raise FileNotFoundError(f"Local VXZ does not exist: {path}")
                    if path.stat().st_size == 0:
                        raise ValueError(f"Local VXZ is empty: {path}")

                # Load and encode the two CPU inputs sequentially.  Holding
                # both decoded VXZs at once needlessly increases host RAM by
                # hundreds of MiB for multi-million-voxel samples.
                pbr = _read_pbr(texture_path, attrs)
                if not torch.isfinite(pbr.feats).all():
                    raise ValueError("PBR VXZ contains non-finite features")
                if not keep_encoders_on_gpu:
                    pbr_encoder.to(device)
                try:
                    pbr_pack = _encode_pbr(pbr_encoder, pbr, device)
                finally:
                    if not keep_encoders_on_gpu:
                        _offload_encoder(pbr_encoder, device)
                pbr_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                pbr_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                pbr = None

                vertices, intersected = _read_geometry(geometry_path)
                if not torch.isfinite(vertices.feats).all():
                    raise ValueError("geometry VXZ contains non-finite features")
                if not keep_encoders_on_gpu:
                    geo_encoder.to(device)
                torch.cuda.reset_peak_memory_stats(device)
                try:
                    geo_pack = _encode_geometry(geo_encoder, vertices, intersected, device)
                finally:
                    if not keep_encoders_on_gpu:
                        _offload_encoder(geo_encoder, device)
                geo_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                geo_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                if not torch.equal(pbr_pack["coords"], geo_pack["coords"]):
                    raise ValueError(
                        "PBR and geometry latent coordinates differ: "
                        f"pbr={tuple(pbr_pack['coords'].shape)}, geo={tuple(geo_pack['coords'].shape)}"
                    )
                pack = {
                    "coords": pbr_pack["coords"],
                    "latents_geo": geo_pack["latents"],
                    "latents": pbr_pack["latents"],
                }
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = output_path.with_suffix(".pt.tmp")
                torch.save(pack, tmp_path)
                os.replace(tmp_path, output_path)
                if options["upload"]:
                    upload_oss_file(output_path, options["oss_location"], sample,
                                    options["ossutil_config"])
                    # Only remove the cache after the upload command succeeds.
                    output_path.unlink()
                action = "uploaded; local cache removed" if options["upload"] else "saved"
                peak_allocated_mb = max(pbr_peak_allocated_mb, geo_peak_allocated_mb)
                peak_reserved_mb = max(pbr_peak_reserved_mb, geo_peak_reserved_mb)
                current_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                current_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                result_queue.put((sample, "ok", time.monotonic() - started,
                                  f"gpu={gpu_id}; {pack['coords'].shape[0]} tokens; "
                                  f"geo={pack['latents_geo'].dtype}; pbr={pack['latents'].dtype}; "
                                  f"baseline_allocated={baseline_allocated_mb:.0f}MiB; "
                                  f"peak_allocated={peak_allocated_mb:.0f}MiB; "
                                  f"peak_reserved={peak_reserved_mb:.0f}MiB; "
                                  f"pbr_peak={pbr_peak_allocated_mb:.0f}MiB; "
                                  f"geo_peak={geo_peak_allocated_mb:.0f}MiB; "
                                  f"current_allocated={current_allocated_mb:.0f}MiB; "
                                  f"current_reserved={current_reserved_mb:.0f}MiB ({action})"))
                break
            except Exception as error:
                errors.append(f"attempt {attempt + 1}: {error!r}")
                _clear_cuda_after_error(device)
                if attempt >= options["retries"]:
                    result_queue.put((sample, "failed", time.monotonic() - started,
                                      f"gpu={gpu_id}; " + "; ".join(errors)))
            finally:
                # In particular, do not leave a partially executed encoder on
                # CUDA when an input/encoding/upload attempt raises.
                if not keep_encoders_on_gpu:
                    _offload_encoder(pbr_encoder, device)
                    _offload_encoder(geo_encoder, device)
                # Do not retain references to the current sample's sparse
                # inputs or outputs across queue iterations.  The outputs in
                # ``pack`` are CPU tensors, but explicit deletion also makes
                # exception/retry paths deterministic for Python GC.
                del pbr, vertices, intersected, pbr_pack, geo_pack, pack
                _release_cuda(device)
                (output_path.with_suffix(".pt.tmp")).unlink(missing_ok=True)


def _choose_gpus(args: argparse.Namespace) -> List[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: no CUDA device is visible")
    visible = torch.cuda.device_count()
    if args.gpus is not None:
        if any(g < 0 or g >= visible for g in args.gpus):
            raise ValueError(f"--gpus must be in [0, {visible - 1}]")
        return list(dict.fromkeys(args.gpus))
    count = args.num_gpus if args.num_gpus is not None else visible
    if count < 1 or count > visible:
        raise ValueError(f"--num-gpus must be between 1 and {visible}")
    return list(range(count))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--uuid-list", required=True)
    p.add_argument("--texture-vxz-folder", default=DEFAULT_TEXTURE_FOLDER)
    p.add_argument("--geometry-vxz-folder", default=DEFAULT_GEOMETRY_FOLDER)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--enc-pretrained", default=DEFAULT_PBR_ENCODER)
    p.add_argument("--geo-enc-pretrained", default=DEFAULT_GEO_ENCODER)
    p.add_argument("--model-root", default=None)
    p.add_argument("--enc-model", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--geo-model-root", default=None)
    p.add_argument("--geo-enc-model", default=None)
    p.add_argument("--geo-ckpt", default=None)
    p.add_argument("--attrs", nargs="+", default=list(DEFAULT_ATTRS))
    p.add_argument("--num-gpus", type=int, default=None)
    p.add_argument("--gpus", type=int, nargs="+", default=None)
    p.add_argument("--work-dir", default=".cache_pbr_geo_latents_local")
    p.add_argument("--output-dir", default=".cache_pbr_geo_latents_local/output")
    p.add_argument("--failed-uuids", default="cache_pbr_geo_latent_local_failed.txt")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--oss-location", default=DEFAULT_OSS_LOCATION)
    p.add_argument("--ossutil-config", default=DEFAULT_OSSUTIL_CONFIG)
    p.add_argument("--no-upload", dest="upload", action="store_false")
    # Retain the old spelling as a hidden compatibility no-op: keeping both
    # models resident is now the default behavior.
    p.add_argument(
        "--keep-encoders-on-gpu",
        dest="offload_encoders",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--offload-encoders",
        action="store_true",
        help="Move encoders back to CPU between phases to reduce peak GPU memory",
    )
    p.set_defaults(upload=True, offload_encoders=False)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    uuids = read_uuid_list(args.uuid_list)
    if args.start < 0 or (args.end is not None and args.end < args.start):
        raise ValueError("invalid --start/--end range")
    uuids = uuids[args.start:args.end]
    if not uuids:
        print("No UUIDs selected; nothing to do.")
        return 0
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    gpus = _choose_gpus(args)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    options = vars(args).copy()
    options.pop("uuid_list", None)
    options.pop("start", None)
    options.pop("end", None)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(2 * len(gpus), 1))
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(i, gpu, task_queue, result_queue, options))
               for i, gpu in enumerate(gpus)]
    for worker in workers:
        worker.start()
    failures: List[Tuple[str, str]] = []
    finished = next_task = sentinels = 0
    try:
        while finished < len(uuids):
            while next_task < len(uuids):
                try:
                    task_queue.put_nowait(uuids[next_task])
                    next_task += 1
                except queue.Full:
                    break
            if next_task == len(uuids):
                while sentinels < len(workers):
                    try:
                        task_queue.put_nowait(None)
                        sentinels += 1
                    except queue.Full:
                        break
            try:
                sample, status, elapsed, detail = result_queue.get(timeout=1.0)
            except queue.Empty:
                if any(not w.is_alive() and w.exitcode not in (None, 0) for w in workers):
                    raise RuntimeError("A GPU worker exited before all UUIDs completed")
                continue
            finished += 1
            if status == "failed":
                failures.append((sample, detail))
            print(f"[{finished}/{len(uuids)}] {status} {sample} ({elapsed:.1f}s) {detail}", flush=True)
    finally:
        for worker in workers:
            worker.join() if worker.is_alive() else None
    if failures:
        failed_path = Path(args.failed_uuids)
        with failed_path.open("w", encoding="utf-8") as f:
            for sample, detail in failures:
                f.write(f"{sample}\t{detail}\n")
        return 2
    print(f"Completed {len(uuids)} UUIDs across GPUs {gpus}; output is in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
