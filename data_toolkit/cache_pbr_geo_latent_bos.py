#!/usr/bin/env python3
"""Cache paired geometry/texture VXZs from BOS as {coords, latents_geo, latents}.

First run download_latent_models.py to populate .hf_models/ckpts.
The BOS upload bucket and prefix are None; set --upload-bucket and
--latent-folder manually. The input bucket is configured separately.
Use --no-upload to retain local latents for inspection without any BOS writes.
Material channels are zeroed after normalization, matching the local script.
"""
from __future__ import annotations
import argparse
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'o-voxel'))
import torch
import torch.multiprocessing as mp
from data_toolkit.cache_pbr_latent_bos import (
    DEFAULT_ATTRS, read_uuid_list, load_encoder,
    _make_bos_client, bos_key, bos_object_exists, download_bos_file,
    upload_bos_file, _env_first, _choose_gpus, _clear_cuda_after_error,
)
from data_toolkit.cache_pbr_geo_latent_local import (
    DEFAULT_PBR_ENCODER, DEFAULT_GEO_ENCODER, _read_pbr, _read_geometry,
    _encode_pbr, _encode_geometry, _offload_encoder, _release_cuda,
)
DEFAULT_INPUT_BUCKET = 'texture-surface-sample'
DEFAULT_UPLOAD_BUCKET = None  # Must be filled in manually before uploading.
DEFAULT_TEXTURE_FOLDER = 'sample_vxz_2048/texture'
DEFAULT_GEOMETRY_FOLDER = 'sample_vxz_2048/geometry'
DEFAULT_LATENT_FOLDER = None  # Must be filled in manually before uploading.
DEFAULT_UUID_LIST = '/mnt/pfs/users/chenjiahao/redisdataprocess/redisdump/pbr_vxz_2048.txt'

def _worker(worker_id: int, gpu_id: int, task_queue: Any, result_queue: Any,
            options: Dict[str, Any]) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    client = _make_bos_client(options['bos_endpoint'], options['access_key_id'], options['secret_access_key'])
    # --offload-encoders keeps only the currently active model on the GPU.
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
        texture_path = work_dir / f"{sample}.texture.vxz"
        geometry_path = work_dir / f"{sample}.geometry.vxz"
        output_path = output_dir / sample[:2] / f"{sample}.pt"
        output_key = bos_key(options['latent_folder'], sample, '.pt') if options['upload'] else None
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
                if options['upload'] and options['skip_existing'] and bos_object_exists(client, options['upload_bucket'], output_key):
                    result_queue.put((sample, 'skipped', 0.0, 'output already exists'))
                    break
                for folder, destination in ((options['texture_vxz_folder'], texture_path),
                                            (options['geometry_vxz_folder'], geometry_path)):
                    download_bos_file(client, options['bucket'], bos_key(folder, sample, '.vxz'), destination)
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
                    upload_bos_file(client, options["upload_bucket"], output_key, output_path)
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
                if not options['keep_inputs']:
                    texture_path.unlink(missing_ok=True)
                    geometry_path.unlink(missing_ok=True)



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--uuid-list", default=DEFAULT_UUID_LIST)
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
    p.add_argument("--work-dir", default=".cache_pbr_geo_latents_bos")
    p.add_argument("--output-dir", default=".cache_pbr_geo_latents_bos/output")
    p.add_argument("--failed-uuids", default=".cache_pbr_geo_latents_bos/failed.txt")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument('--bucket', default=DEFAULT_INPUT_BUCKET, help='Input VXZ bucket only')
    p.add_argument('--upload-bucket', default=DEFAULT_UPLOAD_BUCKET, help='Output bucket; must be set manually')
    p.add_argument('--latent-folder', default=DEFAULT_LATENT_FOLDER)
    p.add_argument('--keep-inputs', action='store_true', help='Retain downloaded VXZs for reconstruction checks')
    p.add_argument('--no-skip-existing', dest='skip_existing', action='store_false')
    p.add_argument('--bos-endpoint', default=os.environ.get('BOS_ENDPOINT', 'bj.bcebos.com'))
    p.add_argument('--bos-access-key-id', default=_env_first('BOS_ACCESS_KEY_ID', 'BOS_ACCESS_KEY'))
    p.add_argument('--bos-secret-access-key', default=_env_first('BOS_SECRET_ACCESS_KEY', 'BOS_SECRET_KEY'))
    p.add_argument("--no-upload", dest="upload", action="store_false")
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
    p.set_defaults(upload=True, offload_encoders=False, skip_existing=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.upload and any(
        value is None or value.strip().lower() in ('none', '/', '')
        for value in (args.upload_bucket, args.latent_folder)
    ):
        parser.error('Upload bucket/folder is None. Set --upload-bucket and --latent-folder manually or use --no-upload.')
    if tuple(args.attrs) != DEFAULT_ATTRS:
        parser.error('Official texture checkpoints require base_color metallic roughness alpha channel order')
    uuids = read_uuid_list(args.uuid_list)
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.end is not None and args.end < args.start:
        raise ValueError("--end must be greater than or equal to --start")
    list_size = len(uuids)
    uuids = uuids[args.start:args.end]
    if not uuids:
        print(
            f"Selected UUID slice [{args.start}:{args.end}] is empty "
            f"(full list contains {list_size} UUIDs); nothing to do."
        )
        return 0
    selected_end = min(args.end if args.end is not None else list_size, list_size)
    print(
        f"Selected {len(uuids)} UUIDs from de-duplicated list indices "
        f"[{args.start}:{selected_end}) (list size: {list_size}).",
        flush=True,
    )
    gpus = _choose_gpus(args)
    options = vars(args).copy()
    options["access_key_id"] = options.pop("bos_access_key_id")
    options["secret_access_key"] = options.pop("bos_secret_access_key")
    # Namespace values are not needed inside workers and can make pickling
    # surprising when this script is embedded by a launcher.
    for parent_only_option in ("uuid_list", "start", "end"):
        options.pop(parent_only_option, None)
    options["gpus"] = None
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    # Keep only a small amount of work queued. This preserves dynamic
    # scheduling without duplicating a 500k-item UUID list in the queue's
    # multiprocessing feeder buffer.
    task_queue = ctx.Queue(maxsize=max(2 * len(gpus), 1))
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=_worker, args=(i, gpu, task_queue, result_queue, options), daemon=False)
        for i, gpu in enumerate(gpus)
    ]
    for worker in workers:
        worker.start()

    failures: List[Tuple[str, str]] = []
    finished = 0
    total = len(uuids)
    next_task = 0
    sentinels_sent = 0
    aborted = False

    def feed_available_slots() -> None:
        """Stream pending UUIDs, then one FIFO termination marker per GPU."""
        nonlocal next_task, sentinels_sent
        while next_task < total:
            try:
                task_queue.put_nowait(uuids[next_task])
            except queue.Full:
                return
            next_task += 1
        while sentinels_sent < len(workers):
            try:
                task_queue.put_nowait(None)
            except queue.Full:
                return
            sentinels_sent += 1

    try:
        while finished < total:
            feed_available_slots()
            try:
                uuid, status, elapsed, detail = result_queue.get(timeout=1.0)
            except queue.Empty:
                # Sentinels are queued after all UUIDs, so a faster worker can
                # finish normally while another worker is still processing its
                # last item. Only fail when no worker remains that could produce
                # the missing result; this also prevents a silent hang after an
                # initialization failure or hard worker crash.
                crashed = [
                    worker
                    for worker in workers
                    if not worker.is_alive() and worker.exitcode not in (None, 0)
                ]
                if crashed or all(not worker.is_alive() for worker in workers):
                    states = ", ".join(
                        f"pid={worker.pid}, exitcode={worker.exitcode}" for worker in workers
                    )
                    raise RuntimeError(
                        f"A GPU worker exited before all UUIDs completed ({states}); "
                        "inspect worker stderr"
                    )
                continue
            finished += 1
            if status == "failed":
                failures.append((uuid, detail))
            print(f"[{finished}/{total}] {status} {uuid} ({elapsed:.1f}s) {detail}", flush=True)
        # The final result can arrive before there was queue capacity for
        # every sentinel. All tasks are complete now, so live workers are
        # ready to consume these remaining termination markers.
        while sentinels_sent < len(workers):
            task_queue.put(None)
            sentinels_sent += 1
    except BaseException:
        aborted = True
        raise
    finally:
        if aborted:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
        for worker in workers:
            worker.join()

    crashed = [worker for worker in workers if worker.exitcode != 0]
    if crashed:
        states = ", ".join(
            f"pid={worker.pid}, exitcode={worker.exitcode}" for worker in crashed
        )
        raise RuntimeError(f"One or more GPU workers failed after processing ({states})")

    if failures:
        failed_path = Path(args.failed_uuids)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_path, "w", encoding="utf-8") as handle:
            for uuid, error in failures:
                handle.write(f"{uuid}\t{error}\n")
        print(f"{len(failures)} UUIDs failed; details written to {failed_path}", file=sys.stderr)
        return 2
    mode = "uploaded" if args.upload else f"saved locally in {args.output_dir}"
    print(f"Completed {total} UUIDs across GPUs {gpus}; latents were {mode}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
