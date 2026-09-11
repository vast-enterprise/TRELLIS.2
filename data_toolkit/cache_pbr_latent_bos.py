#!/usr/bin/env python3
"""Encode PBR VXZ files from BOS and cache the VAE latents back to BOS.

Unlike a distributed sampler, this program does not partition the UUID list by
rank.  The parent process puts all UUIDs in one multiprocessing queue and each
GPU process requests another UUID only after it has finished its previous one.
Consequently, a slow/large VXZ does not leave another GPU idle at the end of a
run.

The cached file has the format used by the diffusion data readers::

    {"coords": <N x 3 CPU Tensor>, "latents": <N x C CPU Tensor>}

Example::

    python data_toolkit/cache_pbr_latent_bos.py \
        --uuid-list /path/to/uuids.txt \
        --enc-pretrained /path/to/tex_enc_next_dc_f16c32_fp16 \
        --num-gpus 8

BOS credentials are read from ``BOS_ACCESS_KEY_ID`` and
``BOS_SECRET_ACCESS_KEY`` (the ``BOS_ACCESS_KEY``/``BOS_SECRET_KEY`` aliases
are also accepted).  No credentials are stored in this repository.
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make ``python data_toolkit/cache_pbr_latent_bos.py`` work from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OVOXEL_ROOT = ROOT / "o-voxel"
if OVOXEL_ROOT.is_dir() and str(OVOXEL_ROOT) not in sys.path:
    # ``setup.sh`` installs o-voxel, but this makes the utility usable from a
    # source checkout before installation as well.
    sys.path.insert(0, str(OVOXEL_ROOT))

import torch
import torch.multiprocessing as mp


UUID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
DEFAULT_BUCKET = "texture-surface-sample"
DEFAULT_VXZ_FOLDER = "sample_vxz"
DEFAULT_LATENT_FOLDER = "latentsvxz64"
DEFAULT_ENCODER = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
DEFAULT_ATTRS = ("base_color", "metallic", "roughness", "alpha")


def read_uuid_list(path: str | os.PathLike[str]) -> List[str]:
    """Read one UUID/object id per line, dropping blanks and duplicates."""
    uuids: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            # It is convenient to pass a list generated from a directory.
            if value.endswith(".vxz"):
                value = value[:-4]
            if not UUID_RE.fullmatch(value):
                raise ValueError(
                    f"Invalid UUID/object id on line {line_number}: {value!r}. "
                    "IDs must not contain path separators."
                )
            if value not in seen:
                seen.add(value)
                uuids.append(value)
    return uuids


def bos_key(folder: str, uuid: str, suffix: str) -> str:
    """Build a POSIX BOS key independent of the host operating system."""
    return "/".join((folder.strip("/"), uuid[:2], uuid + suffix))


def _make_bos_client(endpoint: str, access_key_id: str, secret_access_key: str):
    """Create a BOS client lazily (so ``--help`` works without the SDK)."""
    if not access_key_id or not secret_access_key:
        raise RuntimeError(
            "BOS credentials are missing. Set BOS_ACCESS_KEY_ID and "
            "BOS_SECRET_ACCESS_KEY (or pass --bos-access-key-id and "
            "--bos-secret-access-key)."
        )
    try:
        from baidubce.auth.bce_credentials import BceCredentials
        from baidubce.bce_client_configuration import BceClientConfiguration
        from baidubce.services.bos.bos_client import BosClient
    except ImportError as error:  # pragma: no cover - depends on runtime image
        raise RuntimeError("The baidubce package is required for BOS I/O") from error
    config = BceClientConfiguration(
        credentials=BceCredentials(access_key_id, secret_access_key),
        endpoint=endpoint,
    )
    return BosClient(config)


def bos_object_exists(client: Any, bucket: str, key: str) -> bool:
    """Return False for a missing object, while propagating other failures."""
    try:
        client.get_object_meta_data(bucket, key)
        return True
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        error_code = str(getattr(error, "code", "")).lower()
        # BOS SDK does not expose one stable NotFound exception across
        # versions, so recognize the common status/code/message variants.
        text = str(error).lower()
        not_found_markers = ("404", "not found", "nosuchkey", "does not exist")
        if (
            status_code == 404
            or error_code in {"nosuchkey", "notfound", "no_such_key"}
            or any(marker in text for marker in not_found_markers)
        ):
            return False
        raise


def download_bos_file(client: Any, bucket: str, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.get_object_to_file(bucket, key, str(destination))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"BOS download produced no data: bos://{bucket}/{key}")


def upload_bos_file(client: Any, bucket: str, key: str, source: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Cannot upload missing/empty file: {source}")
    client.put_object_from_file(bucket, key, str(source))


def vxz_to_sparse_tensor(vxz_path: str | os.PathLike[str], attrs: Sequence[str] = DEFAULT_ATTRS):
    """Read one VXZ and convert its quantized attributes to VAE input."""
    import o_voxel
    from trellis2.modules import sparse as sp

    coords, attr = o_voxel.io.read_vxz(str(vxz_path), num_threads=4)
    missing = [name for name in attrs if name not in attr]
    if missing:
        raise ValueError(f"{vxz_path} is missing VXZ attributes: {missing}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Invalid VXZ coordinates shape: {tuple(coords.shape)}")
    features = torch.cat([attr[name] for name in attrs], dim=-1).float()
    # VXZ stores material attributes as uint8 in [0, 255], while the SC-VAE
    # is trained on [-1, 1].
    features = features / 255.0 * 2.0 - 1.0
    batch_column = torch.zeros((coords.shape[0], 1), dtype=coords.dtype)
    return sp.SparseTensor(features, torch.cat([batch_column, coords], dim=-1))


def load_encoder(
    enc_pretrained: str,
    model_root: Optional[str] = None,
    enc_model: Optional[str] = None,
    ckpt: Optional[str] = None,
):
    """Load either a ``models.from_pretrained`` encoder or a local config/ckpt."""
    import json
    import trellis2.models as models
    from easydict import EasyDict as edict

    if enc_model is None:
        encoder = models.from_pretrained(enc_pretrained)
    else:
        if not model_root or not ckpt:
            raise ValueError("--model-root and --ckpt are required with --enc-model")
        config_path = os.path.join(model_root, enc_model, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = edict(json.load(handle))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args)
        ckpt_path = os.path.join(model_root, enc_model, "ckpts", f"encoder_{ckpt}.pt")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=False)
        print(f"Loaded encoder checkpoint from {ckpt_path}", flush=True)
    return encoder.eval()


def _clear_cuda_after_error(device: torch.device) -> None:
    # synchronize can itself fail after a device-side error, so keep cleanup
    # best-effort and do not hide the original exception.
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _encode_one(encoder: Any, voxels: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    voxels = voxels.to(device)
    with torch.inference_mode():
        latent = encoder(voxels)
    if latent.coords.ndim != 2 or latent.coords.shape[1] != 4:
        raise ValueError(f"encoder returned invalid coordinates: {tuple(latent.coords.shape)}")
    if latent.feats.ndim != 2 or latent.feats.shape[0] != latent.coords.shape[0]:
        raise ValueError(
            "encoder returned incompatible coordinate/feature shapes: "
            f"{tuple(latent.coords.shape)} vs {tuple(latent.feats.shape)}"
        )
    if not torch.isfinite(latent.feats).all():
        raise ValueError("encoder returned non-finite latent features")
    return {
        "coords": latent.coords[:, 1:].detach().cpu(),
        # TRELLIS.2's official PBR latent exporter stores float32 features.
        # Make that contract explicit even for locally configured encoders.
        "latents": latent.feats.detach().float().cpu(),
    }


def _worker(
    worker_id: int,
    gpu_id: int,
    task_queue: Any,
    result_queue: Any,
    options: Dict[str, Any],
) -> None:
    """One process per GPU; work is claimed only after the previous item ends."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    client = _make_bos_client(
        options["bos_endpoint"], options["access_key_id"], options["secret_access_key"]
    )
    encoder = load_encoder(
        options["enc_pretrained"], options["model_root"], options["enc_model"], options["ckpt"]
    ).to(device)

    work_dir = Path(options["work_dir"]) / f"worker_{worker_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    attrs = tuple(options["attrs"])
    while True:
        uuid = task_queue.get()
        if uuid is None:
            break
        vxz_path = work_dir / f"{uuid}.vxz"
        latent_path = work_dir / f"{uuid}.pt"
        input_key = bos_key(options["vxz_folder"], uuid, ".vxz")
        output_key = bos_key(options["latent_folder"], uuid, ".pt")
        errors: List[str] = []
        for attempt in range(options["retries"] + 1):
            started = time.monotonic()
            try:
                if options["upload"] and options["skip_existing"] and bos_object_exists(
                    client, options["bucket"], output_key
                ):
                    result_queue.put(
                        (uuid, "skipped", 0.0, f"gpu={gpu_id}; output already exists")
                    )
                    break

                download_bos_file(client, options["bucket"], input_key, vxz_path)
                voxels = vxz_to_sparse_tensor(vxz_path, attrs)
                if not torch.isfinite(voxels.feats).all():
                    raise ValueError("input VXZ contains non-finite features")
                pack = _encode_one(encoder, voxels, device)

                if options["upload"]:
                    # Write locally first, then upload. A same-directory temporary
                    # file prevents a partially written .pt from ever being uploaded.
                    tmp_path = latent_path.with_suffix(".pt.tmp")
                    torch.save(pack, tmp_path)
                    upload_bos_file(client, options["bucket"], output_key, tmp_path)
                    tmp_path.unlink(missing_ok=True)
                    action = "uploaded"
                else:
                    # Pipeline validation mode: keep neither local nor BOS output.
                    action = "discarded"
                result_queue.put((
                    uuid,
                    "ok",
                    time.monotonic() - started,
                    f"gpu={gpu_id}; {pack['coords'].shape[0]} tokens; "
                    f"coords={pack['coords'].dtype}; "
                    f"latents={pack['latents'].dtype} ({action})",
                ))
                break
            except Exception as error:
                errors.append(f"attempt {attempt + 1}: {error!r}")
                _clear_cuda_after_error(device)
                if attempt >= options["retries"]:
                    result_queue.put((
                        uuid,
                        "failed",
                        time.monotonic() - started,
                        f"gpu={gpu_id}; " + "; ".join(errors),
                    ))
            finally:
                vxz_path.unlink(missing_ok=True)
                latent_path.unlink(missing_ok=True)
                # Remove a stale temporary file after an interrupted upload.
                (latent_path.with_suffix(".pt.tmp")).unlink(missing_ok=True)


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--uuid-list", required=True, help="Text file containing one UUID/object id per line")
    parser.add_argument("--start", type=int, default=0, help="Inclusive index in the de-duplicated UUID list")
    parser.add_argument("--end", type=int, default=None, help="Exclusive index in the de-duplicated UUID list")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--vxz-folder", default=DEFAULT_VXZ_FOLDER)
    parser.add_argument("--latent-folder", default=DEFAULT_LATENT_FOLDER)
    parser.add_argument("--enc-pretrained", default=DEFAULT_ENCODER, help="Local model prefix or Hugging Face model path")
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--enc-model", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--attrs", nargs="+", default=list(DEFAULT_ATTRS), help="VXZ attributes passed to the encoder")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of visible GPUs; defaults to torch.cuda.device_count()")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="Explicit CUDA device ids")
    parser.add_argument("--work-dir", default=".cache_vxz_latents")
    parser.add_argument("--failed-uuids", default="cache_pbr_latent_failed.txt")
    parser.add_argument("--retries", type=int, default=1, help="Additional attempts after a failed UUID")
    parser.add_argument("--no-upload", dest="upload", action="store_false", help="Encode and then discard latents; do not write to BOS")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Encode even when the output BOS object exists")
    parser.set_defaults(skip_existing=True, upload=True)
    parser.add_argument("--bos-endpoint", default=os.environ.get("BOS_ENDPOINT", "bj.bcebos.com"))
    parser.add_argument("--bos-access-key-id", default=_env_first("BOS_ACCESS_KEY_ID", "BOS_ACCESS_KEY"))
    parser.add_argument("--bos-secret-access-key", default=_env_first("BOS_SECRET_ACCESS_KEY", "BOS_SECRET_KEY"))
    return parser


def _choose_gpus(args: argparse.Namespace) -> List[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: no CUDA device is visible")
    visible = torch.cuda.device_count()
    if args.gpus is not None:
        if any(g < 0 or g >= visible for g in args.gpus):
            raise ValueError(f"--gpus must be in [0, {visible - 1}] for the visible devices")
        gpus = list(dict.fromkeys(args.gpus))
    else:
        count = args.num_gpus if args.num_gpus is not None else visible
        if count < 1 or count > visible:
            raise ValueError(f"--num-gpus must be between 1 and {visible}")
        gpus = list(range(count))
    if not gpus:
        raise ValueError("At least one GPU is required")
    return gpus


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
    mode = "uploaded" if args.upload else "encoded and discarded"
    print(f"Completed {total} UUIDs across GPUs {gpus}; latents were {mode}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
