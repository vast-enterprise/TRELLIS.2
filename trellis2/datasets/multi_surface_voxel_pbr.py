"""Dataset reader for duplicate-coordinate multi-surface VXZM records.

This deliberately returns ordinary tensors instead of ``SparseTensor``:
torchsparse/spconv assume one feature per coordinate and may coalesce the
normal-separated records that VXZM exists to preserve.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import torch

import o_voxel
from .components import StandardDatasetBase


class MultiSurfaceVoxelPbrDataset(StandardDatasetBase):
    """Load all VXZM records without deduplicating coincident coordinates.

    Each sample contains ``coord`` (fine-grid int32 XYZ), normalized ``feats``,
    the raw uint8 ``attr`` mapping, and the coarse ``region_*`` index. The
    collate function concatenates records and exposes offsets for both samples
    and regions, so downstream models can choose their own multi-surface
    representation without losing information.
    """

    def __init__(
        self,
        roots,
        resolution: int = 1024,
        max_records: int = 5_000_000,
        max_num_faces: int = None,
        min_aesthetic_score: float = 5.0,
        attrs: List[str] = None,
    ):
        self.resolution = int(resolution)
        self.max_records = max_records
        self.max_num_faces = max_num_faces
        self.min_aesthetic_score = min_aesthetic_score
        self.attrs = (list(attrs) if attrs is not None else
                      ["base_color", "metallic", "roughness", "alpha", "normal"])
        if not self.attrs:
            raise ValueError("attrs must contain at least one VXZM attribute")
        super().__init__(roots)
        self.loads = [int(self.metadata.loc[sha256, "num_pbrm_records"])
                      for _, sha256 in self.instances]

    def filter_metadata(self, metadata: pd.DataFrame):
        stats = {}
        required = {"pbrm_voxelized", "num_pbrm_records"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"VXZM metadata is missing columns: {sorted(missing)}")
        metadata = metadata[metadata["pbrm_voxelized"] == True]
        stats["VXZM voxelized"] = len(metadata)
        if self.min_aesthetic_score is not None and "aesthetic_score" in metadata:
            metadata = metadata[metadata["aesthetic_score"] >= self.min_aesthetic_score]
            stats[f"Aesthetic score >= {self.min_aesthetic_score}"] = len(metadata)
        if self.max_records is not None:
            metadata = metadata[metadata["num_pbrm_records"] <= self.max_records]
            stats[f"VXZM records <= {self.max_records}"] = len(metadata)
        if self.max_num_faces is not None and "num_faces" in metadata:
            metadata = metadata[metadata["num_faces"] <= self.max_num_faces]
            stats[f"Faces <= {self.max_num_faces}"] = len(metadata)
        return metadata, stats

    @staticmethod
    def _voxel_root(root):
        return root["pbr_voxel"] if isinstance(root, dict) else root

    def get_instance(self, root, instance: str) -> Dict:
        path = os.path.join(self._voxel_root(root), f"{instance}.vxzm")
        (coord, attr, region_coord, region_counts, region_byte_offsets,
         info) = o_voxel.io.read_vxzm(path, return_regions=True)
        missing = set(self.attrs) - set(attr)
        if missing:
            raise ValueError(f"{path} is missing VXZM attributes: {sorted(missing)}")
        if list(info["grid_size"]) != [self.resolution] * 3:
            raise ValueError(
                f"VXZM grid_size {info['grid_size']} does not match dataset resolution "
                f"{self.resolution}"
            )
        if (region_byte_offsets.numel() != region_counts.numel() + 1 or
                int(region_counts.sum()) != len(coord)):
            raise ValueError("VXZM region index does not cover all records")
        feats = torch.cat([attr[name] for name in self.attrs], dim=-1).float() / 255.0 * 2.0 - 1.0
        return {
            "coord": coord.to(torch.int32),
            "feats": feats,
            "attr": attr,
            "region_coord": region_coord,
            "region_counts": region_counts,
            "region_byte_offsets": region_byte_offsets,
            "info": info,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is not None:
            raise ValueError("MultiSurfaceVoxelPbrDataset does not support split_size collation")
        record_lengths = torch.tensor([len(item["coord"]) for item in batch], dtype=torch.int64)
        region_lengths = torch.tensor([len(item["region_coord"]) for item in batch], dtype=torch.int64)
        record_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), record_lengths.cumsum(0)])
        region_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), region_lengths.cumsum(0)])
        region_counts = torch.cat([item["region_counts"] for item in batch])
        region_record_offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64), region_counts.cumsum(0)
        ])
        return {
            "coord": torch.cat([item["coord"] for item in batch]),
            "feats": torch.cat([item["feats"] for item in batch]),
            "record_offsets": record_offsets,
            "region_coord": torch.cat([item["region_coord"] for item in batch]),
            "region_counts": region_counts,
            "region_offsets": region_offsets,
            "region_record_offsets": region_record_offsets,
            "attr": {
                name: torch.cat([item["attr"][name] for item in batch])
                for name in batch[0]["attr"]
            },
            "info": [item["info"] for item in batch],
        }
