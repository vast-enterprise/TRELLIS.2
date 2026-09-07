#!/usr/bin/env python3
"""VXZM dataset loading/collation without duplicate-coordinate coalescing."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "o-voxel"))
sys.path.insert(0, str(ROOT))

import o_voxel
from trellis2.datasets.multi_surface_voxel_pbr import MultiSurfaceVoxelPbrDataset


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        coord = torch.tensor([[0, 0, 0], [0, 0, 0], [7, 7, 7]], dtype=torch.int32)
        attr = {
            "base_color": torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=torch.uint8),
            "metallic": torch.zeros((3, 1), dtype=torch.uint8),
            "roughness": torch.full((3, 1), 255, dtype=torch.uint8),
            "alpha": torch.full((3, 1), 255, dtype=torch.uint8),
            "normal": torch.tensor([[127, 127, 255], [127, 127, 0], [255, 127, 127]], dtype=torch.uint8),
        }
        for index in range(2):
            o_voxel.io.write_vxzm(root / f"sample{index}.vxzm", coord, attr,
                                   grid_size=8, region_resolution=4,
                                   compression="none")
        pd.DataFrame([
            {"sha256": "sample0", "pbrm_voxelized": True, "num_pbrm_records": 3},
            {"sha256": "sample1", "pbrm_voxelized": True, "num_pbrm_records": 3},
        ]).to_csv(root / "metadata.csv", index=False)

        dataset = MultiSurfaceVoxelPbrDataset(
            str(root), resolution=8, min_aesthetic_score=None, max_records=10,
        )
        sample = dataset[0]
        assert len(sample["coord"]) == 3
        assert torch.equal(sample["coord"][0], sample["coord"][1])
        assert not torch.equal(sample["attr"]["normal"][0], sample["attr"]["normal"][1])
        assert int(sample["region_counts"].sum()) == 3
        assert sample["region_byte_offsets"].tolist()[-1] > 0

        batch = dataset.collate_fn([dataset[0], dataset[1]])
        assert batch["record_offsets"].tolist() == [0, 3, 6]
        assert batch["region_offsets"].tolist()[-1] == len(batch["region_coord"])
        assert batch["region_record_offsets"].tolist()[-1] == 6
        assert len(batch["coord"]) == len(batch["feats"]) == 6
        assert torch.equal(batch["attr"]["normal"][:2], sample["attr"]["normal"][:2])
    print("multi-surface VXZM dataset test passed")


if __name__ == "__main__":
    main()
