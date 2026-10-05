"""CLI and API regression for VXZM -> colored/normal PLY visualization."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "o-voxel")]
import o_voxel
import visualize_vxzm


def main():
    coord = torch.tensor([[0, 0, 0], [0, 0, 0], [7, 7, 7]], dtype=torch.int32)
    attr = {
        "base_color": torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=torch.uint8),
        "normal": torch.tensor([[127, 127, 255], [127, 127, 0], [255, 127, 127]], dtype=torch.uint8),
        "confidence": torch.tensor([[255], [255], [230]], dtype=torch.uint8),
        "topology": torch.tensor([[0], [1], [0]], dtype=torch.uint8),
    }
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "sample.vxzm"
        output = root / "sample.ply"
        o_voxel.io.write_vxzm(source, coord, attr, grid_size=8,
                              region_resolution=4, compression="none",
                              normal_source="geometry_reoriented_face_v1")
        assert visualize_vxzm.main([str(source), "--output", str(output),
                                    "--max-points", "2"]) == 0
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(output))
        assert len(cloud.points) == len(cloud.colors) == len(cloud.normals) == 2
        source_coord, source_attr = o_voxel.io.read_vxzm(source)
        expected_indices = torch.linspace(0, len(source_coord) - 1, 2,
                                          dtype=torch.int64)
        assert np.array_equal(
            np.rint(np.asarray(cloud.colors) * 255).astype(np.uint8),
            source_attr["base_color"][expected_indices].numpy(),
        )
        multi_output = root / "multi.ply"
        multi_coord, multi_attr = o_voxel.io.vxzm_multi_sample_to_ply(
            source, multi_output, min_samples=2, normal_offset=0.0,
        )
        assert len(multi_coord) == 2
        assert torch.equal(multi_attr["base_color"], source_attr["base_color"][:2])
        multi_cloud = o3d.io.read_point_cloud(str(multi_output))
        assert len(multi_cloud.points) == len(multi_cloud.colors) == len(multi_cloud.normals) == 2
        # max_points is also available on the public API and validates bad input.
        try:
            o_voxel.io.vxzm_to_ply(source, root / "bad.ply", max_points=0)
        except ValueError as error:
            assert "max_points" in str(error)
        else:
            raise AssertionError("invalid max_points accepted")
    print("VXZM visualization test passed")


if __name__ == "__main__":
    main()
