#include <torch/extension.h>
#include "api.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>


/**
 * Encode a list of sparse voxel morton codes into a sparse voxel octree
 * NOTE: The input indices must be sorted in ascending order
 * 
 * @param codes    [N] uint32 tensor containing the morton codes
 * @param depth    The depth of the sparse voxel octree
 * 
 * @return         uint8 tensor containing the sparse voxel octree
 */
torch::Tensor encode_sparse_voxel_octree_cpu(
    const torch::Tensor& codes,
    const uint32_t depth
) {
    size_t N_leaf = codes.size(0);
    int* codes_data = codes.data_ptr<int>();
    
    std::vector<uint8_t> svo;
    std::vector<uint8_t> stack(depth-1);
    std::vector<uint32_t> insert_stack(depth);
    std::vector<uint32_t> stack_ptr(depth);
    uint32_t code, insert_from;

    // Root node
    svo.push_back(0);
    stack_ptr[0] = 0;

    // Iterate over all codes and encode them into SVO
    for (int i = 0; i < N_leaf; i++) {
        code = codes_data[i];

        // Convert code to insert stack (3bit per level)
        for (uint32_t j = 0; j < depth; j++) {
            insert_stack[j] = (code >> (3*(depth-1-j))) & 0x7;
        }

        // Compare insert stack to stack to determine which level to insert
        if (i == 0) {
            // First code, insert at level 0
            insert_from = 0;
        }
        else {
            // Compare insert stack to stack
            for (insert_from = 0; insert_from < depth-1; insert_from++) {
                if (insert_stack[insert_from] != stack[insert_from]) {
                    break;
                }
            }
        }

        // Insert new nodes from insert_from to depth-1
        for (uint32_t j = insert_from; j < depth; j++) {
            // Add new node to SVO
            if (j > insert_from) {
                svo.push_back(0);
                stack_ptr[j] = svo.size()-1;
            }
            // Update parent pointers
            svo[stack_ptr[j]] |= (1 << insert_stack[j]);
            // Update stack
            if (j < depth-1) {
                stack[j] = insert_stack[j];
            }
        }
    }

    // Convert SVO to tensor
    torch::Tensor svo_tensor = torch::from_blob(svo.data(), {svo.size()}, torch::kUInt8).clone();
    return svo_tensor;
}


void decode_sparse_voxel_octree_cpu_recursive(
    const uint8_t* svo,
    const uint32_t depth,
    uint32_t& ptr,
    std::vector<uint8_t>& stack,
    std::vector<uint32_t>& codes
) {
    uint8_t node = svo[ptr];
    if (stack.size() == depth-1) {
        // Leaf node, add code to list
        uint32_t code = 0;
        for (uint32_t i = 0; i < depth-1; i++) {
            code |= (static_cast<uint32_t>(stack[i]) << (3*(depth-1-i)));
        }
        for (uint8_t i = 0; i < 8; i++) {
            if (node & (1 << i)) {
                code = (code & ~0x7) | i;
                codes.push_back(code);
            }
        }
        ptr++;
    }
    else {
        // Internal node, recurse
        ptr++;
        for (uint8_t i = 0; i < 8; i++) {
            if (node & (1 << i)) {
                stack.push_back(i);
                decode_sparse_voxel_octree_cpu_recursive(svo, depth, ptr, stack, codes);
                stack.pop_back();
            }
        }
    }
}


/**
 * Decode a sparse voxel octree into a list of sparse voxel morton codes
 * 
 * @param octree   uint8 tensor containing the sparse voxel octree
 * @param depth    The depth of the sparse voxel octree
 * 
 * @return         [N] uint32 tensor containing the morton codes
 *                 The codes are sorted in ascending order
 */
torch::Tensor decode_sparse_voxel_octree_cpu(
    const torch::Tensor& octree,
    const uint32_t depth
) {
    uint8_t* octree_data = octree.data_ptr<uint8_t>();
    std::vector<uint32_t> codes;
    std::vector<uint8_t> stack;
    stack.reserve(depth-2);
    uint32_t ptr = 0;
    // Decode SVO into list of codes
    decode_sparse_voxel_octree_cpu_recursive(octree_data, depth, ptr, stack, codes);
    // Convert codes to tensor
    torch::Tensor codes_tensor = torch::from_blob(codes.data(), {codes.size()}, torch::kInt32).clone();
    return codes_tensor;
}


std::tuple<torch::Tensor, torch::Tensor, int64_t> decode_vxzm_records_cpu(
    const torch::Tensor& octree,
    const torch::Tensor& counts,
    const torch::Tensor& records,
    const torch::Tensor& grid_size,
    const torch::Tensor& block_size,
    const uint32_t depth
) {
    TORCH_CHECK(octree.device().is_cpu() && counts.device().is_cpu() &&
                records.device().is_cpu() && grid_size.device().is_cpu() &&
                block_size.device().is_cpu(),
                "VXZM decode inputs must be CPU tensors");
    TORCH_CHECK(octree.scalar_type() == torch::kUInt8 && octree.dim() == 1,
                "octree must be uint8 [S]");
    TORCH_CHECK(counts.scalar_type() == torch::kInt64 && counts.dim() == 1,
                "counts must be int64 [R]");
    TORCH_CHECK(records.scalar_type() == torch::kUInt8 && records.dim() == 2 &&
                records.size(1) >= 3,
                "records must be uint8 [N,C] with C >= 3");
    TORCH_CHECK(grid_size.scalar_type() == torch::kInt64 &&
                block_size.scalar_type() == torch::kInt64 &&
                grid_size.numel() == 3 && block_size.numel() == 3,
                "grid_size and block_size must be int64 [3]");
    TORCH_CHECK(depth >= 2 && depth <= 10,
                "VXZM coarse SVO depth must be in [2,10]");

    const auto octree_c = octree.contiguous();
    const auto counts_c = counts.contiguous();
    const auto records_c = records.contiguous();
    const auto grid_c = grid_size.contiguous();
    const auto block_c = block_size.contiguous();
    const auto* svo = octree_c.data_ptr<uint8_t>();
    const auto svo_size = static_cast<size_t>(octree_c.numel());
    const auto* count_data = counts_c.data_ptr<int64_t>();
    const auto num_regions = static_cast<size_t>(counts_c.numel());
    const auto* grid = grid_c.data_ptr<int64_t>();
    const auto* block = block_c.data_ptr<int64_t>();
    if (svo_size == 0) throw std::invalid_argument("VXZM coarse SVO is empty");
    if (num_regions == 0) throw std::invalid_argument("VXZM coarse SVO must have at least one leaf");
    for (int axis = 0; axis < 3; ++axis) {
        if (grid[axis] <= 0 || block[axis] <= 0 || block[axis] > 256) {
            throw std::invalid_argument("Invalid VXZM grid or region block size");
        }
    }

    // Parse the preorder tree with explicit bounds before reading every node.
    // The legacy decoder is intentionally retained for trusted VXZ data, while
    // externally supplied VXZM bytes use this bounded path.
    std::vector<uint32_t> codes;
    codes.reserve(num_regions);
    size_t ptr = 0;
    std::function<void(uint32_t, uint32_t)> visit =
        [&](uint32_t level, uint32_t prefix) {
            if (ptr >= svo_size) {
                throw std::invalid_argument("Truncated VXZM coarse SVO");
            }
            const uint8_t node = svo[ptr++];
            if (node == 0) {
                throw std::invalid_argument("VXZM coarse SVO contains an empty node");
            }
            if (level == depth - 1) {
                for (uint32_t child = 0; child < 8; ++child) {
                    if (node & (1u << child)) {
                        codes.push_back((prefix << 3) | child);
                        if (codes.size() > num_regions) {
                            throw std::invalid_argument("VXZM coarse SVO has too many leaves");
                        }
                    }
                }
                return;
            }
            for (uint32_t child = 0; child < 8; ++child) {
                if (node & (1u << child)) {
                    visit(level + 1, (prefix << 3) | child);
                }
            }
        };
    visit(0, 0);
    if (ptr != svo_size) {
        throw std::invalid_argument("VXZM coarse SVO contains trailing nodes");
    }
    if (codes.size() != num_regions) {
        throw std::invalid_argument("VXZM coarse SVO leaf/count mismatch");
    }

    auto region_coord = torch::empty(
        {static_cast<int64_t>(num_regions), 3},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* region_out = region_coord.data_ptr<int32_t>();
    for (size_t rid = 0; rid < num_regions; ++rid) {
        uint32_t code = codes[rid];
        uint32_t xyz[3] = {0, 0, 0};
        for (uint32_t level = 0; level < depth; ++level) {
            const uint32_t shift = 3 * (depth - 1 - level);
            const uint32_t child = (code >> shift) & 7u;
            xyz[0] = (xyz[0] << 1) | ((child >> 2) & 1u);
            xyz[1] = (xyz[1] << 1) | ((child >> 1) & 1u);
            xyz[2] = (xyz[2] << 1) | (child & 1u);
        }
        for (int axis = 0; axis < 3; ++axis) {
            region_out[rid * 3 + axis] = static_cast<int32_t>(xyz[axis]);
        }
    }

    int64_t total_records = 0;
    for (size_t rid = 0; rid < num_regions; ++rid) {
        if (count_data[rid] <= 0 ||
            count_data[rid] > std::numeric_limits<int64_t>::max() - total_records) {
            throw std::invalid_argument("Invalid VXZM region record count");
        }
        total_records += count_data[rid];
    }
    if (total_records != records_c.size(0)) {
        throw std::invalid_argument("VXZM records were not fully consumed");
    }

    auto coords = torch::empty(
        {total_records, 3},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* coord_out = coords.data_ptr<int32_t>();
    const auto* record_data = records_c.data_ptr<uint8_t>();
    const int64_t stride = records_c.size(1);
    int64_t out_i = 0;
    int64_t unique_count = 0;
    for (size_t rid = 0; rid < num_regions; ++rid) {
        int32_t previous[3] = {-1, -1, -1};
        for (int64_t local_i = 0; local_i < count_data[rid]; ++local_i, ++out_i) {
            int32_t current[3];
            for (int axis = 0; axis < 3; ++axis) {
                current[axis] = static_cast<int32_t>(record_data[out_i * stride + axis]);
                if (current[axis] >= block[axis]) {
                    throw std::invalid_argument("VXZM local coordinate exceeds its region block");
                }
            }
            if (local_i > 0) {
                const bool out_of_order =
                    current[0] < previous[0] ||
                    (current[0] == previous[0] && current[1] < previous[1]) ||
                    (current[0] == previous[0] && current[1] == previous[1] &&
                     current[2] < previous[2]);
                if (out_of_order) {
                    throw std::invalid_argument("VXZM local coordinates are not canonically ordered");
                }
            }
            if (local_i == 0 || current[0] != previous[0] ||
                current[1] != previous[1] || current[2] != previous[2]) {
                ++unique_count;
            }
            for (int axis = 0; axis < 3; ++axis) {
                const int64_t value =
                    static_cast<int64_t>(region_out[rid * 3 + axis]) * block[axis] +
                    current[axis];
                if (value < 0 || value >= grid[axis] ||
                    value > std::numeric_limits<int32_t>::max()) {
                    throw std::invalid_argument("VXZM reconstructed coordinate lies outside grid_size");
                }
                coord_out[out_i * 3 + axis] = static_cast<int32_t>(value);
                previous[axis] = current[axis];
            }
        }
    }
    return std::make_tuple(coords, region_coord, unique_count);
}
