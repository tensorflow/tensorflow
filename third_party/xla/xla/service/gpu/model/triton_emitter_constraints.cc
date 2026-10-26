/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/model/triton_emitter_constraints.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "xla/codegen/tiling/tiled_hlo.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/decision.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu::experimental {

namespace {

// Triton enforces that all tensors in the program have less than 1048576
// elements, otherwise it will fail to compile. (See `TRITON_MAX_TENSOR_NUMEL`
// in the Triton codebase.)
constexpr int64_t kMaxTensorNumElements = 1048576;
// For dot operations we don't want to tile the contracting dimension to a size
// larger than this as that leads to a large number of registers being used.
// Also for MMA instruction we don't want tiles greater than 256.
constexpr int64_t kMaxMMADimSize = 256;

// Returns a conservative estimate (in bytes) of the memory required to stage a
// tile whose sizes are `tile_sizes` for an instruction whose element type
// occupies `element_byte_size` bytes.
int64_t GetPaddedTileSizeInBytes(absl::Span<const int64_t> tile_sizes,
                                 int64_t element_byte_size) {
  int64_t padded_tile_elements = 1;
  for (int64_t size : tile_sizes) {
    padded_tile_elements *= llvm::PowerOf2Ceil(size);
  }
  return padded_tile_elements * element_byte_size;
}

void CollectAllInstructions(
    const TiledHloInstruction* inst,
    llvm::SmallVector<const TiledHloInstruction*>& out) {
  out.push_back(inst);
  for (const auto& region : inst->hlo_regions()) {
    for (const TiledHloInstruction* sub_inst : region.instructions()) {
      CollectAllInstructions(sub_inst, out);
    }
  }
}

llvm::SmallVector<const TiledHloInstruction*> CollectAllInstructions(
    const TiledHloComputation& tiled_computation) {
  llvm::SmallVector<const TiledHloInstruction*> all_instructions;
  for (const TiledHloInstruction* inst : tiled_computation.instructions()) {
    CollectAllInstructions(inst, all_instructions);
  }
  return all_instructions;
}

}  // namespace

Decision VerifyTritonConstraints(const TiledHloComputation& tiled_computation,
                                 const se::DeviceDescription& device_info) {
  llvm::SmallVector<const TiledHloInstruction*> all_instructions =
      CollectAllInstructions(tiled_computation);

  // 1. Max Tensor Elements (TRITON_MAX_TENSOR_NUMEL limit in triton.language).
  // Triton's validate_block_shape (triton/python/triton/_utils.py) enforces
  // that the total number of elements in a padded block (product of tile sizes
  // padded to power of 2) does not exceed TRITON_MAX_TENSOR_NUMEL = 1048576.
  for (const TiledHloInstruction* inst : all_instructions) {
    auto static_tile_sizes_or = inst->tile().GetStaticTileSizes();
    if (!static_tile_sizes_or.ok()) {
      return Decision(static_tile_sizes_or.status());
    }
    llvm::SmallVector<int64_t> static_tile_sizes =
        std::move(*static_tile_sizes_or);

    int64_t product = 1;
    for (int64_t size : static_tile_sizes) {
      CHECK_GT(size, 0) << inst->hlo()->name() << " has non-positive tile size "
                        << size;
      int64_t padded_size = llvm::PowerOf2Ceil(size);
      if (padded_size > kMaxTensorNumElements / product) {
        return Decision::Forbid(absl::StrCat(
            "Instruction ", inst->hlo()->name(),
            " has a padded tile size product that exceeds the maximum allowed "
            "tensor size of ",
            kMaxTensorNumElements, " elements."));
      }
      product *= padded_size;
    }
  }

  // 2. MMA Limit.
  // Triton's GPU codegen constrains dot operands to avoid excessive register
  // pressure and comply with hardware MMA layout constraints. Contracting and
  // free dimension tile sizes are limited to kMaxMMADimSize.
  // TODO(b/393299275): There is also a lower bound limit for Triton on what
  // is accepted for dimension size (both contracting and free).
  for (const TiledHloInstruction* inst : all_instructions) {
    if (inst->hlo()->opcode() == HloOpcode::kDot) {
      for (const TiledHloInstruction* operand : inst->operands()) {
        auto op_tile_sizes_or = operand->tile().GetStaticTileSizes();
        if (!op_tile_sizes_or.ok()) {
          return Decision(op_tile_sizes_or.status());
        }
        llvm::SmallVector<int64_t> op_tile_sizes = std::move(*op_tile_sizes_or);

        for (int64_t size : op_tile_sizes) {
          if (size > kMaxMMADimSize) {
            return Decision::Forbid(absl::StrCat(
                "Dot operand instruction ", operand->hlo()->name(),
                " has a tile size of ", size,
                ", which exceeds the maximum MMA dimension size of ",
                kMaxMMADimSize, "."));
          }
        }
      }
    }
  }

  // 3. Transpose Shared Memory Limit.
  // A transpose op stages its (padded) operand tile in shared memory to perform
  // the layout conversion (the classic shared-memory transpose that avoids
  // uncoalesced global memory accesses). Estimate that shared-memory usage as
  // the product of the power-of-2-padded operand tile sizes multiplied by the
  // element byte size, and reject tiles that would exceed the device shared
  // memory limit. This lets the tile search fall back to a smaller tile instead
  // of failing later with a RESOURCE_EXHAUSTED error during Triton compilation.
  const int64_t shared_memory_limit =
      device_info.shared_memory_per_block_optin();
  for (const TiledHloInstruction* inst : all_instructions) {
    if (inst->hlo()->opcode() != HloOpcode::kTranspose ||
        inst->operands().empty()) {
      continue;
    }
    // The transposed operand (operand 0) is the tile that is staged in shared
    // memory.
    CHECK_EQ(inst->operands().size(), 1)
        << "Transpose " << inst->hlo()->name() << " should have exactly one "
        << "operand, but has " << inst->operands().size() << ".";
    const TiledHloInstruction* operand = inst->operands().front();
    auto operand_tile_sizes_or = operand->tile().GetStaticTileSizes();
    if (!operand_tile_sizes_or.ok()) {
      return Decision(operand_tile_sizes_or.status());
    }

    const int64_t shared_memory_bytes = GetPaddedTileSizeInBytes(
        *operand_tile_sizes_or, ShapeUtil::ByteSizeOfPrimitiveType(
                                    operand->hlo()->shape().element_type()));
    if (shared_memory_bytes > shared_memory_limit) {
      return Decision::Forbid(absl::StrCat(
          "Transpose instruction ", inst->hlo()->name(),
          " has an operand tile that requires an estimated ",
          shared_memory_bytes,
          " bytes of shared memory, which exceeds the device limit of ",
          shared_memory_limit, " bytes."));
    }
  }

  VLOG(2) << "VerifyTritonConstraints: checking roots. Count: "
          << tiled_computation.roots().size();
  for (const TiledHloInstruction* root : tiled_computation.roots()) {
    VLOG(2) << "Root: " << root->hlo()->name()
            << ", shape: " << root->hlo()->shape().ToString();
    if (!root->hlo()->shape().IsArray()) {
      return Decision::Forbid(absl::StrCat(
          "Root instruction ", root->hlo()->name(), " has a non-array shape."));
    }
    auto root_tile_sizes_or = root->tile().GetStaticTileSizes();
    if (!root_tile_sizes_or.ok()) {
      return Decision(root_tile_sizes_or.status());
    }
    llvm::SmallVector<int64_t> root_tile_sizes = std::move(*root_tile_sizes_or);

    VLOG(2) << "Root tile sizes: [" << absl::StrJoin(root_tile_sizes, ", ")
            << "]";
    auto dim_sizes = root->hlo()->shape().dimensions();
    if (root_tile_sizes.size() != dim_sizes.size()) {
      return Decision::Forbid(absl::StrCat(
          "Root instruction ", root->hlo()->name(), " has ",
          root_tile_sizes.size(), " tile sizes, but its HLO shape has ",
          dim_sizes.size(), " dimensions."));
    }

    // Verify that the tile sizes are either powers of 2 or equal to the
    // dimension size.
    // TODO(b/365727080): Technically the tile size should always be a power of
    // 2, but currently if we capture a dimension fully, we use the dimension
    // size as tile size.
    for (size_t i = 0; i < root_tile_sizes.size(); ++i) {
      int64_t t = root_tile_sizes[i];
      CHECK_GT(t, 0) << root->hlo()->name() << " has non-positive tile size "
                     << t << " in dimension " << i;
      int64_t d = dim_sizes[i];
      if (!(llvm::isPowerOf2_64(t) || t == d)) {
        return Decision::Forbid(absl::StrCat(
            "Root instruction ", root->hlo()->name(), " has a tile size of ", t,
            " in dimension ", i,
            ", which is neither a power of 2 nor equal to the dimension size ",
            d, "."));
      }
    }

    // Check that the number of blocks fits on the device grid X.
    // TODO(b/418965008): persistent kernels will make this check obsolete.
    uint64_t num_blocks = 1;
    uint64_t limit = device_info.block_dim_limit().x;
    for (size_t i = 0; i < root_tile_sizes.size(); ++i) {
      uint64_t tile_size = static_cast<uint64_t>(root_tile_sizes[i]);
      uint64_t dim_size = static_cast<uint64_t>(dim_sizes[i]);
      uint64_t blocks_in_dim = CeilOfRatio(dim_size, tile_size);
      if (blocks_in_dim > limit || (num_blocks > limit / blocks_in_dim)) {
        return Decision::Forbid(
            absl::StrCat("Number of blocks ", num_blocks, "*", blocks_in_dim,
                         " exceeds the device grid X limit of ", limit, "."));
      }
      num_blocks *= blocks_in_dim;
    }
  }

  return Decision::Allow();
}

Decision VerifySubsetOfTritonConstraints(
    absl::Span<const int64_t> padded_tile_sizes,
    const TilingSpace& tiling_space, const se::DeviceDescription& device_info) {
  auto dimensions = tiling_space.dimensions();

  // Hardware MMA limits for Dot / ScaledDot instructions.
  // TODO(b/393299275): There is also a lower bound limit for Triton on what
  // is accepted for dimension size (both contracting and free).
  for (const auto& [idx, dim] : llvm::enumerate(dimensions)) {
    if (dim.hlo != nullptr && (dim.hlo->opcode() == HloOpcode::kDot ||
                               dim.hlo->opcode() == HloOpcode::kScaledDot)) {
      if (padded_tile_sizes[idx] > kMaxMMADimSize) {
        return Decision::Forbid(absl::StrCat(
            "Tile size ", padded_tile_sizes[idx], " for dimension ", idx,
            " of ", dim.hlo->name(), " exceeds the maximum MMA dimension size ",
            kMaxMMADimSize, "."));
      }
    }
  }

  // Group parallel dimensions by root instruction to evaluate constraints per
  // tiled root (supporting multi-output fusion).
  llvm::SmallVector<llvm::SmallVector<size_t, 4>, 2> root_dim_groups;
  const HloInstruction* current_root = nullptr;
  for (const auto& [idx, dim] : llvm::enumerate(dimensions)) {
    if (dim.type == TilingSpace::DimensionSemantics::kParallel) {
      if (root_dim_groups.empty() || dim.hlo != current_root) {
        root_dim_groups.push_back({});
        current_root = dim.hlo;
      }
      root_dim_groups.back().push_back(idx);
    }
  }

  const uint64_t grid_limit = device_info.block_dim_limit().x;

  for (const auto& dim_indices : root_dim_groups) {
    int64_t root_product = 1;
    uint64_t num_blocks = 1;

    for (size_t idx : dim_indices) {
      const auto& dim = dimensions[idx];
      int64_t tile_size = padded_tile_sizes[idx];

      // Overflow check on cumulative root product.
      if (tile_size > kMaxTensorNumElements / root_product) {
        return Decision::Forbid(absl::StrCat(
            "Padded tile size product exceeds the maximum number of elements "
            "of a Triton tensor (",
            kMaxTensorNumElements, ")."));
      }
      root_product *= tile_size;

      // Launch grid block limits.
      // TODO(b/418965008): persistent kernels will make this check obsolete.
      uint64_t blocks_in_dim =
          CeilOfRatio<uint64_t>(static_cast<uint64_t>(dim.dimension_size),
                                static_cast<uint64_t>(tile_size));
      if (blocks_in_dim == 0 || blocks_in_dim > grid_limit ||
          (num_blocks > grid_limit / blocks_in_dim)) {
        return Decision::Forbid(absl::StrCat(
            "Number of blocks ", num_blocks, "*", blocks_in_dim,
            " exceeds the device grid X limit of ", grid_limit, "."));
      }
      num_blocks *= blocks_in_dim;
    }
  }

  return Decision::Allow();
}

}  // namespace gpu::experimental

}  // namespace xla
