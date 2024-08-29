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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineMap.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

namespace {

// Triton enforces that all tensors in the program have less than 1048576
// elements, otherwise it will fail to compile.
constexpr int64_t kMaxTensorNumElements = 1048576;

}  // namespace

/*static*/ EmitterSpecificConstraintsBuilder
TritonEmitterConstraints::GetBuilder(
    const se::DeviceDescription& device_description) {
  return [=](const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
                 instructions) {
    llvm::DenseSet<mlir::AffineMap> unique_tile_size_maps;
    for (const auto& tiled_hlo_instruction : instructions) {
      unique_tile_size_maps.insert(
          tiled_hlo_instruction->symbolic_tile().size_map());
    }

    llvm::SmallVector<mlir::AffineMap, 4> tile_size_maps(
        unique_tile_size_maps.begin(), unique_tile_size_maps.end());

    return std::make_unique<TritonEmitterConstraints>(
        std::move(tile_size_maps),
        /*root_shape=*/instructions.back()->hlo()->shape(), device_description);
  };
}

absl::StatusOr<bool> TritonEmitterConstraints::ParametersSatisfyConstraints(
    absl::Span<const int64_t> tile_parameters) const {
  // Verify that the tile sizes are not too big.
  for (const auto& tile_size_map : tile_size_maps_) {
    int64_t tile_size = 1;
    for (auto expr : tile_size_map.getResults()) {
      tile_size *= llvm::PowerOf2Ceil(
          EvaluateAffineExpr(expr, /*dim_values=*/tile_parameters));
    }

    if (tile_size > kMaxTensorNumElements) {
      return false;
    }
  }

  int64_t num_tiles = 1;
  for (auto [dim_size, tile_size] :
       llvm::zip(root_shape_.dimensions(), tile_parameters)) {
    num_tiles *= (dim_size + tile_size - 1) / tile_size;
  }

  // Number of blocks will excede the hardware limit. This limitation comes from
  // the fact that one tile is mapped to one block. This constraint can be
  // potentially hoisted to more generic "gpu-specific constraint".
  if (num_tiles >= device_info_.block_dim_limit().x) {
    return false;
  }

  return true;
}

}  // namespace gpu
}  // namespace xla
