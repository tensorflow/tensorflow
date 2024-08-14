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
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineMap.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"

namespace xla {
namespace gpu {

namespace {

// Triton enforces that all tensors in the program have less than 1048576
// elements, otherwise it will fail to compile.
constexpr int64_t kMaxTensorNumElements = 1048576;

}  // namespace

/*static*/ EmitterSpecificConstraintsBuilder
TritonEmitterConstraints::GetBuilder() {
  return [](const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
                instructions) {
    llvm::DenseSet<mlir::AffineMap> unique_tile_size_maps;
    for (const auto& tiled_hlo_instruction : instructions) {
      unique_tile_size_maps.insert(
          tiled_hlo_instruction->symbolic_tile().size_map());
    }

    return std::make_unique<TritonEmitterConstraints>(
        llvm::SmallVector<mlir::AffineMap, 4>(unique_tile_size_maps.begin(),
                                              unique_tile_size_maps.end()));
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
  return true;
}

}  // namespace gpu
}  // namespace xla
