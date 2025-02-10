/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/symbolic_tile.h"

namespace xla {
namespace gpu {

llvm::SmallVector<int64_t> SymbolicTiledHloInstruction::TileOffsets(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateAffineMap(symbolic_tile().offset_map(),
                           /*dim_values=*/tile_parameters);
}

llvm::SmallVector<int64_t> SymbolicTiledHloInstruction::TileSizes(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateAffineMap(symbolic_tile().size_map(),
                           /*dim_values=*/tile_parameters);
}

llvm::SmallVector<int64_t> SymbolicTiledHloInstruction::TileStrides(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateAffineMap(symbolic_tile().stride_map(),
                           /*dim_values=*/tile_parameters);
}

std::string SymbolicTiledHloInstruction::ToString() const {
  std::stringstream ss;
  ss << "\thlo: " << hlo_->ToString() << "\n";
  ss << "\t" << symbolic_tile().ToString() << "\n";
  ss << "\tindexing map: " << indexing_map_ << "\n";
  return ss.str();
}

}  // namespace gpu
}  // namespace xla
