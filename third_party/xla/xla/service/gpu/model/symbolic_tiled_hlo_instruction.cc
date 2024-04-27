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
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/status.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::SmallVector;

std::vector<int64_t> EvaluateTileMap(AffineMap affine_map,
                                     absl::Span<int64_t const> parameters) {
  CHECK_EQ(affine_map.getNumSymbols(), parameters.size());
  CHECK_EQ(affine_map.getNumDims(), 0);

  SmallVector<AffineExpr> symbol_replacements = llvm::to_vector(
      llvm::map_range(parameters, [affine_map](const int64_t v) -> AffineExpr {
        return mlir::getAffineConstantExpr(v, affine_map.getContext());
      }));

  AffineMap simplified_affine_map =
      mlir::simplifyAffineMap(affine_map.replaceDimsAndSymbols(
          /*dimReplacements=*/{}, symbol_replacements, /*numResultDims=*/0,
          /*numResultSyms=*/0));

  SmallVector<int64_t> results = llvm::to_vector(llvm::map_range(
      simplified_affine_map.getResults(), [](AffineExpr result) -> int64_t {
        return llvm::cast<mlir::AffineConstantExpr>(result).getValue();
      }));

  return std::vector<int64_t>(results.begin(), results.end());
}

}  // namespace

std::vector<int64_t> SymbolicTiledHloInstruction::TileOffsets(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateTileMap(symbolic_tile_.offset_map(), tile_parameters);
}

std::vector<int64_t> SymbolicTiledHloInstruction::TileSizes(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateTileMap(symbolic_tile_.size_map(), tile_parameters);
}

std::vector<int64_t> SymbolicTiledHloInstruction::TileStrides(
    absl::Span<int64_t const> tile_parameters) const {
  return EvaluateTileMap(symbolic_tile_.stride_map(), tile_parameters);
}

}  // namespace gpu
}  // namespace xla
