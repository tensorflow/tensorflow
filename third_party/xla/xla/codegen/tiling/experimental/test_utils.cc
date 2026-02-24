/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/tiling/experimental/test_utils.h"

#include <cstdint>
#include <utility>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/symbolic_tile.h"

namespace xla::gpu::experimental {

using ::llvm::SmallVector;
using ::mlir::getAffineDimExpr;
using ::mlir::getAffineSymbolExpr;
using ::mlir::MLIRContext;

SymbolicTile GetTestSymbolicTile(const TilingSpace& tiling_space,
                                 absl::Span<const int64_t> shape) {
  MLIRContext* mlir_context = tiling_space.mlir_context();
  CHECK(mlir_context != nullptr);
  int64_t rank = shape.size();
  SmallVector<DimTile> dim_tiles;
  dim_tiles.reserve(rank);
  for (auto [index, dim] : llvm::enumerate(shape)) {
    auto tid = getAffineDimExpr(index, mlir_context);
    auto ts = getAffineSymbolExpr(index, mlir_context);
    dim_tiles.push_back(DimTile{
        tid * ts, ts, mlir::getAffineConstantExpr(index + 1, mlir_context),
        mlir::getAffineConstantExpr(dim, mlir_context)});
  }
  return SymbolicTile{tiling_space, std::move(dim_tiles)};
}

}  // namespace xla::gpu::experimental
