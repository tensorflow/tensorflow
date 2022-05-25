/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This files implements the logic for converting `scf.parallel` loops into
// tiled loops.

#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

namespace mlir {

using ::llvm::to_vector;
using ::mlir::scf::ParallelOp;

namespace {

// This is the implementation of the TileLoops pass declared in
//  include/mlir-hlo/Transforms/passes.td
class TileLoopsPass : public TileLoopsPassBase<TileLoopsPass> {
 public:
  // Creates a TileLoopsPass with tiles sizes provided through `tile_sizes`
  // and unroll factors provided through `unroll_factors`.
  explicit TileLoopsPass(ArrayRef<int64_t> tile_sizes,
                         ArrayRef<int64_t> unroll_factors) {
    tile_sizes_ = tile_sizes;
    unroll_factors_ = unroll_factors;
  }

  void runOnOperation() override;
};

}  // namespace

// Returns whether the access pattern in `ploop` is "complex". That is, whether
// any memref.load op in its region uses indices that don't correspond to the
// loop induction variables.
static bool IsComplexAccessPattern(ParallelOp ploop) {
  auto is_complex = [&](memref::LoadOp load_op) {
    if (!load_op.getMemRefType().getLayout().isIdentity()) return true;
    if (load_op.getIndices().empty()) return false;
    return load_op.getIndices() != ploop.getInductionVars();
  };
  return llvm::any_of(ploop.getBody()->getOps<memref::LoadOp>(), is_complex);
}

void TileLoopsPass::runOnOperation() {
  auto unrolled_tile = [&]() -> SmallVector<int64_t, 4> {
    if (tile_sizes_.size() != unroll_factors_.size()) return {};
    auto multiply = [](std::tuple<int64_t, int64_t> tuple) {
      return std::get<0>(tuple) * std::get<1>(tuple);
    };
    return to_vector<4>(
        llvm::map_range(llvm::zip(tile_sizes_, unroll_factors_), multiply));
  }();

  SmallVector<ParallelOp, 2> innermostPloops;
  getInnermostParallelLoops(this->getOperation().getOperation(),
                            innermostPloops);

  for (ParallelOp ploop : innermostPloops) {
    // Do not unroll if the multiplier has the wrong rank, or if we have complex
    // memory access patterns.
    if (unrolled_tile.empty() || IsComplexAccessPattern(ploop)) {
      tileParallelLoop(ploop, tile_sizes_, /*noMinMaxBounds=*/false);
      continue;
    }
    auto tiled_loops =
        tileParallelLoop(ploop, unrolled_tile, /*noMinMaxBounds=*/false);
    tileParallelLoop(tiled_loops.second, unroll_factors_,
                     /*noMinMaxBounds=*/false);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTileLoopsPass(
    ArrayRef<int64_t> tile_sizes, ArrayRef<int64_t> unroll_factors) {
  return std::make_unique<TileLoopsPass>(tile_sizes, unroll_factors);
}

}  // namespace mlir
