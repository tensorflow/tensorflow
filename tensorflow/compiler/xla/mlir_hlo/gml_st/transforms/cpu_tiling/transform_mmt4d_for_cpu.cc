/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMMMT4DFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

FailureOr<Operation *> tileUsingSCFForAndReplace(
    PatternRewriter &rewriter, Operation *op,
    const scf::SCFTilingOptions &tilingOptions) {
  auto tilingResult = scf::tileUsingSCFForOp(rewriter, op, tilingOptions);
  if (failed(tilingResult) || tilingResult->loops.empty()) return failure();
  rewriter.replaceOp(op, tilingResult->replacements);
  return tilingResult->tiledOps.front();
}

/// Splits the tile sizes in `parallelSizes` into `reductionSizes` for the
/// reduction loops.
void splitParallelAndReductionTiles(linalg::LinalgOp op,
                                    SmallVectorImpl<int64_t> &parallelSizes,
                                    SmallVectorImpl<int64_t> &reductionSizes) {
  reductionSizes.assign(parallelSizes.begin(), parallelSizes.end());
  for (auto [index, iteratorType] :
       llvm::enumerate(op.getIteratorTypesArray())) {
    if (iteratorType == utils::IteratorType::parallel) {
      reductionSizes[index] = 0;
    } else {
      parallelSizes[index] = 0;
    }
  }
}

// We tile towards SIMD codegen, so the tile sizes depend on the target
// architecture (vector instruction sizes, etc.). Luckily, this information is
// already captured in linalg.mmt4d during linalg.matmul -> linalg.mmt4d
// lowering phase. It is hardcoded for AVX on x86 for now.
LogicalResult tileMmt4DOp(linalg::Mmt4DOp mmt4dOp, PatternRewriter &rewriter) {
  if (hasLabel(mmt4dOp, kTransformedLabel)) {
    return rewriter.notifyMatchFailure(mmt4dOp,
                                       "has already been transformed.");
  }

  // Compute the tile sizes. Note that at this stage we only do layout tiling.
  // Later we might also want to do traversal tiling (only on M and N dims).
  auto getL1TileSizes = [&]() -> SmallVector<int64_t> {
    auto lhsShape =
        mmt4dOp.getInputs()[0].getType().cast<ShapedType>().getShape();
    auto rhsShape =
        mmt4dOp.getInputs()[1].getType().cast<ShapedType>().getShape();
    int64_t m0 = lhsShape[2];
    int64_t n0 = rhsShape[2];
    int64_t k0 = lhsShape[3];
    return {1, 1, 1, m0, n0, k0};
  };

  SmallVector<int64_t> parallelTileSizes = getL1TileSizes();
  SmallVector<int64_t> reductionTileSizes;

  // Search the number of outer parallel loops to separate them from possible
  // inner reduction dimensions.
  auto iterTypes = mmt4dOp.getIteratorTypesArray();
  // Make sure to only look at the leading loops for tiling---we will scan
  // this array to find the first non-parallel loop later and use that for
  // indexing into the tile sizes.
  if (iterTypes.size() > parallelTileSizes.size()) {
    iterTypes.resize(parallelTileSizes.size());
  }

  splitParallelAndReductionTiles(mmt4dOp.getOperation(), parallelTileSizes,
                                 reductionTileSizes);

  // Tile the parallel loops.
  auto tiledOp = tileUsingSCFForAndReplace(
      rewriter, mmt4dOp.getOperation(),
      scf::SCFTilingOptions().setTileSizes(parallelTileSizes));
  if (failed(tiledOp)) return failure();
  mmt4dOp = cast<linalg::Mmt4DOp>(*tiledOp);

  // Tile the reduction loops.
  tiledOp = tileUsingSCFForAndReplace(
      rewriter, mmt4dOp.getOperation(),
      scf::SCFTilingOptions().setTileSizes(reductionTileSizes));
  if (failed(tiledOp)) return failure();
  mmt4dOp = cast<linalg::Mmt4DOp>(*tiledOp);

  setLabel(mmt4dOp, kTransformedLabel);
  return success();
}

struct TransformMmt4DForCpuPass
    : public impl::TransformMmt4DForCpuPassBase<TransformMmt4DForCpuPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add(tileMmt4DOp);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMmt4DForCpuPass() {
  return std::make_unique<TransformMmt4DForCpuPass>();
}

}  // namespace mlir::gml_st
