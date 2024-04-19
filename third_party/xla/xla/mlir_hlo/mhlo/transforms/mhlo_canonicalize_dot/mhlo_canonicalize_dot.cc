/* Copyright 2023 The OpenXLA Authors.

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

// This file implements logic for simplifying HLO dot.

#include <memory>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_HLOCANONICALIZEDOTPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

// Turning mhlo.dot(tensor<1x2xf32>, tensor<2x1xf32>)
// into    mhlo.dot(tensor<2xf32>,   tensor<2xf32>)
LogicalResult canonicalizeDot(DotOp dotOp, PatternRewriter& rewriter) {
  Location loc = dotOp.getLoc();
  auto lhs = dotOp.getLhs();
  auto rhs = dotOp.getRhs();

  SmallVector<ReassociationIndices> reassociationMap = {{0, 1}};
  auto collapseUnitParallelDim = [&](Value orig,
                                     unsigned parallelDimIdx) -> Value {
    auto type = orig.getType().dyn_cast<RankedTensorType>();
    if (!type || type.getRank() == 1 || type.getDimSize(parallelDimIdx) != 1)
      return orig;

    return rewriter.create<tensor::CollapseShapeOp>(
        loc,
        RankedTensorType::get({type.getDimSize(1 - parallelDimIdx)},
                              type.getElementType()),
        orig, reassociationMap);
  };
  auto newLhs = collapseUnitParallelDim(lhs, 0);
  auto newRhs = collapseUnitParallelDim(rhs, 1);

  bool vecmat = newLhs != lhs;
  bool matvec = newRhs != rhs;
  if (!vecmat && !matvec) return failure();

  SmallVector<int64_t> newShape;
  if (!matvec != !vecmat) {
    newShape.push_back(
        vecmat ? rhs.getType().cast<ShapedType>().getShape().back()
               : lhs.getType().cast<ShapedType>().getShape().front());
  }

  auto newTy = RankedTensorType::get(
      newShape, dotOp.getType().cast<ShapedType>().getElementType());

  auto newOp = rewriter.create<DotOp>(
      loc, newTy, newLhs, newRhs, dotOp.getPrecisionConfig().value_or(nullptr));

  rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
      dotOp, dotOp.getType(), newOp,
      (matvec && vecmat) ? SmallVector<ReassociationIndices>()
                         : reassociationMap);
  return success();
}

struct HloCanonicalizeDotPass
    : impl::HloCanonicalizeDotPassBase<HloCanonicalizeDotPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add(canonicalizeDot);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createHloCanonicalizeDotPass() {
  return std::make_unique<HloCanonicalizeDotPass>();
}

}  // namespace mhlo
}  // namespace mlir
