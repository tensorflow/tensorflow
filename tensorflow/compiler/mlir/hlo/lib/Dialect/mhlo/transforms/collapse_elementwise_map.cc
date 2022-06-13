/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

// TODO(b/228448038): consider to move this pattern to mhlo.map canonicalizer.
// Pattern to convert map of pure elementwise ops to directly use elementwise
// ops without map. e.g.
//   %0 = "mhlo.map"(%arg, %arg1) ({
//   ^bb0(%a: tensor<f32>, %b: tensor<f32>):
//     %output = mhlo.add %a, %b : tensor<f32>
//     "mhlo.return"(%output) : (tensor<f32>) -> ()
//   }) {dimensions = dense<[0]> : tensor<1xi64>} :
//   (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// To:
//   %0 = mhlo.add %arg, %arg1 : tensor<?xf32>
struct ConvertMapOfElementwiseOps : public OpRewritePattern<MapOp> {
  using OpRewritePattern<MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp map,
                                PatternRewriter &rewriter) const override {
    // Matches that the computation block only has element-wise ops.
    if (llvm::any_of(map.computation().front().without_terminator(),
                     [](Operation &op) {
                       return op.getNumResults() != 1 ||
                              !op.hasTrait<::mlir::OpTrait::Elementwise>();
                     })) {
      return failure();
    }

    rewriter.setInsertionPointAfter(map);
    BlockAndValueMapping blockAndValueMap;
    for (mlir::BlockArgument barg : map.computation().front().getArguments()) {
      blockAndValueMap.map(barg, map->getOperand(barg.getArgNumber()));
    }
    auto shape = map.getType().getShape();
    for (Operation &op : map.computation().front().without_terminator()) {
      SmallVector<Value, 2> operands;
      // Remaps the operands.
      operands.reserve(op.getNumOperands());
      for (auto value : op.getOperands())
        operands.push_back(blockAndValueMap.lookup(value));
      auto newOp = rewriter.create(
          op.getLoc(), op.getName().getIdentifier(), operands,
          op.getResultTypes()[0].cast<TensorType>().clone(shape));
      // Maps the result.
      blockAndValueMap.map(op.getResult(0), newOp->getResult(0));
    }

    auto retOp = cast<ReturnOp>(map.computation().front().back());
    map->getResult(0).replaceAllUsesWith(
        blockAndValueMap.lookup(retOp->getOperand(0)));
    return success();
  }
};

struct CollapseElementwiseMapPass
    : public CollapseElementwiseMapPassBase<CollapseElementwiseMapPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMapOfElementwiseOps>(ctx);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createCollapseElementwiseMapPass() {
  return std::make_unique<CollapseElementwiseMapPass>();
}

}  // namespace mhlo
}  // namespace mlir
