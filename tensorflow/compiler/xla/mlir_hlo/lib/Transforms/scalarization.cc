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

#include <memory>
#include <utility>

#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

#define GEN_PASS_DEF_SCALARIZATIONPASS
#include "mlir-hlo/Transforms/passes.h.inc"

using linalg::GenericOp;

bool hasSingleElement(RankedTensorType type) {
  return type.hasStaticShape() && type.getNumElements() == 1;
}

class ScalarizeGenericOp : public OpRewritePattern<GenericOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(genericOp->getResultTypes(), [](Type resultType) {
          return !hasSingleElement(resultType.cast<RankedTensorType>());
        }))
      return failure();

    // Map block arguments of genericOp to tensor.extract ops of its args.
    Location loc = genericOp.getLoc();
    BlockAndValueMapping bvm;
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      Value operandValue = opOperand.get();
      Type operandType = operandValue.getType();
      auto bbArg = genericOp.getTiedBlockArgument(&opOperand);
      if (!operandType.isa<ShapedType>()) {
        bvm.map(bbArg, operandValue);
        continue;
      }
      auto tensorType = operandType.dyn_cast<RankedTensorType>();
      if (!tensorType || !hasSingleElement(tensorType)) return failure();

      SmallVector<Value> indices(
          tensorType.getRank(),
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
      Value extractedElement =
          rewriter.create<tensor::ExtractOp>(loc, operandValue, indices);
      bvm.map(bbArg, extractedElement);
    }

    // Clone everything but terminator.
    Block *body = genericOp.getBody();
    for (Operation &op : body->without_terminator()) rewriter.clone(op, bvm);

    // Wrap every scalar result into a tensor using `tensor.from_elements`.
    SmallVector<Value> newResults;
    for (auto [resultType, yieldOperand] :
         llvm::zip(genericOp->getResultTypes(),
                   body->getTerminator()->getOperands())) {
      auto scalarValue = bvm.lookup(yieldOperand);
      newResults.push_back(rewriter.create<tensor::FromElementsOp>(
          loc, resultType, scalarValue));
    }
    rewriter.replaceOp(genericOp, newResults);

    return success();
  }
};

struct ScalarizationPass
    : public impl::ScalarizationPassBase<ScalarizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ScalarizeGenericOp>(context);
    tensor::FromElementsOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createScalarizationPass() {
  return std::make_unique<ScalarizationPass>();
}

}  // namespace mlir
