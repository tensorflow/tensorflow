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

#include <utility>

#include "mhlo_tosa/Transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "passes_detail.h"

#define PASS_NAME "tosa-legalize-tf"
#define DEBUG_TYPE PASS_NAME

#include "mhlo_tosa/Transforms/legalize_mhlo.pdll.h.inc"

namespace mlir {
namespace tosa {
namespace {

struct LegalizeMhlo : TosaLegalizeMhloPassBase<LegalizeMhlo> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext* ctx) override;

 private:
  FrozenRewritePatternSet patterns;
};

struct ConvertMhloReduceOp : public OpRewritePattern<mhlo::ReduceOp> {
  using OpRewritePattern<mhlo::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Block& bodyBlock = op.body().front();

    // To lower to a tosa.reduce_* op, the body should contain the reduce op and
    // a return op.
    if (bodyBlock.getOperations().size() != 2) {
      return rewriter.notifyMatchFailure(op, "body required to contain 2 ops");
    }

    auto operands = op.operands().front();
    ShapedType inputType = operands.getType().cast<ShapedType>();
    uint64_t dimension = op.dimensions().getValues<uint64_t>().begin()[0];
    Operation& innerOp = bodyBlock.front();
    Value reduceOpResult;

    if (isa<mhlo::AddOp>(innerOp)) {
      reduceOpResult =
          rewriter
              .create<tosa::ReduceSumOp>(op->getLoc(), inputType, operands,
                                         rewriter.getI64IntegerAttr(dimension))
              .getResult();
    } else if (isa<mhlo::MaxOp>(innerOp)) {
      reduceOpResult =
          rewriter
              .create<tosa::ReduceMaxOp>(op->getLoc(), inputType, operands,
                                         rewriter.getI64IntegerAttr(dimension))
              .getResult();
    } else {
      return rewriter.notifyMatchFailure(
          op, "reducing along a " + innerOp.getName().getStringRef().str() +
                  " op not supported");
    }

    // TOSA reduce ops do not remove the dimension being reduced, so reshape the
    // reduced output and remove the reduction dimension.
    ArrayRef<int64_t> innerShape = inputType.getShape();
    llvm::SmallVector<int64_t, 2> outputShape;
    int outputShapeLength = innerShape.size() - 1;
    outputShape.resize(outputShapeLength);
    for (int i = 0; i < outputShapeLength; i++) {
      if (i < dimension) {
        outputShape[i] = innerShape[i];
      } else {
        outputShape[i] = innerShape[i + 1];
      }
    }

    rewriter
        .replaceOpWithNewOp<tosa::ReshapeOp>(
            op, op.getResultTypes().front(), reduceOpResult,
            rewriter.getI64ArrayAttr(outputShape))
        .getResult();

    return success();
  }
};

LogicalResult LegalizeMhlo::initialize(MLIRContext* ctx) {
  RewritePatternSet patternList(ctx);
  populateGeneratedPDLLPatterns(patternList);
  patternList.addWithLabel<ConvertMhloReduceOp>({"MhloReduce"}, ctx);
  patterns = std::move(patternList);
  return success();
}

void LegalizeMhlo::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeMhloPass() {
  return std::make_unique<LegalizeMhlo>();
}

}  // namespace tosa
}  // namespace mlir
