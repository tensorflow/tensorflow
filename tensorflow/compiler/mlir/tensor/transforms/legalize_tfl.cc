/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Legalize TensorFlow Lite to Tensor

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensor/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tensor/transforms/passes.h"


namespace mlir {
namespace tensor {

namespace {

#define GEN_PASS_DEF_TENSORLEGALIZETFLPASS
#include "tensorflow/compiler/mlir/tensor/transforms/passes.h.inc"

class LegalizeTFL : public impl::TensorLegalizeTFLPassBase<LegalizeTFL> {
public:
  LegalizeTFL() = default;
  void runOnOperation() override;
};

#define DECL_CONVERT_OP(tfl_op)                                              \
  struct ConvertTFL##tfl_op##Op : public RewritePattern {                    \
    explicit ConvertTFL##tfl_op##Op(MLIRContext* context)                    \
        : RewritePattern(TFL::tfl_op##Op::getOperationName(), 1, context) {} \
    LogicalResult matchAndRewrite(Operation* op,                             \
                                  PatternRewriter& rewriter) const override; \
  }

DECL_CONVERT_OP(Reshape);

LogicalResult ConvertTFLReshapeOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tfl_reshape_op = cast<TFL::ReshapeOp>(op);
  Value input = tfl_reshape_op.getInput();
  Value shape = tfl_reshape_op.getShape();
  Type result_type = tfl_reshape_op.getResult().getType();

  // If shape is unranked, cast it to a 1D tensor
  if (!shape.getType().isa<RankedTensorType>()) {
    auto element_type = shape.getType().cast<TensorType>().getElementType();
    auto ranked_shape_type = RankedTensorType::get({ShapedType::kDynamic}, element_type);
    shape = rewriter.create<tensor::CastOp>(op->getLoc(), ranked_shape_type, shape);
  }

  // Substitute a possible value set to -1 in the target shape
  shape = substituteShapeWildcard(rewriter, op->getLoc(), input, shape);

  // Translate op
  rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(tfl_reshape_op, result_type,
                                                 input, shape);
  return success();
}

void LegalizeTFL::runOnOperation() {
  auto* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  populateLegalizeTFLPatterns(ctx, patterns);

  auto func = getOperation();
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

void populateLegalizeTFLPatterns(MLIRContext* ctx,
                                 RewritePatternSet& patterns) {
#define ADD_PATTERN(pattern) \
  patterns.addWithLabel<ConvertTFL##pattern##Op>({#pattern}, ctx);

  ADD_PATTERN(Reshape);

#undef ADD_PATTERN
}

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFLPass() {
  return std::make_unique<LegalizeTFL>();
}

}  // namespace tensor
}  // namespace mlir

