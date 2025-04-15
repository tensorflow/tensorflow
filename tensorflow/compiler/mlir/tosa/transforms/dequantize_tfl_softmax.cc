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

#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-dequantize_tfl-softmax"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {

#define GEN_PASS_DEF_TOSADEQUANTIZETFLSOFTMAXPASS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

class TosaDequantizeTFLSoftmax
    : public impl::TosaDequantizeTFLSoftmaxPassBase<TosaDequantizeTFLSoftmax> {
 public:
  explicit TosaDequantizeTFLSoftmax() = default;
  void runOnOperation() override;
};

struct TosaDequantizeTFLSoftmaxPattern : public RewritePattern {
  explicit TosaDequantizeTFLSoftmaxPattern(MLIRContext* context)
      : RewritePattern(TFL::SoftmaxOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult TosaDequantizeTFLSoftmaxPattern::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  TFL::SoftmaxOp tfl_softmax_op = cast<TFL::SoftmaxOp>(op);
  RankedTensorType input_type =
      mlir::cast<RankedTensorType>(tfl_softmax_op.getInput().getType());
  if (!mlir::isa<mlir::quant::QuantizedType>(input_type.getElementType())) {
    return failure();
  }
  Location loc = tfl_softmax_op.getLoc();
  RankedTensorType dequantized_input_type =
      RankedTensorType::get(input_type.getShape(), rewriter.getF32Type());
  Value dequantized_input = rewriter.create<TFL::DequantizeOp>(
      loc, dequantized_input_type, tfl_softmax_op.getInput());
  Value dequantized_softmax_output = rewriter.create<TFL::SoftmaxOp>(
      loc, dequantized_input_type, dequantized_input, tfl_softmax_op.getBeta());
  Type qtype = tfl_softmax_op.getOutput().getType();
  rewriter.replaceOpWithNewOp<TFL::QuantizeOp>(tfl_softmax_op, qtype,
                                               dequantized_softmax_output,
                                               mlir::TypeAttr::get(qtype));
  return success();
}

void TosaDequantizeTFLSoftmax::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<TosaDequantizeTFLSoftmaxPattern>(&getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDequantizeTFLSoftmaxPass() {
  return std::make_unique<TosaDequantizeTFLSoftmax>();
}

}  // namespace tosa

}  // namespace mlir
