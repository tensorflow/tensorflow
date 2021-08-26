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

// Fuse tf.Op + tf.BiasAdd and legalized to TOSA

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"

#define PASS_NAME "tosa-fuse-bias-tf"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace tosa {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

class FuseBiasTF : public TosaFusebiasTFPassBase<FuseBiasTF> {
 public:
  explicit FuseBiasTF() {}
  void runOnFunction() override;
};

struct ConvertTFBiasAddOp : public RewritePattern {
  explicit ConvertTFBiasAddOp(MLIRContext* context)
      : RewritePattern(TF::BiasAddOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

// Replaces the following pattern:
//   %1 = tf.Conv2D (%ifm, %filter)
//   %2 = tf.BiasAdd(%1, %bias)
//   with
//   %1 = tosa.conv2d(%ifm, %filter, %bias)
//   This can also be done using the pair ot Pat<> options in
//   tf_optimize_patterns.td
//   However, this explicit code can handle both when the LHS or RHS is the
//   defining conv2d op.
// TODO: support other pattern. e.g. tf.DepthwiseConv2DNative

LogicalResult ConvertTFBiasAddOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_biasadd_op = cast<TF::BiasAddOp>(op);

  auto output_type =
      tf_biasadd_op.getResult().getType().dyn_cast<RankedTensorType>();
  // Not a ranked tensor output
  if (!output_type) return failure();

  auto value = tf_biasadd_op.value();
  auto bias = tf_biasadd_op.bias();

  TF::Conv2DOp tf_conv2d_op =
      dyn_cast_or_null<TF::Conv2DOp>(value.getDefiningOp());

  if (!tf_conv2d_op) {
    return failure();
  }

  // Sanity check to confirm rhs() has the expected shape of bias
  auto filter_shape =
      tf_conv2d_op.filter().getType().dyn_cast<RankedTensorType>().getShape();

  auto bias_shape = bias.getType().dyn_cast<RankedTensorType>().getShape();

  // Bias dimension must match filter output channels, where tf.conv2d's filter
  // is [H, W, I, O]
  if (filter_shape.back() != bias_shape.back()) return failure();

  // Bias tensor that feeds into tosa.conv2d must be rank 1
  if (bias_shape.size() != 1) return failure();

  auto result = convertTFConv2DCommon(
      rewriter, op, output_type, tf_conv2d_op.input(), tf_conv2d_op.filter(),
      bias, tf_conv2d_op.strides(), tf_conv2d_op.dilations(),
      tf_conv2d_op.explicit_paddings(), tf_conv2d_op.padding(),
      tf_conv2d_op.data_format());

  if (!result) return failure();

  rewriter.replaceOp(op, {result.getValue()});

  return success();
}

void FuseBiasTF::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto* ctx = &getContext();
  auto func = getFunction();

  // Add the generated patterns to the list.
  patterns.insert<ConvertTFBiasAddOp>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createFuseBiasTFPass() {
  return std::make_unique<FuseBiasTF>();
}

}  // namespace tosa

}  // namespace mlir
