/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_hlo {

namespace {

// Broadcasts the 1D value tensor to rank.
Value broadcastToFeatureDim(Location loc, Type result_type, Value value_1d,
                            int64_t feature_dim,
                            ConversionPatternRewriter& rewriter) {
  Builder b(rewriter.getContext());
  auto dims_type = RankedTensorType::get({1}, b.getIntegerType(64));
  auto dims = DenseIntElementsAttr::get(dims_type, {feature_dim});
  return rewriter.create<xla_hlo::BroadcastInDimOp>(loc, result_type, value_1d,
                                                    dims);
}

Value MaterializeEpsilon(Operation* op, FloatAttr epsilon_attr,
                         FloatType fp_type, Type broadcast_to_type,
                         ConversionPatternRewriter& rewriter) {
  Builder b(rewriter.getContext());
  if (epsilon_attr.getType() != fp_type) {
    // Need to convert.
    bool loses_info;
    APFloat epsilon_float = epsilon_attr.getValue();
    auto status = epsilon_float.convert(
        fp_type.getFloatSemantics(), APFloat::rmNearestTiesToEven, &loses_info);
    if ((status & (~APFloat::opInexact)) != APFloat::opOK) {
      op->emitWarning() << "Could not convert batch_norm epsilon to target fp "
                           "type: opStatus = "
                        << static_cast<int>(status);
      return nullptr;
    }
    if (loses_info) {
      op->emitWarning("Conversion of epsilon loses precision");
    }
    epsilon_attr = b.getFloatAttr(fp_type, epsilon_float);
  }

  auto scalar_type = RankedTensorType::get({}, fp_type);
  auto epsilon_tensor_attr =
      DenseElementsAttr::get(scalar_type, {epsilon_attr.cast<Attribute>()});
  Value epsilon =
      rewriter.create<xla_hlo::ConstOp>(op->getLoc(), epsilon_tensor_attr);
  epsilon = rewriter.create<xla_hlo::BroadcastInDimOp>(
      op->getLoc(), broadcast_to_type, epsilon, /*broadcast_dims=*/nullptr);
  return epsilon;
}

class UnfuseBatchNormInferencePattern
    : public OpConversionPattern<xla_hlo::BatchNormInferenceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_hlo::BatchNormInferenceOp bn_op, ArrayRef<Value> raw_operands,
      ConversionPatternRewriter& rewriter) const override {
    xla_hlo::BatchNormInferenceOpOperandAdaptor operands(raw_operands);

    // Enforce type invariants.
    // Note that we deduce the actual element type from the variance,
    // which should not be subject to quantization at a higher level.
    auto input_type = operands.operand().getType();
    auto variance_type = operands.variance().getType().dyn_cast<ShapedType>();
    if (!variance_type) {
      return matchFailure();
    }
    auto fp_type = variance_type.getElementType().dyn_cast<FloatType>();
    if (!fp_type) {
      return matchFailure();
    }
    int64_t feature_dim = bn_op.feature_index().getSExtValue();

    // Add epsilon to the variance and sqrt to get stddev:
    // stddev = sqrt(variance + epsilon)
    auto epsilon = MaterializeEpsilon(bn_op.getOperation(), bn_op.epsilonAttr(),
                                      fp_type, variance_type, rewriter);
    if (!epsilon) {
      return matchFailure();
    }
    Value stddev =
        rewriter.create<xla_hlo::AddOp>(bn_op.getLoc(), operands.variance(),
                                        epsilon, /*broadcast_dims=*/nullptr);
    stddev = rewriter.create<xla_hlo::SqrtOp>(bn_op.getLoc(), stddev);

    // Broadcast all terms.
    auto broadcast_scale = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, operands.scale(), feature_dim, rewriter);
    auto broadcast_offset = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, operands.offset(), feature_dim, rewriter);
    auto broadcast_mean = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, operands.mean(), feature_dim, rewriter);
    auto broadcast_stddev = broadcastToFeatureDim(
        bn_op.getLoc(), input_type, stddev, feature_dim, rewriter);

    // Compute:
    // scale * (input - mean) / stddev + offset
    Value result = rewriter.create<xla_hlo::SubOp>(
        bn_op.getLoc(), operands.operand(), broadcast_mean, nullptr);
    result = rewriter.create<xla_hlo::MulOp>(bn_op.getLoc(), result,
                                             broadcast_scale, nullptr);
    result = rewriter.create<xla_hlo::DivOp>(bn_op.getLoc(), result,
                                             broadcast_stddev, nullptr);
    rewriter.replaceOpWithNewOp<xla_hlo::AddOp>(bn_op, result, broadcast_offset,
                                                nullptr);

    return matchSuccess();
  }
};

}  // namespace

// Populates conversion patterns to unfuse batch normalization operations.
// In combination with marking such ops as illegal, this allows backends that
// do not have special support for fused batchnorm to use simpler arithmetic
// primitives.
void PopulateUnfuseBatchNormPatterns(MLIRContext* context,
                                     OwningRewritePatternList* patterns) {
  patterns->insert<UnfuseBatchNormInferencePattern>(context);
}

}  // namespace xla_hlo
}  // namespace mlir
