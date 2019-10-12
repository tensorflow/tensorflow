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

#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {
namespace {

// Infers ExpandDims op output type for the given input type `ty` and dimension
// to expand at the given `axis`.
Type InferExpandDimsType(Type ty, int64_t axis, Builder *builder) {
  auto ranked_ty = ty.dyn_cast<RankedTensorType>();

  // Unranked type.
  if (!ranked_ty) return ty;

  auto shape = llvm::to_vector<4>(ranked_ty.getShape());
  if (axis < 0) axis += ranked_ty.getRank() + 1;

  shape.insert(shape.begin() + axis, 1);
  return builder->getTensorType(shape, ranked_ty.getElementType());
}

// Lowers Pack op to ConcatV2 op after changing shape of the inputs with
// ExpandDims op.
//
// Sample result with 2 inputs to pack:
//
//   %axis = "tf.Const"() {value = dense<1> : tensor<i64>}
//   %inp0 = "tf.ExpandDims"(%operand0, %axis): tensor<2xf32> -> tensor<2x1xf32>
//   %inp1 = "tf.ExpandDims"(%operand1, %axis): tensor<2xf32> -> tensor<2x1xf32>
//   %result = "tf.ConcatV2"(%operand0, %operand1, %axis) { N = 2 : i64 }:
//
class LowerPackOp : public OpRewritePattern<TF::PackOp> {
 public:
  explicit LowerPackOp(MLIRContext *context)
      : OpRewritePattern<TF::PackOp>(context) {}

  PatternMatchResult matchAndRewrite(TF::PackOp op,
                                     PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto axis_value = rewriter.create<TF::ConstOp>(
        loc, DenseElementsAttr::get(
                 rewriter.getTensorType({}, rewriter.getIntegerType(64)),
                 op.axis()));
    int64_t axis = op.axis().getSExtValue();

    Type prev_input_ty, inferred_ty;
    SmallVector<Value *, 4> expanded_inputs;
    expanded_inputs.reserve(op.N().getSExtValue());
    for (Value *input : op.values()) {
      // If input type is different than the previous input type, infer the
      // output type. Otherwise, use the already inferred output type from the
      // previous iteration.
      Type input_ty = input->getType();
      if (input_ty != prev_input_ty) {
        inferred_ty = InferExpandDimsType(input_ty, axis, &rewriter);
        prev_input_ty = input_ty;
      }
      expanded_inputs.push_back(rewriter.create<TF::ExpandDimsOp>(
          loc, inferred_ty, input, axis_value));
    }

    rewriter.replaceOpWithNewOp<TF::ConcatV2Op>(
        op, op.getType(), expanded_inputs, axis_value,
        op.getAttrOfType<IntegerAttr>("N"));
    return matchSuccess();
  }
};

}  // namespace

void PopulateLoweringTFPatterns(MLIRContext *context,
                                OwningRewritePatternList *patterns) {
  patterns->insert<LowerPackOp>(context);
}

}  // namespace TF
}  // namespace mlir
