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

#include <numeric>

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace TF {
namespace {

// Returns 1D 64-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                               Builder *builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
static DenseIntElementsAttr GetI64ElementsAttrForSeq(int start, int end,
                                                     Builder *builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = RankedTensorType::get({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, vals);
}

// Returns int or float DenseElementsAttr with scalar shape with the given
// element type and the integer value.
static DenseElementsAttr GetScalarOfType(Type ty, int64_t raw_value) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);
  if (auto float_ty = ty.dyn_cast_or_null<FloatType>()) {
    FloatAttr attr = FloatAttr::get(float_ty, raw_value);
    return DenseElementsAttr::get(scalar_ty, attr);
  }

  auto int_ty = ty.cast<IntegerType>();
  IntegerAttr attr = IntegerAttr::get(int_ty, raw_value);
  return DenseElementsAttr::get(scalar_ty, attr);
}

// Returns reduction indices to use while lowering tf.BiasAddGrad op to tf.Sum
// op.
DenseIntElementsAttr GetBiasAddGradReductionIndices(int64_t rank,
                                                    StringAttr data_format,
                                                    Builder *builder) {
  tensorflow::TensorFormat format;
  if (!FormatFromString(data_format.getValue().str(), &format)) return {};

  // Reudce along all dimensions except the feature dimension.
  int64_t feature_dim = GetTensorFeatureDimIndex(rank, format);
  llvm::SmallVector<int64_t, 4> dims_to_reduce(rank - 1);
  std::iota(dims_to_reduce.begin(), dims_to_reduce.begin() + feature_dim, 0);
  std::iota(dims_to_reduce.begin() + feature_dim, dims_to_reduce.end(),
            feature_dim + 1);
  return GetI64ElementsAttr(dims_to_reduce, builder);
}

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_lower_tf.inc"

// Infers ExpandDims op output type for the given input type `ty` and dimension
// to expand at the given `axis`.
Type InferExpandDimsType(Type ty, int64_t axis, Builder *builder) {
  auto ranked_ty = ty.dyn_cast<RankedTensorType>();

  // Unranked type.
  if (!ranked_ty) return ty;

  auto shape = llvm::to_vector<4>(ranked_ty.getShape());
  if (axis < 0) axis += ranked_ty.getRank() + 1;

  shape.insert(shape.begin() + axis, 1);
  return RankedTensorType::get(shape, ranked_ty.getElementType());
}

// Lowers AddN op to a sequence of AddV2 ops to accumulate operands.
//
//   %result = "tf.AddN"(%0, %1, %2)
//
// is lowered to:
//
//   %sum_0 = "tf.AddV2"(%0, %1)
//   %result = "tf.AddV2"(%sum_0, %2)
//
class LowerAddNOp : public OpRewritePattern<TF::AddNOp> {
 public:
  explicit LowerAddNOp(MLIRContext *context)
      : OpRewritePattern<TF::AddNOp>(context) {}

  PatternMatchResult matchAndRewrite(TF::AddNOp op,
                                     PatternRewriter &rewriter) const override {
    // TODO(hinsu): Support variant with TensorList type. tf.AddV2 doesn't
    // support variant type so variant types require special handling.
    if (getElementTypeOrSelf(op.getType()).isa<VariantType>())
      return matchFailure();

    // TODO(hinsu): Improve parallelism by splitting operands in two halves and
    // accumulating them first.
    Value *result = *op.inputs().begin();
    for (Value *operand : llvm::drop_begin(op.inputs(), 1)) {
      result = rewriter.create<TF::AddV2Op>(op.getLoc(), result, operand);
    }

    rewriter.replaceOp(op, result);
    return matchSuccess();
  }
};

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
        loc,
        DenseElementsAttr::get(
            RankedTensorType::get({}, rewriter.getIntegerType(64)), op.axis()));
    int64_t axis = op.axis().getSExtValue();

    Type prev_input_ty, inferred_ty;
    SmallVector<Value *, 4> expanded_inputs;
    expanded_inputs.reserve(op.N());
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

    rewriter.replaceOpWithNewOp<TF::ConcatV2Op>(op, op.getType(),
                                                expanded_inputs, axis_value);
    return matchSuccess();
  }
};

}  // namespace

void PopulateLoweringTFPatterns(MLIRContext *context,
                                OwningRewritePatternList *patterns) {
  patterns->insert<LowerAddNOp>(context);
  patterns->insert<LowerPackOp>(context);
  populateWithGenerated(context, patterns);
}

}  // namespace TF
}  // namespace mlir
