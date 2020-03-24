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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Diagnostics.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
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

// Returns float DenseElementsAttr with scalar shape with the specified value.
static DenseElementsAttr GetScalarOfFloatType(Type ty, double raw_value) {
  auto float_ty = ty.cast<FloatType>();
  FloatAttr attr = FloatAttr::get(float_ty, raw_value);
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);
  return DenseElementsAttr::get(scalar_ty, attr);
}

// Returns reduction indices to use while lowering tf.BiasAddGrad op to tf.Sum
// op.
DenseIntElementsAttr GetBiasAddGradReductionIndices(int64_t rank,
                                                    StringAttr data_format,
                                                    Builder *builder) {
  tensorflow::TensorFormat format;
  if (!FormatFromString(data_format.getValue().str(), &format)) return {};

  // Reduce along all dimensions except the feature dimension.
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

  LogicalResult matchAndRewrite(TF::AddNOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(hinsu): Support variant with TensorList type. tf.AddV2 doesn't
    // support variant type so variant types require special handling.
    if (getElementTypeOrSelf(op.getType()).isa<VariantType>()) return failure();

    // TODO(hinsu): Improve parallelism by splitting operands in two halves and
    // accumulating them first.
    Value result = *op.inputs().begin();
    for (Value operand : llvm::drop_begin(op.inputs(), 1)) {
      result = rewriter.create<TF::AddV2Op>(op.getLoc(), result, operand);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lowers DynamicStitch op with constant indices and with static input and
// output shapes using Reshape, UnPack and ConcatV2 op.
//
//   %indices0 = "tf.Const"() {value = dense<4> : tensor<i32>}
//   %indices1 = "tf.Const"() {value = dense<[[3, 2], [1, 0]]> :
//   tensor<2x2xi32>} %0 = "tf.DynamicStitch"(%indices0, %indices1, %arg0,
//   %arg1)
//     : (tensor<i32>, tensor<2x2xi32>, tensor<2xf32>, tensor<2x2x2xf32>)
//     -> tensor<5x2xf32>
//
// is lowered to
//
//   %shape = "tf.Const"() {value = dense<[-1, 2]> : tensor<2xi64>}
//   %inp0 = "tf.Reshape"(%arg0, %shape)
//     : (tensor<2xf32>, tensor<2xi64>) -> tensor<1x2xf32>
//   %inp1 = "tf.Reshape"(%arg1, %shape)
//     : (tensor<2x2x2xf32>, tensor<2xi64>) -> tensor<4x2xf32>
//   %items0 = "tf.Unpack"(%[[INP0]]) {axis = 0 : i64}
//     : (tensor<1x2xf32>) -> tensor<2xf32>
//   %items1:4 = "tf.Unpack"(%[[INP1]]) {axis = 0 : i64}
//     : (tensor<4x2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
//     tensor<2xf32>)
//   %axis = "tf.Const"() {value = dense<0> : tensor<i64>}
//   %0 = "tf.ConcatV2"(items1#3, items1#2, items1#1, items1#0, %items0, %axis)
//     : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
//        tensor<2xf32>, tensor<i64>) -> tensor<5x2xf32>
//
class LowerDynamicStitchOp : public OpRewritePattern<TF::DynamicStitchOp> {
 public:
  explicit LowerDynamicStitchOp(MLIRContext *context)
      : OpRewritePattern<TF::DynamicStitchOp>(context) {}

  LogicalResult matchAndRewrite(DynamicStitchOp op,
                                PatternRewriter &rewriter) const override {
    // Static output type is used to compute intermediate values. Note that the
    // output type doesn't have to be static but if input types and indices are
    // constant, then the output type can be statically determined.
    RankedTensorType out_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!out_ty || !out_ty.hasStaticShape()) return failure();

    // Extract out all the constant indices' attributes and verify that data
    // types are static.
    SmallVector<DenseIntElementsAttr, 4> indices;
    indices.reserve(op.N());
    for (auto it : llvm::zip(op.indices(), op.data())) {
      Value index = std::get<0>(it);
      Value data = std::get<1>(it);

      DenseIntElementsAttr index_attr;
      if (!matchPattern(index, m_Constant(&index_attr))) return failure();
      indices.push_back(index_attr);

      RankedTensorType data_ty = data.getType().dyn_cast<RankedTensorType>();
      if (!data_ty || !data_ty.hasStaticShape()) return failure();
    }

    // Compute type of each of the items and shape to use while reshaping inputs
    // so that they can be unpacked to extract out individual items.
    ArrayRef<int64_t> item_shape = out_ty.getShape().drop_front(1);
    auto item_ty = RankedTensorType::get(item_shape, out_ty.getElementType());

    SmallVector<int64_t, 4> packed_shape;
    packed_shape.push_back(-1);
    packed_shape.append(item_shape.begin(), item_shape.end());
    Location loc = op.getLoc();
    auto packed_shape_val = rewriter.create<ConstOp>(
        loc, GetI64ElementsAttr(packed_shape, &rewriter));

    // Prepare each of the output item by unpacking data and then putting it to
    // the specified index.
    SmallVector<Value, 8> values(out_ty.getDimSize(0));
    for (auto it : llvm::zip(indices, op.data())) {
      DenseIntElementsAttr index_attr = std::get<0>(it);
      Value data = std::get<1>(it);

      auto reshaped_data =
          rewriter.create<ReshapeOp>(loc, data, packed_shape_val);
      auto num_items =
          reshaped_data.getType().cast<RankedTensorType>().getShape()[0];
      auto items = rewriter.create<UnpackOp>(
          loc, SmallVector<Type, 4>(num_items, item_ty), reshaped_data,
          /*axis=*/APInt(64, 0));
      for (auto index_item : llvm::zip(index_attr, items.getResults())) {
        int64_t output_index = std::get<0>(index_item).getSExtValue();
        Value item = std::get<1>(index_item);
        values[output_index] = item;
      }
    }

    auto axis = rewriter.create<ConstOp>(loc, rewriter.getI64IntegerAttr(0));
    rewriter.replaceOpWithNewOp<ConcatV2Op>(op, op.getType(), values, axis);
    return success();
  }
};

// Lowers InvertPermutation op to TensorScatterUpdate op.
//
// Example:
//
//   %x = "tf.Const"() {value = dense<[3, 4, 0, 1, 2]> : tensor<5xi32>}
//   "tf.InvertPermutation"(%x) : (tensor<5xi32>) -> tensor<5xi32>
//
// is lowered to
//
//   %x = "tf.Const"() {value = dense<[3, 4, 0, 1, 2]> : tensor<5xi32>}
//   %start = "tf.Const"() {value = dense<0> : tensor<i32>}
//   %limit = "tf.Const"() {value = dense<5> : tensor<i32>}
//   %delta = "tf.Const"() {value = dense<1> : tensor<i32>}
//   %updates = "tf.Range"(%start, %limit, %delta) :
//     (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<5xi32>
//   %perm = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>}
//   %indices = "tf.Transpose"(%x, %perm) : (tensor<5xi32, tensor<2xi32) ->
//     tensor<5x1xi32>
//   "tf.TensorScatterUpdate"(%x, %indices, %updates) :
//     (tensor<5xi32>, tensor<5x1xi32>, tensor<5xi32>) -> tensor<5xi32>
//
class LowerInvertPermutationOp
    : public OpRewritePattern<TF::InvertPermutationOp> {
 public:
  explicit LowerInvertPermutationOp(MLIRContext *context)
      : OpRewritePattern<TF::InvertPermutationOp>(context) {}

  LogicalResult matchAndRewrite(TF::InvertPermutationOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto x_type = op.x().getType().cast<TensorType>();
    Type int_type = x_type.getElementType();  // Could be i32 or i64.

    // x input must have static shape.
    if (!x_type.hasStaticShape()) {
      return failure();
    }

    auto result_type = x_type;
    auto start =
        rewriter.create<TF::ConstOp>(loc, GetScalarOfType(int_type, 0));
    Value limit = rewriter.create<TF::ConstOp>(
        loc, GetScalarOfType(int_type, x_type.getShape()[0]));
    auto delta =
        rewriter.create<TF::ConstOp>(loc, GetScalarOfType(int_type, 1));
    // Construct a sequence of numbers [0, 1, ... len(x)-1].
    auto updates =
        rewriter.create<TF::RangeOp>(loc, result_type, start, limit, delta);

    auto perm_type = RankedTensorType::get({2}, int_type);
    auto perm = rewriter.create<TF::ConstOp>(
        loc, DenseElementsAttr::get(perm_type, {1, 0}));
    auto transposed_x_type =
        RankedTensorType::get({x_type.getShape()[0], 1}, int_type);
    auto indices =
        rewriter.create<TF::TransposeOp>(loc, transposed_x_type, op.x(), perm);

    rewriter.replaceOpWithNewOp<TF::TensorScatterUpdateOp>(
        op, result_type, op.x(), indices, updates);
    return success();
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

  LogicalResult matchAndRewrite(TF::PackOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto axis_value = rewriter.create<TF::ConstOp>(
        loc,
        DenseElementsAttr::get(
            RankedTensorType::get({}, rewriter.getIntegerType(64)), op.axis()));
    int64_t axis = op.axis().getSExtValue();

    Type prev_input_ty, inferred_ty;
    SmallVector<Value, 4> expanded_inputs;
    expanded_inputs.reserve(op.N());
    for (Value input : op.values()) {
      // If input type is different than the previous input type, infer the
      // output type. Otherwise, use the already inferred output type from the
      // previous iteration.
      Type input_ty = input.getType();
      if (input_ty != prev_input_ty) {
        inferred_ty = InferExpandDimsType(input_ty, axis, &rewriter);
        prev_input_ty = input_ty;
      }
      expanded_inputs.push_back(rewriter.create<TF::ExpandDimsOp>(
          loc, inferred_ty, input, axis_value));
    }

    rewriter.replaceOpWithNewOp<TF::ConcatV2Op>(op, op.getType(),
                                                expanded_inputs, axis_value);
    return success();
  }
};

}  // namespace

void PopulateLoweringTFPatterns(MLIRContext *context,
                                OwningRewritePatternList *patterns) {
  patterns->insert<LowerAddNOp, LowerDynamicStitchOp, LowerInvertPermutationOp,
                   LowerPackOp>(context);
  populateWithGenerated(context, patterns);
}

}  // namespace TF
}  // namespace mlir
