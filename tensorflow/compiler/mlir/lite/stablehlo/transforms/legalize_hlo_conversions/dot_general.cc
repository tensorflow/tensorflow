/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for legalizing mhlo.dot_general to
// tflite.batch_matmul.
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/dot_general.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {
namespace {
// A struct to hold axes and sizes for a set of dimensions.
struct DimensionVector {
  llvm::ArrayRef<int64_t> AxesArray() const { return axes; }
  llvm::ArrayRef<int64_t> SizesArray() const { return sizes; }

  llvm::SmallVector<int64_t, 4> axes;
  llvm::SmallVector<int64_t, 4> sizes;
};

// Appends all elements in `range` to `values`.
template <typename ValueT, typename Range>
void Append(llvm::SmallVectorImpl<ValueT>& values, Range&& range) {
  values.insert(values.end(), range.begin(), range.end());
}

// Appends all elements in `range` to `values`.
template <typename ValueT, typename Range, typename... RangeTs>
void Append(llvm::SmallVectorImpl<ValueT>& values, Range&& range,
            RangeTs&&... ranges) {
  values.insert(values.end(), range.begin(), range.end());
  Append(values, ranges...);
}

// Returns the number of elements in `range`.
template <typename Range>
size_t Size(Range&& range) {
  return range.size();
}

// Returns the total number of elements in a variadic number of `ranges`.
template <typename Range, typename... RangeTs>
size_t Size(Range&& range, RangeTs&&... ranges) {
  return range.size() + Size(std::forward<RangeTs>(ranges)...);
}

// Concats all elements in `ranges` and returns a small vector as a result.
template <typename ValueT, typename... RangeTs>
llvm::SmallVector<ValueT, 4> Concat(RangeTs&&... ranges) {
  llvm::SmallVector<int64_t, 4> results;
  results.reserve(Size(std::forward<RangeTs>(ranges)...));
  Append(results, std::forward<RangeTs>(ranges)...);
  return results;
}

// A struct to hold information about dimensions of dot_general operands.
class DotDimensionsInfo {
 public:
  DotDimensionsInfo(ShapedType type, ArrayRef<int64_t> batch_dimensions,
                    ArrayRef<int64_t> contracting_dimensions) {
    const int64_t rank = type.getRank();
    for (const int64_t dim : batch_dimensions) {
      batch_dimensions_.axes.push_back(dim);
      batch_dimensions_.sizes.push_back(type.getDimSize(dim));
    }

    // Create a sorted contracting dimensions array. This currently doesn't
    // support dynamic tensors.
    llvm::SmallVector<int64_t, 4> sorted_contracting_dimensions =
        llvm::to_vector(contracting_dimensions);
    if (type.hasStaticShape()) {
      llvm::sort(sorted_contracting_dimensions);
    }
    for (const int64_t dim : sorted_contracting_dimensions) {
      contracting_dimensions_.axes.push_back(dim);
      contracting_dimensions_.sizes.push_back(type.getDimSize(dim));
    }

    for (int64_t dim = 0; dim < rank; ++dim) {
      if (llvm::count(contracting_dimensions_.axes, dim) > 0 ||
          llvm::count(batch_dimensions_.axes, dim) > 0) {
        continue;
      }
      out_dimensions_.axes.push_back(dim);
      out_dimensions_.sizes.push_back(type.getDimSize(dim));
    }
  }

  const DimensionVector& batch_dimensions() const { return batch_dimensions_; }
  const DimensionVector& contracting_dimensions() const {
    return contracting_dimensions_;
  }
  // Out dimensions are any dimensions that are neither batch nor contracting
  // dimensions, hence will be propagated to output shape.
  const DimensionVector& out_dimensions() const { return out_dimensions_; }

  // Returns the total dimension size after flattening all contracting
  // dimensions.
  int64_t FlattenedContractingDimensionSize() const {
    if (ShapedType::isDynamicShape(contracting_dimensions_.sizes)) {
      return ShapedType::kDynamic;
    }
    return std::accumulate(contracting_dimensions_.sizes.begin(),
                           contracting_dimensions_.sizes.end(), 1,
                           std::multiplies<int64_t>());
  }

  // Returns the total dimension size after flattening all out dimensions.
  int64_t FlattenedOutDimensionSize() const {
    if (ShapedType::isDynamicShape(out_dimensions_.sizes)) {
      return ShapedType::kDynamic;
    }
    return std::accumulate(out_dimensions_.sizes.begin(),
                           out_dimensions_.sizes.end(), 1,
                           std::multiplies<int64_t>());
  }

 private:
  DimensionVector batch_dimensions_;
  DimensionVector contracting_dimensions_;
  // Out dimensions are any dimensions that are neither batch nor contracting
  // dimensions, hence will be propagated to output shape.
  DimensionVector out_dimensions_;
};

// Calculates the flattened shapes for dynamic shaped operands in
// mhlo.dot_general:
//   1. flattened_out_dim = UnsortedSegmentProdOp(operand_shape, out_axes)
//   2. flattened_contracting_dim = UnsortedSegmentProdOp(operand_shape,
//   contracting_axes)
//   3. batch_dimensions = Gather(operand_shape, batch_axes)
//   4. flattened_shape = Concat(batch_dimensions, flattened_out_dim,
//   flattened_contracting_dim)
// The flattened shape for LHS
// is like [batch_dimensions, flattened_out_dimension,
// flattened_contracting_dimension] and [batch_dimensions,
// flattened_contracting_dimension, flattened_out_dimension] for RHS.
Value BuildDotOperandFlattenedShapeOp(Value operand,
                                      DotDimensionsInfo dot_dimensions_info,
                                      ImplicitLocOpBuilder& builder,
                                      bool is_lhs) {
  auto operand_type = mlir::cast<ShapedType>(operand.getType());
  auto operand_shape = builder.create<TFL::ShapeOp>(
      RankedTensorType::get(static_cast<int32_t>(operand_type.getRank()),
                            builder.getIntegerType(32)),
      operand);
  const int64_t operand_rank = operand_type.getRank();
  // Compute flattened out dimension and contracting dimension using
  // TFL::UnsortedSegmentProdOp.
  llvm::SmallVector<int32_t, 4> flattened_out_segids =
      llvm::SmallVector<int32_t, 4>(operand_rank, static_cast<int32_t>(-1));
  for (int64_t i : dot_dimensions_info.out_dimensions().AxesArray()) {
    flattened_out_segids[i] = 0;
  }
  llvm::SmallVector<int32_t, 4> flattened_contracting_segids =
      llvm::SmallVector<int32_t, 4>(operand_rank, static_cast<int32_t>(-1));
  for (int64_t i : dot_dimensions_info.contracting_dimensions().AxesArray()) {
    flattened_contracting_segids[i] = 0;
  }
  auto seg_prod_result_type =
      RankedTensorType::get(static_cast<int32_t>(1), builder.getI32Type());
  auto out_segids_cst = builder.create<TFL::ConstOp>(
      builder.getI32TensorAttr(flattened_out_segids));
  auto contracting_segids_cst = builder.create<TFL::ConstOp>(
      builder.getI32TensorAttr(flattened_contracting_segids));
  auto num_segids_tensor =
      builder.create<TFL::ConstOp>(DenseIntElementsAttr::get(
          RankedTensorType::get({}, builder.getIntegerType(32)), 1));
  auto flattened_out_dims = builder.create<TFL::UnsortedSegmentProdOp>(
      seg_prod_result_type, operand_shape, out_segids_cst, num_segids_tensor);
  auto flattened_contracting_dims = builder.create<TFL::UnsortedSegmentProdOp>(
      seg_prod_result_type, operand_shape, contracting_segids_cst,
      num_segids_tensor);
  llvm::SmallVector<Value, 3> flattend_shape_values;
  // Gather the batch dimensions.
  if (!dot_dimensions_info.batch_dimensions().AxesArray().empty()) {
    if (ShapedType::isDynamicShape(
            dot_dimensions_info.batch_dimensions().SizesArray())) {
      auto batch_axes_tensor =
          builder.create<TFL::ConstOp>(builder.getI64TensorAttr(
              dot_dimensions_info.batch_dimensions().AxesArray()));
      auto batch_dims = builder.create<TFL::GatherOp>(
          RankedTensorType::get(
              {static_cast<int>(
                  dot_dimensions_info.batch_dimensions().AxesArray().size())},
              builder.getIntegerType(32)),
          operand_shape, batch_axes_tensor, /*axis*/ 0, /*batch_dims*/ 0);
      flattend_shape_values.push_back(batch_dims);
    } else {
      llvm::SmallVector<int32_t> batch_i32_vec;
      for (int64_t element :
           dot_dimensions_info.batch_dimensions().SizesArray()) {
        batch_i32_vec.push_back(static_cast<int32_t>(element));
      }
      auto batch_dims =
          builder.create<TFL::ConstOp>(builder.getI32TensorAttr(batch_i32_vec));
      flattend_shape_values.push_back(batch_dims);
    }
  }
  flattend_shape_values.push_back(
      (is_lhs ? flattened_out_dims : flattened_contracting_dims));
  flattend_shape_values.push_back(
      (is_lhs ? flattened_contracting_dims : flattened_out_dims));

  auto concat_result_type = RankedTensorType::get(
      {static_cast<int32_t>(
           dot_dimensions_info.batch_dimensions().AxesArray().size()) +
       2},
      builder.getIntegerType(32));
  // Concatenate the batch dimensions, flattened out dimension and flattened
  // contracting dimension.
  return builder.create<TFL::ConcatenationOp>(
      concat_result_type, flattend_shape_values, /*axis*/ 0,
      /*fused_activation_function*/ "NONE");
}
}  // namespace

Value ConvertDot(PatternRewriter& rewriter, Value lhs, Value rhs,
                 mhlo::DotDimensionNumbersAttr dot_dimension_numbers,
                 ShapedType result_type, mlir::Location loc) {
  auto lhs_type = mlir::cast<ShapedType>(lhs.getType());
  auto rhs_type = mlir::cast<ShapedType>(rhs.getType());
  const int lhs_rank = lhs_type.getRank();
  const int rhs_rank = rhs_type.getRank();
  ImplicitLocOpBuilder builder(loc, rewriter);

  // Collects lhs and rhs dimensions information.
  DotDimensionsInfo lhs_dot_dimensions_info(
      lhs_type, dot_dimension_numbers.getLhsBatchingDimensions(),
      dot_dimension_numbers.getLhsContractingDimensions());
  DotDimensionsInfo rhs_dot_dimensions_info(
      rhs_type, dot_dimension_numbers.getRhsBatchingDimensions(),
      dot_dimension_numbers.getRhsContractingDimensions());

  // Transposes lhs shape to be in the order of {batch_dimensions,
  // out_dimensions, contracting dimensions}.
  llvm::SmallVector<int64_t, 4> lhs_permutation = Concat<int64_t>(
      lhs_dot_dimensions_info.batch_dimensions().AxesArray(),
      lhs_dot_dimensions_info.out_dimensions().AxesArray(),
      lhs_dot_dimensions_info.contracting_dimensions().AxesArray());
  llvm::SmallVector<int64_t, 4> lhs_transposed_shape = Concat<int64_t>(
      lhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      lhs_dot_dimensions_info.out_dimensions().SizesArray(),
      lhs_dot_dimensions_info.contracting_dimensions().SizesArray());
  auto lhs_transposed = rewriter.create<mhlo::TransposeOp>(
      loc,
      RankedTensorType::get(lhs_transposed_shape, lhs_type.getElementType()),
      lhs,
      DenseIntElementsAttr::get(
          RankedTensorType::get({lhs_rank}, rewriter.getI64Type()),
          lhs_permutation));

  // Transposes rhs shape to be in the order of {batch_dimensions, contracting
  // dimensions, out_dimensions}.
  llvm::SmallVector<int64_t, 4> rhs_permutation = Concat<int64_t>(
      rhs_dot_dimensions_info.batch_dimensions().AxesArray(),
      rhs_dot_dimensions_info.contracting_dimensions().AxesArray(),
      rhs_dot_dimensions_info.out_dimensions().AxesArray());
  llvm::SmallVector<int64_t, 4> rhs_transposed_shape = Concat<int64_t>(
      rhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      rhs_dot_dimensions_info.contracting_dimensions().SizesArray(),
      rhs_dot_dimensions_info.out_dimensions().SizesArray());
  auto rhs_transposed = rewriter.create<mhlo::TransposeOp>(
      loc,
      RankedTensorType::get(rhs_transposed_shape, rhs_type.getElementType()),
      rhs,
      DenseIntElementsAttr::get(
          RankedTensorType::get({rhs_rank}, rewriter.getI64Type()),
          rhs_permutation));
  // Reshapes lhs to flatten out_dimensions and contracting_dimensions.
  llvm::SmallVector<int64_t, 4> lhs_flattened_shape = Concat<int64_t>(
      lhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      llvm::ArrayRef<int64_t>{
          lhs_dot_dimensions_info.FlattenedOutDimensionSize()},
      llvm::ArrayRef<int64_t>{
          lhs_dot_dimensions_info.FlattenedContractingDimensionSize()});
  Value lhs_flattend;
  if (lhs_type.hasStaticShape()) {
    lhs_flattend = rewriter.create<mhlo::ReshapeOp>(
        loc,
        RankedTensorType::get(lhs_flattened_shape, lhs_type.getElementType()),
        lhs_transposed.getResult());
  } else {
    auto lhs_flattend_shape_op = BuildDotOperandFlattenedShapeOp(
        lhs, lhs_dot_dimensions_info, builder, /*is_lhs=*/true);
    lhs_flattend = rewriter.create<mhlo::DynamicReshapeOp>(
        loc,
        RankedTensorType::get(lhs_flattened_shape, lhs_type.getElementType()),
        lhs_transposed, lhs_flattend_shape_op);
  }

  // Reshapes rhs to flatten out_dimensions and contracting_dimensions.
  llvm::SmallVector<int64_t, 4> rhs_flattened_shape = Concat<int64_t>(
      rhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      llvm::ArrayRef<int64_t>{
          rhs_dot_dimensions_info.FlattenedContractingDimensionSize()},
      llvm::ArrayRef<int64_t>{
          rhs_dot_dimensions_info.FlattenedOutDimensionSize()});
  Value rhs_flattend;
  if (rhs_type.hasStaticShape()) {
    rhs_flattend = rewriter.create<mhlo::ReshapeOp>(
        loc,
        RankedTensorType::get(rhs_flattened_shape, rhs_type.getElementType()),
        rhs_transposed.getResult());
  } else {
    auto rhs_flattend_shape_op = BuildDotOperandFlattenedShapeOp(
        rhs, rhs_dot_dimensions_info, builder, /*is_lhs=*/false);
    rhs_flattend = rewriter.create<mhlo::DynamicReshapeOp>(
        loc,
        RankedTensorType::get(rhs_flattened_shape, rhs_type.getElementType()),
        rhs_transposed, rhs_flattend_shape_op);
  }

  // Creates matmul op of `lhs_flattend` and `rhs_flattend`.
  llvm::SmallVector<int64_t, 4> matmul_shape =
      Concat<int64_t>(lhs_dot_dimensions_info.batch_dimensions().SizesArray(),
                      llvm::ArrayRef<int64_t>{
                          lhs_dot_dimensions_info.FlattenedOutDimensionSize()},
                      llvm::ArrayRef<int64_t>{
                          rhs_dot_dimensions_info.FlattenedOutDimensionSize()});
  BoolAttr false_attr = rewriter.getBoolAttr(false);
  auto matmul = rewriter.create<TFL::BatchMatMulOp>(
      loc, RankedTensorType::get(matmul_shape, result_type.getElementType()),
      lhs_flattend, rhs_flattend, /*adj_x*/ false_attr, /*adj_y*/ false_attr,
      /*asym_quant_input*/ false_attr);
  if (result_type.hasStaticShape()) {
    auto reshaped =
        rewriter.create<mhlo::ReshapeOp>(loc, result_type, matmul.getResult());
    return reshaped.getResult();
  }

  // Reshape for dynamic shaped operands. The result shape is
  // [lhs_batch_dimensions, lhs_out_dimensions, rhs_out_dimensions].
  auto lhs_shape = rewriter.create<TFL::ShapeOp>(
      loc,
      RankedTensorType::get(static_cast<int32_t>(lhs_type.getRank()),
                            builder.getIntegerType(32)),
      lhs);
  auto rhs_shape = rewriter.create<TFL::ShapeOp>(
      loc,
      RankedTensorType::get(static_cast<int32_t>(rhs_type.getRank()),
                            builder.getIntegerType(32)),
      rhs);
  llvm::SmallVector<int64_t, 4> lhs_batch_and_out =
      Concat<int64_t>(lhs_dot_dimensions_info.batch_dimensions().AxesArray(),
                      lhs_dot_dimensions_info.out_dimensions().AxesArray());
  auto lhs_batch_and_out_cst = rewriter.create<TFL::ConstOp>(
      loc, rewriter.getI64TensorAttr(lhs_batch_and_out));
  auto lhs_batch_and_out_dims = rewriter.create<TFL::GatherOp>(
      loc,
      RankedTensorType::get({static_cast<int>(lhs_batch_and_out.size())},
                            rewriter.getIntegerType(32)),
      lhs_shape, lhs_batch_and_out_cst,
      /*axis*/ 0, /*batch_dims*/ 0);
  auto rhs_out_cst = rewriter.create<TFL::ConstOp>(
      loc, rewriter.getI64TensorAttr(
               rhs_dot_dimensions_info.out_dimensions().AxesArray()));
  auto rhs_out_dims = rewriter.create<TFL::GatherOp>(
      loc,
      RankedTensorType::get(
          {static_cast<int32_t>(
              rhs_dot_dimensions_info.out_dimensions().AxesArray().size())},
          rewriter.getIntegerType(32)),
      rhs_shape, rhs_out_cst,
      /*axis*/ 0, /*batch_dims*/ 0);
  auto result_shape_type = RankedTensorType::get(
      {static_cast<int32_t>(
          lhs_dot_dimensions_info.batch_dimensions().AxesArray().size() +
          lhs_dot_dimensions_info.out_dimensions().AxesArray().size() +
          rhs_dot_dimensions_info.out_dimensions().AxesArray().size())},
      rewriter.getIntegerType(32));
  auto result_shape = rewriter.create<TFL::ConcatenationOp>(
      loc, result_shape_type, ValueRange{lhs_batch_and_out_dims, rhs_out_dims},
      0, "NONE");

  auto reshaped = rewriter.create<mhlo::DynamicReshapeOp>(
      loc, result_type, matmul.getResult(), result_shape);
  return reshaped.getResult();
}

LogicalResult LowerDotGeneralOp::matchAndRewrite(
    mhlo::DotGeneralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto val = ConvertDot(
      rewriter, op.getLhs(), op.getRhs(), op.getDotDimensionNumbers(),
      mlir::cast<ShapedType>(op.getResult().getType()), op.getLoc());
  rewriter.replaceOp(op, val.getDefiningOp());
  return mlir::success();
}

}  // namespace odml
}  // namespace mlir
