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

// This file implements logic for legalizing HLO to TensorFlow.

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/broadcast_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace mlir {
namespace TF {
namespace {

using mhlo::DotDimensionNumbers;

class ConvertConvOp : public OpConversionPattern<mhlo::ConvOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvOp conv_op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    if (!IsSupportedConvOp(conv_op)) {
      return failure();
    }

    // Constructs strides array.
    // For example, [2, 3] -> [1, 2, 3, 1].
    SmallVector<int64_t, 4> strides({1});
    for (const auto v :
         conv_op.window_strides().getValue().getValues<int64_t>()) {
      strides.emplace_back(v);
    }
    strides.emplace_back(1);

    // Constructs dilation array.
    SmallVector<int64_t, 4> dilation;
    if (auto rhs_dilation = conv_op.rhs_dilation()) {
      // For example, [2, 3] -> [1, 2, 3, 1].
      dilation.emplace_back(1);
      dilation.append(rhs_dilation.getValue().getValues<int64_t>().begin(),
                      rhs_dilation.getValue().getValues<int64_t>().end());
      dilation.emplace_back(1);
    } else {
      // Default value
      dilation = {1, 1, 1, 1};
    }

    const int input_feature_dimension =
        conv_op.dimension_numbers().input_feature_dimension().getInt();
    const int input_channels =
        conv_op.lhs().getType().cast<ShapedType>().getDimSize(
            input_feature_dimension);
    int feature_group_count = conv_op.feature_group_count();

    if (feature_group_count != 1 && feature_group_count != input_channels) {
      // Group convolution is not supported yet.
      return failure();
    }

    const bool is_depthwise_conv = input_channels == feature_group_count;
    std::string padding;

    if (!conv_op.padding().hasValue() ||
        (conv_op.padding().getValue().isSplat() &&
         conv_op.padding()->getSplatValue<int64_t>() == 0)) {
      padding = "VALID";
    } else {
      // Check if padding is "SAME".
      // TODO(chhe): To support "EXPLICIT" padding.
      SmallVector<int64_t, 8> padding_array;
      for (const auto v : conv_op.padding().getValue().getValues<int64_t>()) {
        padding_array.emplace_back(v);
      }

      const int num_spatial_dims = conv_op.dimension_numbers()
                                       .input_spatial_dimensions()
                                       .getNumElements();
      if (!IsSamePadding(conv_op, num_spatial_dims, strides, dilation,
                         padding_array))
        return failure();

      padding = "SAME";
    }

    CreateConvOp(conv_op, strides, padding, dilation, is_depthwise_conv,
                 input_channels, rewriter);
    return success();
  };

 private:
  bool IsSamePadding(mhlo::ConvOp conv_op, int num_spatial_dims,
                     ArrayRef<int64_t> strides, ArrayRef<int64_t> dilation,
                     ArrayRef<int64_t> padding_array) const {
    for (auto i : llvm::seq<int>(0, num_spatial_dims)) {
      int dim = i + 1;
      tensorflow::int64 output_size;
      tensorflow::int64 pad_low_int64;
      tensorflow::int64 pad_high_int64;
      tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerboseV2(
          conv_op.lhs().getType().cast<ShapedType>().getDimSize(dim),
          conv_op.rhs().getType().cast<ShapedType>().getDimSize(i),
          dilation[dim], strides[dim], tensorflow::Padding::SAME, &output_size,
          &pad_low_int64, &pad_high_int64);
      if (!status.ok()) return false;
      if (padding_array[2 * i] != pad_low_int64 ||
          padding_array[2 * i + 1] != pad_high_int64)
        return false;
    }

    return true;
  }

  void CreateConvOp(mhlo::ConvOp conv_op, ArrayRef<int64_t> strides,
                    StringRef padding, ArrayRef<int64_t> dilation,
                    bool is_depthwise_conv, int input_channels,
                    ConversionPatternRewriter &rewriter) const {
    // TODO(chhe): To support more data formats other than "NHWC".
    if (is_depthwise_conv) {
      // Reshapes filter format to [filter_height, filter_width, in_channels,
      // channel_multiplier] from HLO's [filter_height, filter_width, 1,
      // in_channels * channel_multiplier] format.
      auto filter_type = conv_op.rhs().getType().cast<ShapedType>();
      llvm::ArrayRef<int64_t> hlo_filter_shape = filter_type.getShape();
      llvm::SmallVector<int64_t, 4> tf_filter_shape(hlo_filter_shape.begin(),
                                                    hlo_filter_shape.end());
      tf_filter_shape[2] = input_channels;
      tf_filter_shape[3] = hlo_filter_shape.back() / input_channels;
      auto reshaped_filter = rewriter.create<mhlo::ReshapeOp>(
          conv_op.rhs().getLoc(),
          RankedTensorType::get(tf_filter_shape, filter_type.getElementType()),
          conv_op.rhs());

      rewriter.replaceOpWithNewOp<DepthwiseConv2dNativeOp>(
          conv_op, conv_op.getType(), conv_op.lhs(), reshaped_filter,
          rewriter.getI64ArrayAttr(strides),
          /*padding=*/rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr({}),
          /*data_format=*/rewriter.getStringAttr("NHWC"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation));
    } else {
      rewriter.replaceOpWithNewOp<Conv2DOp>(
          conv_op, conv_op.getType(), conv_op.lhs(), conv_op.rhs(),
          rewriter.getI64ArrayAttr(strides),
          /*use_cudnn_on_gpu=*/rewriter.getBoolAttr(true),
          /*padding=*/rewriter.getStringAttr(padding),
          /*explicit_paddings=*/rewriter.getI64ArrayAttr({}),
          /*data_format=*/rewriter.getStringAttr("NHWC"),
          /*dilations=*/rewriter.getI64ArrayAttr(dilation));
    }
  }

  bool IsSupportedConvOp(mhlo::ConvOp conv_op) const {
    if (!conv_op.lhs().getType().cast<ShapedType>().hasStaticShape() ||
        !conv_op.rhs().getType().cast<ShapedType>().hasStaticShape() ||
        !conv_op.getType().cast<ShapedType>().hasStaticShape())
      return false;

    // All ones in "lhs_dilation" means this "mhlo.conv" op should be
    // converted to "tf.Conv2D" or "tf.DepthwiseConv2dNativeOp".
    if (conv_op.lhs_dilation().hasValue()) {
      auto lhs_dilation = conv_op.lhs_dilation().getValue();
      if (!lhs_dilation.isSplat() || lhs_dilation.getSplatValue<int64_t>() != 1)
        return false;
    }

    if (!conv_op.window_strides().hasValue() || conv_op.window_strides()
                                                        .getValue()
                                                        .getType()
                                                        .cast<ShapedType>()
                                                        .getRank() != 1)
      return false;

    int num_spatial_dims =
        conv_op.dimension_numbers().input_spatial_dimensions().getNumElements();
    // TODO(b/158636600): Currently we don't support 3D Convolution.
    if (num_spatial_dims != 2) return false;

    // TODO(chhe): To support more data formats other than "NHWC".
    // Checks input dimensions.
    if (conv_op.dimension_numbers().input_batch_dimension().getInt() != 0 ||
        conv_op.dimension_numbers().input_feature_dimension().getInt() !=
            num_spatial_dims + 1)
      return false;
    DenseIntElementsAttr input_spatial_dimensions =
        conv_op.dimension_numbers().input_spatial_dimensions();
    for (auto p :
         llvm::enumerate(input_spatial_dimensions.getValues<int64_t>())) {
      if (p.value() != p.index() + 1) return false;
    }

    // Checks output dimensions.
    if (conv_op.dimension_numbers().output_batch_dimension().getInt() != 0 ||
        conv_op.dimension_numbers().output_feature_dimension().getInt() !=
            num_spatial_dims + 1)
      return false;
    DenseIntElementsAttr output_spatial_dimensions =
        conv_op.dimension_numbers().output_spatial_dimensions();
    for (auto p :
         llvm::enumerate(output_spatial_dimensions.getValues<int64_t>())) {
      if (p.value() != p.index() + 1) return false;
    }

    // Checks kernel dimensions.
    if (conv_op.dimension_numbers().kernel_input_feature_dimension().getInt() !=
            num_spatial_dims ||
        conv_op.dimension_numbers()
                .kernel_output_feature_dimension()
                .getInt() != num_spatial_dims + 1)
      return false;
    DenseIntElementsAttr kernal_spatial_dimensions =
        conv_op.dimension_numbers().kernel_spatial_dimensions();
    for (auto p :
         llvm::enumerate(kernal_spatial_dimensions.getValues<int64_t>())) {
      if (p.value() != p.index()) return false;
    }

    return true;
  }
};

class ConvertSliceOp : public OpConversionPattern<mhlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::SliceOp slice_op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    DenseIntElementsAttr strides = slice_op.strides();
    // Strides must be 1 otherwise we cannot legalize this `mhlo.slice` op.
    if (!strides.isSplat() ||
        strides.getSplatValue().cast<IntegerAttr>().getInt() != 1)
      return failure();

    rewriter.setInsertionPointAfter(slice_op.getOperation());
    auto start_indices = slice_op.start_indices();
    auto limit_indices = slice_op.limit_indices();
    std::vector<int64_t> size_values;
    for (auto pair : llvm::zip(start_indices.getValues<APInt>(),
                               limit_indices.getValues<APInt>())) {
      size_values.emplace_back(std::get<1>(pair).getSExtValue() -
                               std::get<0>(pair).getSExtValue());
    }

    RankedTensorType ty =
        RankedTensorType::get({static_cast<int64_t>(size_values.size())},
                              rewriter.getIntegerType(64));
    auto start = rewriter.create<ConstOp>(slice_op.getLoc(), start_indices);
    auto size = rewriter.create<ConstOp>(
        slice_op.getLoc(), DenseIntElementsAttr::get(ty, size_values));
    rewriter.replaceOpWithNewOp<SliceOp>(slice_op, slice_op.getType(),
                                         slice_op.operand(), start, size);
    return success();
  };
};

class ConvertDynamicSliceOp : public OpConversionPattern<mhlo::DynamicSliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicSliceOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    ShapedType input_type = op.operand().getType().cast<ShapedType>();
    if (!input_type.hasStaticShape()) return failure();
    Type start_indices_element_type = op.start_indices()
                                          .front()
                                          .getType()
                                          .cast<ShapedType>()
                                          .getElementType();

    // Clamp indices to [0, input_size - output_size]
    llvm::SmallVector<Value, 4> start_indices_vector;
    start_indices_vector.reserve(op.start_indices().size());
    Value clamp_min = rewriter.create<ConstOp>(
        op.getLoc(), rewriter.getIntegerAttr(start_indices_element_type, 0));
    for (uint64_t i = 0, e = op.start_indices().size(); i < e; ++i) {
      Value clamp_max = rewriter.create<ConstOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(start_indices_element_type,
                                  input_type.getShape()[i] -
                                      op.slice_sizes().getValue<int64_t>({i})));
      Value clamped_index = rewriter.create<mhlo::ClampOp>(
          op.getLoc(), op.start_indices()[i].getType(), op.start_indices()[i],
          clamp_min, clamp_max);
      start_indices_vector.push_back(clamped_index);
    }

    // Pack individual start indices to start indices tensor.
    Type start_indices_type = RankedTensorType::get(
        {static_cast<int64_t>(start_indices_vector.size())},
        start_indices_element_type);
    Value start_indices_op = rewriter.create<PackOp>(
        op.getLoc(), start_indices_type, ValueRange(start_indices_vector));

    Value slice_sices_op =
        rewriter.create<ConstOp>(op.getLoc(), op.slice_sizes());
    rewriter.replaceOpWithNewOp<SliceOp>(op, op.getType(), op.operand(),
                                         start_indices_op, slice_sices_op);
    return success();
  };
};

// Appends all elements in `range` to `values`.
template <typename ValueT, typename Range>
void Append(llvm::SmallVectorImpl<ValueT> &values, Range &&range) {
  values.insert(values.end(), range.begin(), range.end());
}

// Appends all elements in `range` to `values`.
template <typename ValueT, typename Range, typename... RangeTs>
void Append(llvm::SmallVectorImpl<ValueT> &values, Range &&range,
            RangeTs &&...ranges) {
  values.insert(values.end(), range.begin(), range.end());
  Append(values, ranges...);
}

// Returns the number of elements in `range`.
template <typename Range>
size_t Size(Range &&range) {
  return range.size();
}

// Returns the total number of elements in a variadic number of `ranges`.
template <typename Range, typename... RangeTs>
size_t Size(Range &&range, RangeTs &&...ranges) {
  return range.size() + Size(std::forward<RangeTs>(ranges)...);
}

// Concats all elements in `ranges` and returns a small vector as a result.
template <typename ValueT, typename... RangeTs>
llvm::SmallVector<ValueT, 4> Concat(RangeTs &&...ranges) {
  llvm::SmallVector<int64_t, 4> results;
  results.reserve(Size(std::forward<RangeTs>(ranges)...));
  Append(results, std::forward<RangeTs>(ranges)...);
  return results;
}

// A struct to hold axes and sizes for a set of dimensions.
struct DimensionVector {
  llvm::ArrayRef<int64_t> AxesArray() const { return axes; }
  llvm::ArrayRef<int64_t> SizesArray() const { return sizes; }

  llvm::SmallVector<int64_t, 4> axes;
  llvm::SmallVector<int64_t, 4> sizes;
};

// A struct to hold information about dimensions of dot_general operands.
class DotDimensionsInfo {
 public:
  DotDimensionsInfo(ShapedType type, DenseIntElementsAttr batch_dimensions,
                    DenseIntElementsAttr contracting_dimensions) {
    const int rank = type.getRank();
    for (const int dim : batch_dimensions.getValues<int64_t>()) {
      batch_dimensions_.axes.push_back(dim);
      batch_dimensions_.sizes.push_back(type.getDimSize(dim));
    }

    for (const int dim : contracting_dimensions.getValues<int64_t>()) {
      contracting_dimensions_.axes.push_back(dim);
      contracting_dimensions_.sizes.push_back(type.getDimSize(dim));
    }

    for (int dim = 0; dim < rank; ++dim) {
      if (llvm::count(contracting_dimensions_.axes, dim) > 0 ||
          llvm::count(batch_dimensions_.axes, dim) > 0) {
        continue;
      }
      out_dimensions_.axes.push_back(dim);
      out_dimensions_.sizes.push_back(type.getDimSize(dim));
    }
  }

  const DimensionVector &batch_dimensions() const { return batch_dimensions_; }
  const DimensionVector &contracting_dimensions() const {
    return contracting_dimensions_;
  }
  // Out dimensions are any dimensions that are neither batch nor contracting
  // dimensions, hence will be propagated to output shape.
  const DimensionVector &out_dimensions() const { return out_dimensions_; }

  // Returns the total dimension size after flattening all contracting
  // dimensions.
  int FlattenedContractingDimensionSize() const {
    return std::accumulate(contracting_dimensions_.sizes.begin(),
                           contracting_dimensions_.sizes.end(), 1,
                           std::multiplies<int64_t>());
  }

  // Returns the total dimension size after flattening all out dimensions.
  int FlattenedOutDimensionSize() const {
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

Value ConvertDot(PatternRewriter &rewriter, Value lhs, Value rhs,
                 DotDimensionNumbers dot_dimension_numbers,
                 ShapedType result_type, mlir::Location loc) {
  auto lhs_type = lhs.getType().cast<ShapedType>();
  auto rhs_type = rhs.getType().cast<ShapedType>();
  const int lhs_rank = lhs_type.getRank();
  const int rhs_rank = rhs_type.getRank();

  // Collects lhs and rhs dimensions information.
  DotDimensionsInfo lhs_dot_dimensions_info(
      lhs_type, dot_dimension_numbers.lhs_batching_dimensions(),
      dot_dimension_numbers.lhs_contracting_dimensions());
  DotDimensionsInfo rhs_dot_dimensions_info(
      rhs_type, dot_dimension_numbers.rhs_batching_dimensions(),
      dot_dimension_numbers.rhs_contracting_dimensions());

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
  auto lhs_flattend = rewriter.create<mhlo::ReshapeOp>(
      loc,
      RankedTensorType::get(lhs_flattened_shape, lhs_type.getElementType()),
      lhs_transposed.getResult());

  // Reshapes rhs to flatten out_dimensions and contracting_dimensions.
  llvm::SmallVector<int64_t, 4> rhs_flattened_shape = Concat<int64_t>(
      rhs_dot_dimensions_info.batch_dimensions().SizesArray(),
      llvm::ArrayRef<int64_t>{
          rhs_dot_dimensions_info.FlattenedContractingDimensionSize()},
      llvm::ArrayRef<int64_t>{
          rhs_dot_dimensions_info.FlattenedOutDimensionSize()});
  auto rhs_flattend = rewriter.create<mhlo::ReshapeOp>(
      loc,
      RankedTensorType::get(rhs_flattened_shape, rhs_type.getElementType()),
      rhs_transposed.getResult());

  // Creates matmul op of `lhs_flattend` and `rhs_flattend`.
  llvm::SmallVector<int64_t, 4> matmul_shape =
      Concat<int64_t>(lhs_dot_dimensions_info.batch_dimensions().SizesArray(),
                      llvm::ArrayRef<int64_t>{
                          lhs_dot_dimensions_info.FlattenedOutDimensionSize()},
                      llvm::ArrayRef<int64_t>{
                          rhs_dot_dimensions_info.FlattenedOutDimensionSize()});
  auto matmul = rewriter.create<TF::BatchMatMulV2Op>(
      loc, RankedTensorType::get(matmul_shape, result_type.getElementType()),
      lhs_flattend.getResult(), rhs_flattend.getResult());
  auto reshaped =
      rewriter.create<mhlo::ReshapeOp>(loc, result_type, matmul.getResult());
  return reshaped.getResult();
}

// Converts mhlo.dot to tf.MatMul. Reshape ops will be inserted when
// necessary.
Value ConvertDotOp(PatternRewriter &rewriter, Operation *old_op) {
  auto dot_op = cast<mhlo::DotOp>(old_op);
  auto lhs_rank = dot_op.lhs().getType().cast<ShapedType>().getRank();
  auto dot_dimension_numbers = DotDimensionNumbers::get(
      /*lhs_batching_dimensions=*/rewriter.getI64TensorAttr({}),
      /*rhs_batching_dimensions=*/rewriter.getI64TensorAttr({}),
      /*lhs_contracting_dimensions=*/
      rewriter.getI64TensorAttr({lhs_rank == 1 ? 0 : 1}),
      /*rhs_contracting_dimensions=*/rewriter.getI64TensorAttr({0}),
      rewriter.getContext());
  return ConvertDot(rewriter, dot_op.lhs(), dot_op.rhs(), dot_dimension_numbers,
                    dot_op.getResult().getType().cast<ShapedType>(),
                    dot_op.getLoc());
}

// Converts mhlo.dot to tf.BatchMatMul. Reshape or Transpose ops will also be
// inserted to convert to well-formed matrix multiply.
Value ConvertDotGeneralOp(PatternRewriter &rewriter, Operation *old_op) {
  auto dot_general_op = cast<mhlo::DotGeneralOp>(old_op);
  return ConvertDot(rewriter, dot_general_op.lhs(), dot_general_op.rhs(),
                    dot_general_op.dot_dimension_numbers(),
                    dot_general_op.getResult().getType().cast<ShapedType>(),
                    dot_general_op.getLoc());
}

// Checks if the specified region is a binary reduction function what takes 2
// inputs, passes it to an instance of the specifiied reduction op and then
// returns the result.
template <typename ReductionOp>
LogicalResult MatchBinaryReduceFunction(mlir::Region &function) {
  Block &body = function.front();
  if (body.getNumArguments() != 2) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op) return failure();
  if (return_op.getNumOperands() != 1) return failure();

  ReductionOp reduce_op = dyn_cast_or_null<ReductionOp>(
      return_op.getOperands().front().getDefiningOp());
  if (!reduce_op) return failure();
  if (reduce_op.lhs() != body.getArgument(0) ||
      reduce_op.rhs() != body.getArgument(1))
    return failure();

  return success();
}

// Check if the specified region is a binary reduction function what takes 2
// inputs and returns the second input. Functions like this are used by update
// scatter like ops.
template <>
LogicalResult MatchBinaryReduceFunction<void>(mlir::Region &function) {
  Block &body = function.front();
  if (body.getNumArguments() != 2) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op) return failure();
  if (return_op.getNumOperands() != 1) return failure();
  if (return_op.getOperands().front() != body.getArgument(1)) return failure();
  return success();
}

// Converts an mhlo.reduce op with the specified BinaryOp as the reduction
// operation into the specified TfOp.
template <typename BinaryOp, typename TfOp>
class ConvertReduceOpToTfOp : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    if (failed(MatchReduceOpInput(reduce_op))) return failure();

    if (failed(MatchBinaryReduceFunction<BinaryOp>(reduce_op.body())))
      return failure();

    // In `MatchReduceOpInput` function, we already match that the
    // "mhlo::ReduceOp" only has one input, one init_value and one result.
    if (failed(MatchInitValue(reduce_op.init_values()[0]))) return failure();

    auto input = reduce_op.operands()[0];

    // Get reduction dimension.
    DenseIntElementsAttr dimension = reduce_op.dimensions();
    SmallVector<int64_t, 4> reduce_dims;
    for (const int64_t &dim : dimension.getValues<int64_t>()) {
      reduce_dims.emplace_back(dim);
    }
    auto dim_type = RankedTensorType::get(
        {static_cast<int64_t>(reduce_dims.size())}, rewriter.getI64Type());
    auto reduction_indices = rewriter.create<ConstOp>(
        reduce_op.getLoc(), dim_type, rewriter.getI64TensorAttr(reduce_dims));

    rewriter.replaceOpWithNewOp<TfOp>(reduce_op, reduce_op.getType(0), input,
                                      reduction_indices,
                                      /*keep_dim=*/rewriter.getBoolAttr(false));
    return success();
  }

 private:
  // Checks that the init value matches with the init value expected for the
  // target TfOp.
  virtual LogicalResult MatchInitValue(Value init_value) const = 0;

  // This function tries to match that the "mhlo::ReduceOp" only has one
  // input, one init_value and one result.
  LogicalResult MatchReduceOpInput(mhlo::ReduceOp reduce_op) const {
    if (reduce_op.operands().size() != 1 ||
        reduce_op.init_values().size() != 1 ||
        reduce_op.getResults().size() != 1)
      return failure();

    if (!reduce_op.operands()[0].getType().isa<RankedTensorType>())
      return failure();
    if (!reduce_op.getType(0).isa<RankedTensorType>()) return failure();
    return success();
  }
};

class ConvertReduceOpToTfSum
    : public ConvertReduceOpToTfOp<mhlo::AddOp, TF::SumOp> {
 public:
  using ConvertReduceOpToTfOp::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    DenseFPElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.isSplat() || !init_attr.getSplatValue<APFloat>().isZero())
      return failure();
    return success();
  }
};

class ConvertReduceOpToTfMax
    : public ConvertReduceOpToTfOp<mhlo::MaxOp, TF::MaxOp> {
 public:
  using ConvertReduceOpToTfOp::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    DenseFPElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.isSplat() ||
        !init_attr.getSplatValue<APFloat>().isInfinity() ||
        !init_attr.getSplatValue<APFloat>().isNegative())
      return failure();
    return success();
  }
};

class ConvertReduceOpToTfMin
    : public ConvertReduceOpToTfOp<mhlo::MinOp, TF::MinOp> {
 public:
  using ConvertReduceOpToTfOp::ConvertReduceOpToTfOp;

  LogicalResult MatchInitValue(Value init_value) const override {
    DenseFPElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.isSplat() ||
        !init_attr.getSplatValue<APFloat>().isInfinity() ||
        init_attr.getSplatValue<APFloat>().isNegative())
      return failure();
    return success();
  }
};

template <typename TfReduce, typename TfArgReduce>
class ConvertReduceOpToTfArgMinMax
    : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    if (reduce_op.operands().size() != 2) return failure();
    if (reduce_op.dimensions().getNumElements() != 1) return failure();

    // Check that the input init is the expected value.
    DenseElementsAttr input_init;
    if (!matchPattern(reduce_op.init_values().front(), m_Constant(&input_init)))
      return failure();
    if (!IsValueInitValue(input_init)) return failure();

    // Check that the iota init is zero.
    DenseElementsAttr iota_init;
    if (!matchPattern(reduce_op.init_values().back(), m_Constant(&iota_init)))
      return failure();
    if (*iota_init.getIntValues().begin() != 0) return failure();

    // Verify that the second argument is an Iota op along the same dimenion as
    // the reduction.
    Value iota = reduce_op.operands().back();
    mhlo::BroadcastInDimOp iota_broadcast =
        llvm::dyn_cast_or_null<mhlo::BroadcastInDimOp>(iota.getDefiningOp());
    if (!iota_broadcast ||
        iota_broadcast.broadcast_dimensions() != reduce_op.dimensions())
      return failure();
    if (!llvm::isa<mhlo::IotaOp>(iota_broadcast.operand().getDefiningOp()))
      return failure();

    // Match the reduction computation.
    if (failed(matchReduceComputation(reduce_op.body()))) return failure();

    Value input = reduce_op.operands().front();
    int64_t axis = reduce_op.dimensions().getValue<int64_t>({0});

    auto dim_type = RankedTensorType::get({1}, rewriter.getI64Type());
    auto reduction_indices = rewriter.create<ConstOp>(
        reduce_op.getLoc(), dim_type, rewriter.getI64TensorAttr({axis}));

    // Generate a Max and an ArgMax of as the mhlo op returns both while in TF
    // we have separate ops for them. If only one of them is used then the other
    // one will be garbage collected later.
    auto result_type = reduce_op.getType(0).cast<TupleType>();
    auto tf_reduce_op = rewriter.create<TfReduce>(
        reduce_op.getLoc(), result_type.getType(0), input, reduction_indices,
        /*keep_dim=*/rewriter.getBoolAttr(false));
    auto tf_argreduce_op = rewriter.create<TfArgReduce>(
        reduce_op.getLoc(), result_type.getType(1), input, reduction_indices);

    // Pack the result into a TupleOp to match return type. The Tuple will be
    // optimised out by a subsequent pass.
    SmallVector<Value, 2> result{tf_reduce_op, tf_argreduce_op};
    rewriter.replaceOpWithNewOp<mhlo::TupleOp>(reduce_op, result);
    return success();
  }

  // Pattern matches the following reduction function for ArgMax/ArgMin:
  // %0 = compare{GT}(%lhs_value, %rhs_value)
  // %1 = select(%0, %lhs_value, %rhs_value)
  // %2 = compare{EQ}(%lhs_value, %rhs_value)
  // %3 = compare{LT}(%lhs_index, %rhs_index)
  // %4 = and(%2, %3)
  // %5 = or(%0, %4)
  // %6 = select(%5, %lhs_index, %rhs_index)
  // %7 = tuple(%1, %6)
  // return %7
  LogicalResult matchReduceComputation(Region &computation) const {
    Block &body = computation.front();
    if (body.getNumArguments() != 4) return failure();

    mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
    if (!return_op) return failure();
    if (return_op.getNumOperands() != 1) return failure();

    mhlo::TupleOp return_tuple = llvm::dyn_cast_or_null<mhlo::TupleOp>(
        return_op.getOperand(0).getDefiningOp());
    if (!return_tuple ||
        return_tuple.getType().cast<TupleType>().getTypes().size() != 2)
      return failure();

    mhlo::SelectOp value_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
        return_tuple.getOperand(0).getDefiningOp());
    if (!value_select || value_select.on_true() != body.getArgument(0) ||
        value_select.on_false() != body.getArgument(2))
      return failure();

    mhlo::SelectOp index_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
        return_tuple.getOperand(1).getDefiningOp());
    if (!index_select || index_select.on_true() != body.getArgument(1) ||
        index_select.on_false() != body.getArgument(3))
      return failure();

    mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        value_select.pred().getDefiningOp());
    if (!value_gt || value_gt.comparison_direction() != CompareDirection() ||
        value_gt.lhs() != body.getArgument(0) ||
        value_gt.rhs() != body.getArgument(2))
      return failure();

    mhlo::OrOp index_or =
        llvm::dyn_cast_or_null<mhlo::OrOp>(index_select.pred().getDefiningOp());
    if (!index_or || index_or.lhs() != value_gt) return failure();

    mhlo::AndOp index_and =
        llvm::dyn_cast_or_null<mhlo::AndOp>(index_or.rhs().getDefiningOp());
    if (!index_and) return failure();

    mhlo::CompareOp value_eq = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        index_and.lhs().getDefiningOp());
    if (!value_eq || value_eq.comparison_direction() != "EQ" ||
        value_eq.lhs() != body.getArgument(0) ||
        value_eq.rhs() != body.getArgument(2))
      return failure();

    mhlo::CompareOp index_lt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        index_and.rhs().getDefiningOp());
    if (!index_lt || index_lt.comparison_direction() != "LT" ||
        index_lt.lhs() != body.getArgument(1) ||
        index_lt.rhs() != body.getArgument(3))
      return failure();

    return success();
  }

  virtual const char *CompareDirection() const = 0;

  virtual bool IsValueInitValue(const DenseElementsAttr &attr) const = 0;
};

class ConvertReduceOpToTfArgmax
    : public ConvertReduceOpToTfArgMinMax<TF::MaxOp, TF::ArgMaxOp> {
 public:
  using ConvertReduceOpToTfArgMinMax::ConvertReduceOpToTfArgMinMax;

  const char *CompareDirection() const override { return "GT"; }
  bool IsValueInitValue(const DenseElementsAttr &attr) const override {
    if (attr.getNumElements() != 1 ||
        !attr.getType().getElementType().isa<FloatType>())
      return false;
    auto value = *attr.getFloatValues().begin();
    return value.isNegative() && value.isInfinity();
  }
};

class ConvertReduceOpToTfArgmin
    : public ConvertReduceOpToTfArgMinMax<TF::MinOp, TF::ArgMinOp> {
 public:
  using ConvertReduceOpToTfArgMinMax::ConvertReduceOpToTfArgMinMax;

  const char *CompareDirection() const override { return "LT"; }
  bool IsValueInitValue(const DenseElementsAttr &attr) const override {
    if (attr.getNumElements() != 1 ||
        !attr.getType().getElementType().isa<FloatType>())
      return false;
    auto value = *attr.getFloatValues().begin();
    return !value.isNegative() && value.isInfinity();
  }
};

class ConvertIotaOpToTfRange : public OpConversionPattern<mhlo::IotaOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::IotaOp iota_op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    RankedTensorType type =
        iota_op.getType().dyn_cast_or_null<RankedTensorType>();
    if (!type) return failure();

    const uint64_t dimension = iota_op.iota_dimension();
    Type element_type = type.getElementType();
    Attribute start, limit, delta;
    if (element_type.isa<FloatType>()) {
      start = rewriter.getFloatAttr(element_type, 0.0);
      limit = rewriter.getFloatAttr(element_type, type.getShape()[dimension]);
      delta = rewriter.getFloatAttr(element_type, 1.0);
    } else if (element_type.isa<IntegerType>()) {
      start = rewriter.getIntegerAttr(element_type, 0);
      limit = rewriter.getIntegerAttr(element_type, type.getShape()[dimension]);
      delta = rewriter.getIntegerAttr(element_type, 1);
    } else {
      return failure();
    }

    auto range_type =
        RankedTensorType::get({type.getShape()[dimension]}, element_type);
    Value start_op = rewriter.create<TF::ConstOp>(iota_op.getLoc(), start);
    Value limit_op = rewriter.create<TF::ConstOp>(iota_op.getLoc(), limit);
    Value delta_op = rewriter.create<TF::ConstOp>(iota_op.getLoc(), delta);
    Value result = rewriter.create<TF::RangeOp>(iota_op.getLoc(), range_type,
                                                start_op, limit_op, delta_op);

    if (type.getRank() > 1) {
      std::vector<int64_t> reshape_shape(type.getRank(), 1);
      reshape_shape[iota_op.iota_dimension()] = type.getShape()[dimension];
      auto reshape_type = RankedTensorType::get(reshape_shape, element_type);
      Value reshape_shape_op = rewriter.create<TF::ConstOp>(
          iota_op.getLoc(), rewriter.getI64TensorAttr(reshape_shape));
      result = rewriter.create<TF::ReshapeOp>(iota_op.getLoc(), reshape_type,
                                              result, reshape_shape_op);

      Value broadcast_shape_op = rewriter.create<TF::ConstOp>(
          iota_op.getLoc(), rewriter.getI64TensorAttr(type.getShape()));
      result = rewriter.create<TF::BroadcastToOp>(iota_op.getLoc(), type,
                                                  result, broadcast_shape_op);
    }

    rewriter.replaceOp(iota_op, result);
    return success();
  }
};

// Maps the following represenattions of AvgPool in MHLO into a tf.AvgPool{3D}
// operation when they cleanly map to 2D or 3D average pool with VALID or SAME
// padding:
// * div(reduce_sum_window(x), constant(sizeof(window)))
// * div(reduce_sum_window(x), reduce_sum_window(constant(1)))
class ConvertAvgPoolOp : public OpConversionPattern<mhlo::DivOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DivOp div_op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    auto rw =
        dyn_cast_or_null<mhlo::ReduceWindowOp>(div_op.lhs().getDefiningOp());
    if (!rw) return failure();

    // Check that the reduce-window is a sum-reduce-window.
    if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(rw.body())))
      return failure();

    // Check that this is a floating point reduce window with a rank of 4 or 5.
    RankedTensorType rw_type = rw.getType().dyn_cast<RankedTensorType>();
    if (!rw_type || !rw_type.getElementType().isa<FloatType>() ||
        rw_type.getRank() <= 3 || rw_type.getRank() > 5)
      return failure();

    // Check that the Div op doesn't do broadcasting on the output of the reduce
    // window.
    if (div_op.getType() != rw.getType()) return failure();

    // tf.avg_pool need at least 3 dimensions (batch, spatial, channel)
    const uint64_t rank = rw.window_dimensions().size();
    if (rank <= 2) return failure();

    // If the init value isn't zero then it can't be an average pool.
    if (!isFloatZero(rw.init_value())) return failure();

    llvm::SmallVector<int64_t, 5> window_strides;
    if (rw.window_strides().hasValue()) {
      window_strides.insert(window_strides.end(),
                            rw.window_strides()->getValues<int64_t>().begin(),
                            rw.window_strides()->getValues<int64_t>().end());
    } else {
      window_strides.resize(rank, 1);
    }

    llvm::SmallVector<int64_t, 10> padding;
    if (rw.padding().hasValue()) {
      padding.insert(padding.begin(),
                     rw.padding()->getValues<int64_t>().begin(),
                     rw.padding()->getValues<int64_t>().end());
    } else {
      padding.resize(2 * rank, 0);
    }

    // Check that we don't do any reduction along the batch (first) and channel
    // (last) dimensions.
    const uint64_t batch_dim = 0;
    const uint64_t channel_dim = rank - 1;
    if (rw.window_dimensions().getValue<int64_t>({batch_dim}) != 1 ||
        rw.window_dimensions().getValue<int64_t>({channel_dim}) != 1 ||
        window_strides[batch_dim] != 1 || window_strides[channel_dim] != 1 ||
        padding[2 * batch_dim] != 0 || padding[2 * batch_dim + 1] != 0 ||
        padding[2 * channel_dim] != 0 || padding[2 * channel_dim + 1] != 0)
      return failure();

    if (rw.window_dilations().hasValue() &&
        !(rw.window_dilations()->isSplat() &&
          rw.window_dilations()->getSplatValue<APInt>() == 1))
      return failure();

    if (rw.base_dilations().hasValue() &&
        !(rw.base_dilations()->isSplat() &&
          rw.base_dilations()->getSplatValue<APInt>() == 1))
      return failure();

    DenseFPElementsAttr divisor;
    if (matchPattern(div_op.rhs(), m_Constant(&divisor))) {
      // If the divisor is a constant then check that it matches with the number
      // of elements inside the window what is required for a VALID AvgPool.
      if (!divisor.isSplat()) return failure();
      int64_t window_size = 1;
      for (int64_t w : rw.window_dimensions().getValues<int64_t>()) {
        window_size *= w;
      }
      if (!divisor.getSplatValue<APFloat>().isExactlyValue(window_size))
        return failure();

      // Check that we have no padding.
      if (!llvm::all_of(padding, [](int64_t i) { return i == 0; }))
        return failure();

      return replaceWithAvgPool(
          div_op, rw.operand(),
          llvm::to_vector<4>(rw.window_dimensions().getValues<int64_t>()),
          window_strides, "VALID", rewriter);
    }

    auto rw_rhs =
        dyn_cast_or_null<mhlo::ReduceWindowOp>(div_op.rhs().getDefiningOp());
    if (rw_rhs) {
      // Check that RHS is a sum-reduce-window.
      if (failed(MatchBinaryReduceFunction<mhlo::AddOp>(rw_rhs.body())))
        return failure();

      // Check that the RHS is a reduce_window over a constant 1 input with 0 as
      // the init value.
      DenseFPElementsAttr rhs_input;
      if (!isFloatZero(rw_rhs.init_value()) ||
          !matchPattern(rw_rhs.operand(), m_Constant(&rhs_input)) ||
          !rhs_input.isSplat() ||
          !rhs_input.getSplatValue<APFloat>().isExactlyValue(1.0))
        return failure();

      // Check that the two reduce window have the same window configuration.
      if (rw.window_dimensions() != rw_rhs.window_dimensions() ||
          rw.window_strides() != rw_rhs.window_strides() ||
          rw.window_dilations() != rw_rhs.window_dilations() ||
          rw.base_dilations() != rw_rhs.base_dilations() ||
          rw.padding() != rw_rhs.padding())
        return failure();

      if (llvm::all_of(padding, [](int64_t i) { return i == 0; }))
        return replaceWithAvgPool(
            div_op, rw.operand(),
            llvm::to_vector<4>(rw.window_dimensions().getValues<int64_t>()),
            window_strides, "VALID", rewriter);

      RankedTensorType input_type =
          rw.operand().getType().dyn_cast<RankedTensorType>();
      RankedTensorType output_type = rw.getType().dyn_cast<RankedTensorType>();
      if (!input_type || !output_type) return failure();

      // Check that the individual padding values are corresponding to SAME
      // padding from TensorFlow.
      for (uint64_t i = 1; i < rank - 1; ++i) {
        int64_t padding_size =
            (output_type.getShape()[i] - 1) * window_strides[i] +
            rw.window_dimensions().getValue<int64_t>({i}) -
            input_type.getShape()[i];
        if (padding[2 * i] !=
                tensorflow::MathUtil::FloorOfRatio(padding_size, int64_t(2)) ||
            padding[2 * i + 1] !=
                tensorflow::MathUtil::CeilOfRatio(padding_size, int64_t(2)))
          return failure();
      }
      return replaceWithAvgPool(
          div_op, rw.operand(),
          llvm::to_vector<4>(rw.window_dimensions().getValues<int64_t>()),
          window_strides, "SAME", rewriter);
    }
    return failure();
  }

 private:
  bool isFloatZero(Value value) const {
    DenseFPElementsAttr initial_value;
    return matchPattern(value, m_Constant(&initial_value)) &&
           initial_value.getNumElements() == 1 &&
           initial_value.getValue<APFloat>({}).isZero();
  }

  LogicalResult replaceWithAvgPool(mhlo::DivOp op, Value input,
                                   llvm::ArrayRef<int64_t> ksizes,
                                   llvm::ArrayRef<int64_t> kstrides,
                                   llvm::StringRef padding,
                                   ConversionPatternRewriter &rewriter) const {
    if (ksizes.size() == 4) {
      rewriter.replaceOpWithNewOp<AvgPoolOp>(
          op, op.getType(), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          rewriter.getStringAttr("NHWC"));
      return success();
    } else if (ksizes.size() == 5) {
      rewriter.replaceOpWithNewOp<AvgPool3DOp>(
          op, op.getType(), input, rewriter.getI64ArrayAttr(ksizes),
          rewriter.getI64ArrayAttr(kstrides), rewriter.getStringAttr(padding),
          rewriter.getStringAttr("NDHWC"));
      return success();
    }
    return failure();
  }
};

class LegalizeHloToTf : public PassWrapper<LegalizeHloToTf, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

 public:
  LegalizeHloToTf() = default;
  LegalizeHloToTf(const LegalizeHloToTf &) {}

  /// Performs the legalization to the TF dialect.
  void runOnFunction() override;
};

// Returns the shape of the given value in a Constant Op.
ConstantOp ShapeToConst(PatternRewriter &rewriter, Value value) {
  ArrayRef<int64_t> shape = value.getType().cast<ShapedType>().getShape();
  auto attr_type = RankedTensorType::get({static_cast<int64_t>(shape.size())},
                                         rewriter.getIntegerType(64));
  auto attr = DenseElementsAttr::get(attr_type, shape);
  return rewriter.create<ConstantOp>(value.getLoc(), attr_type, attr);
}

// If index_vector_dim == indices.rank() then insert the implicit extra
// dimension into indices to normalize everything to index_vector_dim ==
// indices.rank() - 1.
LogicalResult NormalizeIndexVector(Operation *parent_op, Value &indices,
                                   ShapedType &indices_type,
                                   int64_t index_vector_dim,
                                   ConversionPatternRewriter &rewriter) {
  if (index_vector_dim == indices_type.getRank()) {
    llvm::SmallVector<int64_t, 4> new_start_indices_shape(
        indices_type.getShape().begin(), indices_type.getShape().end());
    new_start_indices_shape.push_back(1);
    indices_type = RankedTensorType::get(new_start_indices_shape,
                                         indices_type.getElementType());
    indices = rewriter.create<mhlo::ReshapeOp>(parent_op->getLoc(),
                                               indices_type, indices);
  } else if (index_vector_dim != indices_type.getRank() - 1) {
    // If index_vector_dim isn't the last dimension in indices then it isn't
    // supported yet.
    // TODO(tberghammer): Transpose indices to support this usecase.
    return rewriter.notifyMatchFailure(
        parent_op,
        "index vector dim isn't the last dimension in start indices");
  }
  return success();
}

// Check that `attr` is an R1 iota with integer element type starting from `0`
// with `size` number of values.
bool IsIotaAttr(const DenseIntElementsAttr &attr, int64_t size) {
  if (!attr.getType().getElementType().isa<IntegerType>()) return false;
  if (attr.getType().getRank() != 1) return false;
  if (attr.getNumElements() != size) return false;
  int64_t iota = 0;
  for (auto s : attr.getIntValues()) {
    if (s != iota) return false;
    ++iota;
  }
  return true;
}

class ConvertGatherOp : public OpConversionPattern<mhlo::GatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::GatherOp gather_op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    Value operand = gather_op.operand();
    Value start_indices = gather_op.start_indices();

    // Can only convert with static shaped gather.
    ShapedType operand_type = operand.getType().cast<ShapedType>();
    ShapedType start_indices_type = start_indices.getType().cast<ShapedType>();
    ShapedType result_type = gather_op.getResult().getType().cast<ShapedType>();
    if (!operand_type.hasStaticShape() ||
        !start_indices_type.hasStaticShape() || !result_type.hasStaticShape()) {
      return failure();
    }

    // Normalize start_indices so index_vector_dim == start_indices.rank() - 1.
    int64_t index_vector_dim =
        gather_op.dimension_numbers().index_vector_dim().getInt();
    if (failed(NormalizeIndexVector(gather_op, start_indices,
                                    start_indices_type, index_vector_dim,
                                    rewriter))) {
      return failure();
    }

    // Verify that start_index_map and collapsed_slice_dims contains the same
    // values.
    auto start_index_map = gather_op.dimension_numbers().start_index_map();
    auto collapsed_slice_dims =
        gather_op.dimension_numbers().collapsed_slice_dims();
    if (start_index_map.getNumElements() !=
        collapsed_slice_dims.getNumElements()) {
      return rewriter.notifyMatchFailure(
          gather_op,
          "different size for start index map and collapsed slice dims");
    }
    for (auto c : collapsed_slice_dims) {
      if (llvm::count(start_index_map, c) == 0) {
        return rewriter.notifyMatchFailure(
            gather_op, "collapsed slice dim isn't present in start index map");
      }
    }

    // Verify that slice_sizes is 1 for the indexed dimensions and the full
    // shape for the rest of the dimensions.
    auto slice_sizes = gather_op.slice_sizes();
    int64_t index = 0;
    for (int64_t s : slice_sizes.getValues<int64_t>()) {
      if (llvm::count(start_index_map, index)) {
        if (s != 1) {
          return rewriter.notifyMatchFailure(gather_op,
                                             "unsupported slice sizes");
        }
      } else {
        if (s != operand_type.getShape()[index]) {
          return rewriter.notifyMatchFailure(gather_op,
                                             "unsupported slice sizes");
        }
      }
      ++index;
    }

    // Verify that offset_dims are the tailing dimensions in the output tensor.
    auto offset_dims = gather_op.dimension_numbers().offset_dims();
    int64_t offset = start_indices_type.getRank() - 1;
    for (int64_t o : offset_dims.getValues<int64_t>()) {
      if (o != offset) {
        return rewriter.notifyMatchFailure(gather_op,
                                           "unsupported offset dims");
      }
      ++offset;
    }

    // Transpose the operand to handle non-iota start index map.
    llvm::SmallVector<int64_t, 4> transpose_dimensions;
    llvm::SmallVector<int64_t, 4> transpose_shape;
    for (auto s : start_index_map) {
      transpose_dimensions.push_back(s.getZExtValue());
      transpose_shape.push_back(operand_type.getShape()[s.getZExtValue()]);
    }
    for (int64_t i = 0, e = operand_type.getRank(); i < e; ++i) {
      if (llvm::count(start_index_map, i) == 0) {
        transpose_dimensions.push_back(i);
        transpose_shape.push_back(operand_type.getShape()[i]);
      }
    }
    operand_type =
        RankedTensorType::get(transpose_shape, operand_type.getElementType());
    operand = rewriter.create<mhlo::TransposeOp>(
        gather_op.getLoc(), operand_type, operand,
        rewriter.getI64TensorAttr(transpose_dimensions));

    rewriter.replaceOpWithNewOp<TF::GatherNdOp>(gather_op, result_type, operand,
                                                start_indices);
    return success();
  }
};

template <typename BinaryOp, typename TfOp>
class ConvertScatterOp : public OpConversionPattern<mhlo::ScatterOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp scatter_op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    Value operand = scatter_op.operand();
    Value indices = scatter_op.scatter_indices();
    Value updates = scatter_op.updates();
    ShapedType operand_type = operand.getType().cast<ShapedType>();
    ShapedType indices_type = indices.getType().cast<ShapedType>();
    ShapedType updates_type = updates.getType().cast<ShapedType>();

    // Can only convert with static shaped scatter.
    if (!operand_type.hasStaticShape() || !indices_type.hasStaticShape() ||
        !updates_type.hasStaticShape()) {
      return failure();
    }

    // Normalize start_indices so index_vector_dim == start_indices.rank() - 1.
    int64_t index_vector_dim =
        scatter_op.scatter_dimension_numbers().index_vector_dim().getInt();
    if (failed(NormalizeIndexVector(scatter_op, indices, indices_type,
                                    index_vector_dim, rewriter))) {
      return failure();
    }

    // Verify that inserted_window_dims and scatter_dims_to_operand_dims are
    // both an iota with the same number of elements as the last dimension of
    // start_indices.
    auto inserted_window_dims =
        scatter_op.scatter_dimension_numbers().inserted_window_dims();
    auto scatter_dims_to_operand_dims =
        scatter_op.scatter_dimension_numbers().scatter_dims_to_operand_dims();
    if (!IsIotaAttr(inserted_window_dims, indices_type.getShape().back()) ||
        !IsIotaAttr(scatter_dims_to_operand_dims,
                    indices_type.getShape().back())) {
      // TODO(tberghammer): Transform indices to support non-standard
      // scatter_dims_to_operand_dims.
      return rewriter.notifyMatchFailure(
          scatter_op,
          "unsupported inserted window dims and/or scatter dims to operand "
          "dims");
    }

    // Verify that update window dims are the tailing dimensions in the update
    // tensor.
    auto update_window_dims =
        scatter_op.scatter_dimension_numbers().update_window_dims();
    int64_t offset = indices_type.getRank() - 1;
    for (int64_t o : update_window_dims.getValues<int64_t>()) {
      if (o != offset) {
        return rewriter.notifyMatchFailure(scatter_op,
                                           "unsupported update window dims");
      }
      ++offset;
    }

    // Match the scatter computation against computations supported by TF.
    if (failed(MatchBinaryReduceFunction<BinaryOp>(
            scatter_op.update_computation()))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TfOp>(scatter_op,
                                      scatter_op.getResult().getType(), operand,
                                      indices, updates);
    return success();
  }
};
using ConvertScatterAddOp =
    ConvertScatterOp<mhlo::AddOp, TF::TensorScatterAddOp>;
using ConvertScatterMaxOp =
    ConvertScatterOp<mhlo::MaxOp, TF::TensorScatterMaxOp>;
using ConvertScatterMinOp =
    ConvertScatterOp<mhlo::MinOp, TF::TensorScatterMinOp>;
using ConvertScatterSubOp =
    ConvertScatterOp<mhlo::SubOp, TF::TensorScatterSubOp>;
using ConvertScatterUpdateOp =
    ConvertScatterOp<void, TF::TensorScatterUpdateOp>;

// Converts mhlo.pad to tf.PadV2
Value ConvertPadOp(PatternRewriter &rewriter, Operation *old_op) {
  auto pad_op = cast<mhlo::PadOp>(old_op);
  mlir::Location loc = pad_op.getLoc();

  llvm::SmallVector<APInt, 8> padding;
  for (auto p : llvm::zip(pad_op.edge_padding_low().getValues<APInt>(),
                          pad_op.edge_padding_high().getValues<APInt>())) {
    padding.push_back(std::get<0>(p));
    padding.push_back(std::get<1>(p));
  }
  auto attr_type = RankedTensorType::get({pad_op.edge_padding_low().size(), 2},
                                         rewriter.getI64Type());
  auto padding_attr = DenseIntElementsAttr::get(attr_type, padding);
  auto padding_op = rewriter.create<ConstantOp>(loc, attr_type, padding_attr);
  return rewriter.create<PadV2Op>(loc, pad_op.getType(), pad_op.operand(),
                                  padding_op, pad_op.padding_value());
}

// Returns true if broadcast_dimensions obey Tensorflow convention, as in new
// dimensions are added as prefix.
bool IsTFStyleBroadcast(DenseIntElementsAttr broadcast_dimensions,
                        Value output) {
  // broadcast_dimensions is an increasing list by definition, thus it suffices
  // to check the first element.
  int64_t input_rank = broadcast_dimensions.getNumElements();
  int64_t output_rank = output.getType().cast<ShapedType>().getRank();
  return input_rank == 0 ||
         (broadcast_dimensions.getValue({0}).cast<IntegerAttr>().getInt() ==
          output_rank - input_rank);
}

// Returns the intermediate shape that input tensor should be reshaped to during
// legalization of BroadcastInDimOp.
ConstantOp ExpandedShape(PatternRewriter &rewriter, Value input,
                         DenseIntElementsAttr broadcast_dimensions,
                         Value output) {
  // Initialize expanded shape with output rank and dimensions of 1.
  SmallVector<Attribute, 4> expanded_shape(
      output.getType().cast<ShapedType>().getRank(),
      /*Value=*/rewriter.getI64IntegerAttr(1));

  // Set dimension sizes specified by broadcast_dimensions.
  ArrayRef<int64_t> input_shape = input.getType().cast<ShapedType>().getShape();
  for (auto x : llvm::enumerate(broadcast_dimensions)) {
    expanded_shape[x.value().getSExtValue()] =
        rewriter.getI64IntegerAttr(input_shape[x.index()]);
  }

  // Create the expanded type wrapped in a ConstantOp.
  auto attr_type =
      RankedTensorType::get({static_cast<int64_t>(expanded_shape.size())},
                            rewriter.getIntegerType(64));
  auto attr = DenseElementsAttr::get(attr_type, expanded_shape);
  return rewriter.create<ConstantOp>(output.getLoc(), attr_type, attr);
}

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_legalize_hlo.inc"

/// Performs the lowering to XLA dialect.
void LegalizeHloToTf::runOnFunction() {
  MLIRContext &context = getContext();

  // Add legalization patterns to the list.
  OwningRewritePatternList patterns;
  PopulateLegalizeHloToTfPatterns(&patterns, &context);

  ConversionTarget target(context);
  target.addLegalDialect<TensorFlowDialect>();
  target.addLegalOp<CallOp, ConstantOp>();
  target.addLegalOp<mhlo::TupleOp>();
  if (failed(
          applyPartialConversion(getFunction(), target, std::move(patterns)))) {
    getFunction().emitError("mhlo to TF legalization failed.");
    signalPassFailure();
  }
}

static PassRegistration<LegalizeHloToTf> pass(
    "tf-legalize-hlo", "Legalize from HLO to the TF dialect");

}  // end namespace

void PopulateLegalizeHloToTfPatterns(OwningRewritePatternList *patterns,
                                     MLIRContext *context) {
  patterns->insert<ConvertAvgPoolOp, ConvertConvOp, ConvertDynamicSliceOp,
                   ConvertGatherOp, ConvertScatterAddOp, ConvertScatterMaxOp,
                   ConvertScatterMinOp, ConvertScatterSubOp,
                   ConvertScatterUpdateOp, ConvertSliceOp,
                   ConvertReduceOpToTfArgmax, ConvertReduceOpToTfArgmin,
                   ConvertReduceOpToTfMax, ConvertReduceOpToTfMin,
                   ConvertReduceOpToTfSum, ConvertIotaOpToTfRange>(context);
  populateWithGenerated(context, *patterns);
}

std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeHloToTfPass() {
  return std::make_unique<LegalizeHloToTf>();
}

}  // end namespace TF
}  // end namespace mlir
