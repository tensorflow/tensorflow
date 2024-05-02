/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_patterns.h"

#include <limits>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verification_utils.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

// ================== Common ========================

// Converts any IntegerAttr to an IntegerAttr of an i32 type.
// The value won't change in the new attribute, but if the value is out of
// the bound of i32, the function returns a failure.
LogicalResult ConvertToI32Attr(IntegerAttr attr, IntegerAttr* attr_i32) {
  if (attr.getType().isInteger(/*width=*/32)) {
    *attr_i32 = attr;
    return success();
  }

  int64_t value = attr.getInt();
  if (value > std::numeric_limits<int>::max() ||
      value < std::numeric_limits<int>::min()) {
    return failure();
  }

  *attr_i32 = IntegerAttr::get(
      IntegerType::get(attr.getContext(), /*width=*/32), value);
  return success();
}

TFL::ReshapeOp InsertReshapeOp(Location loc, Value input, Type element_type,
                               llvm::ArrayRef<int64_t> new_shape_array,
                               OpBuilder* builder) {
  auto reshape_shape_type = mlir::RankedTensorType::get(
      new_shape_array.size(), builder->getIntegerType(32));

  // This is to workaround the unnecessary cast i64 -> i32. :(
  // TODO(renjieliu): Revisit this later.
  SmallVector<int32_t, 4> new_shape_array_i32;
  for (auto size : new_shape_array) {
    new_shape_array_i32.push_back(
        ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size));
  }
  auto new_shape_attr =
      mlir::DenseIntElementsAttr::get(reshape_shape_type, new_shape_array_i32);

  auto new_shape = builder->create<TFL::ConstOp>(loc, new_shape_attr);

  auto reshape_out_type = RankedTensorType::get(new_shape_array, element_type);
  return builder->create<TFL::ReshapeOp>(loc, reshape_out_type, input,
                                         new_shape);
}

LogicalResult EnsureBias(Operation* op, int bias_idx,
                         PatternRewriter& rewriter) {
  auto bias = op->getOperand(bias_idx);

  if (!mlir::isa<NoneType>(bias.getType())) return failure();

  // Proceed to create a zero bias.
  auto output = op->getResult(0);
  auto output_type = mlir::dyn_cast_or_null<RankedTensorType>(output.getType());
  if (!output_type) return failure();

  // bias should be a vector sized of the last output dim.
  int64_t num_units = output_type.getDimSize(output_type.getRank() - 1);
  auto bias_type =
      mlir::RankedTensorType::get({num_units}, output_type.getElementType());

  mlir::DenseElementsAttr bias_attr;
  if (output_type.getElementType().isF32()) {
    float val = 0.0;
    bias_attr = mlir::DenseFPElementsAttr::get(bias_type, val);
  } else {
    // TODO(renjieliu): Refactor this and share the logic with
    // CreateConstOpWithSingleValue. Also, make sure it works with QConst.
    return failure();
  }

  auto zero_bias = rewriter.create<TFL::ConstOp>(op->getLoc(), bias_attr);
  op->setOperand(bias_idx, zero_bias);

  return success();
}

TF::ConstOp PadConstValues(Operation* input_op, int value_to_pad,
                           int pad_dimensions, Location loc,
                           OpBuilder* builder) {
  if (input_op == nullptr) return nullptr;

  mlir::DenseIntElementsAttr attr;
  if (!matchPattern(input_op, m_Constant(&attr))) {
    return nullptr;
  }

  auto value_shape_type = mlir::RankedTensorType::get(
      {pad_dimensions}, builder->getIntegerType(32));

  SmallVector<int32_t, 4> value_i32;
  value_i32.reserve(pad_dimensions);
  for (int i = 0; i < pad_dimensions - attr.getNumElements(); ++i) {
    value_i32.push_back(value_to_pad);
  }
  for (const auto& size : attr) {
    value_i32.push_back(static_cast<int32_t>(size.getSExtValue()));
  }
  auto new_value_i32_attr =
      mlir::DenseIntElementsAttr::get(value_shape_type, value_i32);

  return builder->create<TF::ConstOp>(loc, new_value_i32_attr);
}

SmallVector<Value, 4> SliceOutputs(Operation* split_op, Value input,
                                   RankedTensorType input_type, int split_dim,
                                   int num_splits, PatternRewriter* rewriter) {
  SmallVector<Value, 4> slice_outputs;
  int begin = 0;
  for (int i = 0; i < num_splits; ++i) {
    // Create slice op.
    // Populate begin & size.
    SmallVector<int32_t, 4> slice_begin;
    SmallVector<int32_t, 4> slice_size;
    auto current_output = split_op->getResult(i);
    auto current_output_type =
        mlir::cast<RankedTensorType>(current_output.getType());
    for (int d = 0; d < input_type.getRank(); ++d) {
      if (d == split_dim) {
        // Split dimension.
        slice_begin.push_back(begin);
        int64_t size = current_output_type.getDimSize(d);
        slice_size.push_back(size);
        begin += size;
      } else {
        slice_begin.push_back(0);
        // -1 means every elements.
        slice_size.push_back(-1);
      }
    }

    auto slice_type = mlir::RankedTensorType::get(slice_begin.size(),
                                                  rewriter->getIntegerType(32));
    auto slice_begin_attr =
        mlir::DenseIntElementsAttr::get(slice_type, slice_begin);
    auto slice_size_attr =
        mlir::DenseIntElementsAttr::get(slice_type, slice_size);

    auto slice_begin_const =
        rewriter->create<TFL::ConstOp>(split_op->getLoc(), slice_begin_attr);
    auto slice_size_const =
        rewriter->create<TFL::ConstOp>(split_op->getLoc(), slice_size_attr);

    auto slice_op = rewriter->create<TFL::SliceOp>(
        split_op->getLoc(), current_output_type, input, slice_begin_const,
        slice_size_const);

    // Rewire output.
    slice_outputs.push_back(slice_op.getResult());
  }
  return slice_outputs;
}

}  // namespace

// ================== Pack ========================

LogicalResult LowerPackIntoConcatReshape::matchAndRewrite(
    TFL::PackOp pack_op, PatternRewriter& rewriter) const {
  // Pack op should have same shape type.
  SmallVector<Value, 5> pack_inputs(pack_op.getValues());
  auto input_type = mlir::dyn_cast<RankedTensorType>(pack_inputs[0].getType());
  if (!input_type) return failure();

  // Figure out output shapes.
  SmallVector<int64_t, 4> concat_out_shape;
  SmallVector<int64_t, 4> pack_out_shape;

  const int64_t rank = input_type.getRank();
  int64_t pack_axis = pack_op.getAxis();
  size_t count = pack_inputs.size();
  if (pack_axis < 0) {
    pack_axis += rank;
  }

  // Concat out shape.
  for (int i = 0; i < rank; ++i) {
    int64_t dim_size = input_type.getDimSize(i);
    if (i == pack_axis) {
      dim_size *= count;
    }
    concat_out_shape.push_back(dim_size);
  }

  // Pack out shape.
  int j = 0;
  for (int i = 0; i < rank + 1; ++i) {
    if (i == pack_axis) {
      pack_out_shape.push_back(count);
    } else {
      pack_out_shape.push_back(input_type.getDimSize(j));
      j++;
    }
  }

  if (failed(TF::VerifyShapeOfReshapeOp(pack_out_shape))) return failure();

  // Insert the concat op.
  auto concat_out_type =
      RankedTensorType::get(concat_out_shape, input_type.getElementType());
  auto concat_op = rewriter.create<TFL::ConcatenationOp>(
      pack_op.getLoc(), concat_out_type, pack_inputs, pack_op.getAxis(),
      "NONE");

  auto reshape_op =
      InsertReshapeOp(pack_op.getLoc(), concat_op, input_type.getElementType(),
                      pack_out_shape, &rewriter);

  // Rewire output & get rid of the pack op.
  rewriter.replaceOp(pack_op, reshape_op.getResult());
  return success();
}

// ================== squared_difference ========================

LogicalResult SquaredDifference::matchAndRewrite(
    TFL::SquaredDifferenceOp squared_diff_op, PatternRewriter& rewriter) const {
  auto x = squared_diff_op.getLhs();
  auto y = squared_diff_op.getRhs();
  auto x_type = mlir::dyn_cast<RankedTensorType>(x.getType());
  auto y_type = mlir::dyn_cast<RankedTensorType>(y.getType());
  if (!x_type || !y_type) return failure();
  if (x_type.getShape() != y_type.getShape()) return failure();

  auto result_type = squared_diff_op.getType();
  if (!result_type) return failure();

  auto sub_op =
      rewriter.create<TF::SubOp>(squared_diff_op.getLoc(), result_type, x, y);
  auto mul_op =
      rewriter.create<TF::MulOp>(squared_diff_op.getLoc(), result_type,
                                 sub_op.getResult(), sub_op.getResult());
  rewriter.replaceOp(squared_diff_op, mul_op.getResult());

  return success();
}

// ================== split ========================

LogicalResult UnrollSplit::matchAndRewrite(TFL::SplitOp split_op,
                                           PatternRewriter& rewriter) const {
  auto num_splits = split_op.getNumSplits();
  auto input = split_op.getValue();
  auto input_type = mlir::dyn_cast<RankedTensorType>(input.getType());
  if (input_type == nullptr || !input_type.hasStaticShape()) return failure();

  for (auto result : split_op.getResults()) {
    auto result_type = mlir::dyn_cast<RankedTensorType>(result.getType());
    if (result_type == nullptr) return failure();
  }

  auto output = split_op.getResult(0);
  auto output_type = mlir::cast<RankedTensorType>(output.getType());

  // TODO(renjieliu): change to use split_dim when we raise the constants
  // as well.
  int64_t split_dim = -1;
  for (int64_t d = 0; d < input_type.getRank(); ++d) {
    if (input_type.getDimSize(d) != output_type.getDimSize(d)) split_dim = d;
  }

  const SmallVector<Value, 4>& slice_outputs = SliceOutputs(
      split_op, input, input_type, split_dim, num_splits, &rewriter);
  rewriter.replaceOp(split_op, slice_outputs);
  return success();
}

// ================== splitV ========================

LogicalResult UnrollSplitV::matchAndRewrite(TFL::SplitVOp splitv_op,
                                            PatternRewriter& rewriter) const {
  // We need to make sure both splits & split dim are constants.
  auto splits = splitv_op.getSizeSplits().getDefiningOp();
  mlir::DenseIntElementsAttr splits_attr;
  if (!splits || !matchPattern(splits, m_Constant(&splits_attr)))
    return failure();

  auto split_dim = splitv_op.getSplitDim().getDefiningOp();
  mlir::ElementsAttr split_dim_attr;
  if (!split_dim || !matchPattern(split_dim, m_Constant(&split_dim_attr)))
    return failure();

  auto input = splitv_op.getValue();
  auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!input_type || !input_type.hasRank()) return failure();

  for (auto result : splitv_op.getResults()) {
    auto result_type = mlir::dyn_cast<RankedTensorType>(result.getType());
    if (result_type == nullptr) return failure();
  }

  const int64_t rank = input_type.getRank();

  IntegerAttr dim_int = ExtractSingleElementAsInteger(split_dim_attr);

  // "axis" operand could be a i64 tensor. Resolve it here.
  IntegerAttr dim_i32;
  if (failed(ConvertToI32Attr(dim_int, &dim_i32))) return failure();

  int dim = dim_i32.getInt();
  if (dim < 0) dim += rank;

  const SmallVector<Value, 4>& slice_outputs = SliceOutputs(
      splitv_op, input, input_type, dim, splitv_op.getNumSplits(), &rewriter);
  rewriter.replaceOp(splitv_op, slice_outputs);

  return success();
}

// ================== conv_2d ========================

LogicalResult EnsureBiasForConv2d::matchAndRewrite(
    TFL::Conv2DOp conv_op, PatternRewriter& rewriter) const {
  return EnsureBias(conv_op, 2, rewriter);
}

// ================== slice ============================

// If a slice op has < 4d dimension, will pad it to 4d.
LogicalResult PadSlice::matchAndRewrite(TFL::SliceOp slice_op,
                                        PatternRewriter& rewriter) const {
  // We have to know the shape of the input, as well as the begin/size.
  // also, begin and size have to be constants.
  auto input = slice_op.getInput();
  auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!input_type || !input_type.hasStaticShape()) return failure();

  if (input_type.getRank() >= 4) return failure();

  auto begin = slice_op.getBegin();
  auto begin_type = mlir::dyn_cast_or_null<RankedTensorType>(begin.getType());
  if (!begin_type || !begin_type.hasStaticShape()) return failure();

  auto size = slice_op.getSize();
  auto size_type = mlir::dyn_cast_or_null<RankedTensorType>(size.getType());
  if (!size_type || !size_type.hasStaticShape()) return failure();

  auto output_type =
      mlir::dyn_cast_or_null<RankedTensorType>(slice_op.getType());
  if (!output_type || !output_type.hasStaticShape()) return failure();

  // Pad 0s in front of the begin.
  TF::ConstOp new_begin =
      PadConstValues(begin.getDefiningOp(), 0, 4, slice_op.getLoc(), &rewriter);
  if (!new_begin) return failure();

  // Pad 1s in front of the size.
  TF::ConstOp new_size =
      PadConstValues(size.getDefiningOp(), 1, 4, slice_op.getLoc(), &rewriter);
  if (!new_size) return failure();

  // Reshape the input to 4d.
  SmallVector<int64_t, 4> new_shape;
  const int current_rank = input_type.getRank();
  for (int i = 0; i < 4 - current_rank; ++i) {
    new_shape.push_back(1);
  }
  for (auto size : input_type.getShape()) {
    new_shape.push_back(size);
  }

  auto reshape_op =
      InsertReshapeOp(slice_op.getLoc(), input, input_type.getElementType(),
                      new_shape, &rewriter);

  // Replace with the new slice op.
  SmallVector<int64_t, 4> new_output_shape;
  for (int i = 0; i < 4 - current_rank; ++i) {
    new_output_shape.push_back(1);
  }
  for (auto size : output_type.getShape()) {
    new_output_shape.push_back(size);
  }

  RankedTensorType new_output_type =
      RankedTensorType::get(new_output_shape, output_type.getElementType());

  auto new_slice = rewriter.create<TFL::SliceOp>(
      slice_op.getLoc(), new_output_type, reshape_op, new_begin, new_size);

  // Append a reshape at the bottom.
  auto output_reshape_op = InsertReshapeOp(slice_op.getLoc(), new_slice,
                                           output_type.getElementType(),
                                           output_type.getShape(), &rewriter);
  rewriter.replaceOp(slice_op, output_reshape_op.getResult());

  return success();
}

// ================== fully_connected ========================

// TFL fully_connected basically does:
// Weight * Input + bias.
// Input layout is : [..., depth]
// Weight layout is : [output, depth]
// Bias is [output].
//
// While conv2d is:
// Filter: [NHWC]
// Input is also: [NHWC]
// Bias is [N]
//
// So to perform the transform, we need to insert a few reshape ops:
//
//  Input   weight   bias
//   \      /      /
//       FC
//       |
//     output
//
//     |
//    \/
//
//  Input   weight
//   |        |
//  Reshape  Reshape  bias
//  |         |      /
//     conv
//      |
//     reshape
//      |
//    output
LogicalResult FullyConnectedToConv::matchAndRewrite(
    TFL::FullyConnectedOp fc_op, PatternRewriter& rewriter) const {
  // We have to know the shape of the input.
  auto input = fc_op.getInput();
  auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!input_type || !input_type.hasStaticShape()) return failure();

  // We have to know the shape of the weight.
  auto weight = fc_op.getFilter();
  auto weight_type = mlir::dyn_cast_or_null<RankedTensorType>(weight.getType());
  if (!weight_type || !weight_type.hasStaticShape()) return failure();

  // We have to know the shape of the output as well.
  auto output = fc_op.getResult(0);
  auto output_type = mlir::dyn_cast_or_null<RankedTensorType>(output.getType());
  if (!output_type || !output_type.hasStaticShape()) return failure();

  // Insert a reshape after the input.
  // Since the input maybe more than 2-d, we may collect the flat size of the
  // input then reshape into [1, 1, flat_size / depth, depth].
  const int64_t depth = input_type.getDimSize(input_type.getRank() - 1);
  const int64_t flat_size = input_type.getNumElements();
  const int64_t width = flat_size / depth;
  SmallVector<int64_t, 4> input_new_shape({1, 1, width, depth});
  auto reshaped_input =
      InsertReshapeOp(fc_op.getLoc(), input, input_type.getElementType(),
                      input_new_shape, &rewriter);

  // Insert a reshape after the weight.
  // We will reshape the weight into [output, 1, 1, depth]
  const int64_t output_size = weight_type.getDimSize(0);
  SmallVector<int64_t, 2> weight_new_shape({output_size, 1, 1, depth});
  auto reshaped_weight =
      InsertReshapeOp(fc_op.getLoc(), weight, weight_type.getElementType(),
                      weight_new_shape, &rewriter);

  // Replace the fc with conv.
  // The output would be [1, 1, width, output].
  auto conv_output_type = RankedTensorType::get({1, 1, width, output_size},
                                                output_type.getElementType());
  auto conv = rewriter.create<TFL::Conv2DOp>(
      fc_op.getLoc(), conv_output_type, reshaped_input, reshaped_weight,
      fc_op.getBias(), rewriter.getI32IntegerAttr(1),
      rewriter.getI32IntegerAttr(1), fc_op.getFusedActivationFunctionAttr(),
      rewriter.getStringAttr("VALID"), rewriter.getI32IntegerAttr(1),
      rewriter.getI32IntegerAttr(1));

  // Insert a shape after the conv.
  auto reshaped_conv =
      InsertReshapeOp(fc_op.getLoc(), conv, output_type.getElementType(),
                      output_type.getShape(), &rewriter);

  rewriter.replaceOp(fc_op, reshaped_conv.getResult());

  return success();
}

// ================== concat ============================

// If a concat op has < 4d dimension, will pad it to 4d.
LogicalResult PadConcat::matchAndRewrite(TFL::ConcatenationOp concat_op,
                                         PatternRewriter& rewriter) const {
  int rank = -1;
  for (auto input : concat_op.getValues()) {
    auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
    if (!input_type || !input_type.hasStaticShape()) return failure();

    rank = input_type.getRank();
  }

  auto output_type =
      mlir::dyn_cast_or_null<RankedTensorType>(concat_op.getType());
  if (!output_type || !output_type.hasStaticShape()) return failure();

  if (rank >= 4) return failure();

  // All values should have the same rank.
  // We will insert a reshape op after every input.
  SmallVector<Value, 4> reshape_ops;
  for (auto input : concat_op.getValues()) {
    auto input_type = mlir::cast<RankedTensorType>(input.getType());
    // Get the new shape.
    SmallVector<int64_t, 4> new_shape;
    for (int i = 0; i < 4 - rank; ++i) {
      new_shape.push_back(1);
    }
    for (auto size : input_type.getShape()) {
      new_shape.push_back(size);
    }

    auto reshape_op =
        InsertReshapeOp(concat_op.getLoc(), input, input_type.getElementType(),
                        new_shape, &rewriter);
    reshape_ops.push_back(reshape_op.getResult());
  }

  // Deal with the axis.
  // We don't need to handle axis < 0, since it's counting reversely.
  int32_t axis = concat_op.getAxis();
  if (axis >= 0) {
    axis += (4 - rank);
  }

  // Replace with the new concat op.
  SmallVector<int64_t, 4> new_output_shape;
  for (int i = 0; i < 4 - rank; ++i) {
    new_output_shape.push_back(1);
  }
  for (auto size : output_type.getShape()) {
    new_output_shape.push_back(size);
  }

  RankedTensorType new_output_type =
      RankedTensorType::get(new_output_shape, output_type.getElementType());

  auto new_concat = rewriter.create<TFL::ConcatenationOp>(
      concat_op.getLoc(), new_output_type, reshape_ops, axis,
      concat_op.getFusedActivationFunction());

  // Append a reshape at the bottom.
  auto output_reshape_op = InsertReshapeOp(concat_op.getLoc(), new_concat,
                                           output_type.getElementType(),
                                           output_type.getShape(), &rewriter);
  rewriter.replaceOp(concat_op, output_reshape_op.getResult());

  return success();
}

// ================== mean ========================

// Currently NNAPI does not support mean op with different scales (quantization
// cases), and in TFLite avg_pool will ensure the input & output has the same
// scales.
LogicalResult ReduceMeanToAvgPool::matchAndRewrite(
    TFL::MeanOp mean_op, PatternRewriter& rewriter) const {
  auto input = mean_op.getInput();
  auto input_type = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  // Only 4d is supported here.
  if (!input_type || input_type.getRank() != 4) return failure();

  // The axes has to be [1, 2].
  DenseElementsAttr axis_const;
  if (!matchPattern(mean_op.getAxis(), m_Constant(&axis_const)))
    return failure();
  if (axis_const.size() != 2) return failure();
  auto axis_values = axis_const.getValues<APInt>();
  int i = 1;
  for (auto axis_value : axis_values) {
    if (axis_value != i++) return failure();
  }

  auto output = mean_op.getOutput();
  auto output_type = mlir::dyn_cast_or_null<RankedTensorType>(output.getType());
  if (!output_type) return failure();

  auto input_quantized_type =
      quant::QuantizedType::getQuantizedElementType(input_type);
  auto output_quantized_type =
      quant::QuantizedType::getQuantizedElementType(output_type);
  // If both the input & output types are non-quantized, they will be both
  // nullptrs.
  if (input_quantized_type != output_quantized_type) {
    return failure();
  }

  int64_t batch = input_type.getDimSize(0);
  int64_t height = input_type.getDimSize(1);
  int64_t width = input_type.getDimSize(2);
  int64_t channel = input_type.getDimSize(3);

  auto avg_pool_output_type = RankedTensorType::get(
      {batch, 1, 1, channel}, input_type.getElementType());
  auto avg_pool = rewriter.create<TFL::AveragePool2DOp>(
      mean_op.getLoc(), avg_pool_output_type, input,
      rewriter.getI32IntegerAttr(height), rewriter.getI32IntegerAttr(width),
      rewriter.getStringAttr("VALID"), rewriter.getI32IntegerAttr(1),
      rewriter.getI32IntegerAttr(1), rewriter.getStringAttr("NONE"));

  auto value_to_replace = avg_pool.getResult();

  // If it's not keep dim, we need to insert a reshape after the average
  // pool.
  if (!mean_op.getKeepDims()) {
    // Insert the reshape.
    SmallVector<int64_t, 2> new_shape({batch, channel});
    auto reshape_op =
        InsertReshapeOp(mean_op.getLoc(), avg_pool.getResult(),
                        input_type.getElementType(), new_shape, &rewriter);
    value_to_replace = reshape_op.getResult();
  }

  rewriter.replaceOp(mean_op, value_to_replace);
  return success();
}

// Insert a "requant" op after the mean op if the mean has different scales for
// input & output.
// Please note: THIS IS NOT a mathmetically-equivalent transformation and it may
// loose accuracy, so we need to use this very very carefully.
LogicalResult InsertRequantForReduceMean::matchAndRewrite(
    TFL::MeanOp mean_op, PatternRewriter& rewriter) const {
  auto input = mean_op.getInput();
  auto input_type = mlir::dyn_cast_or_null<ShapedType>(input.getType());
  if (!input_type) return failure();

  // Only need to do this for quantized input.
  auto input_quantized_type =
      quant::QuantizedType::getQuantizedElementType(input_type);
  if (!input_quantized_type) return failure();

  auto output = mean_op.getOutput();
  auto output_type = mlir::dyn_cast_or_null<ShapedType>(output.getType());
  if (!output_type) return failure();
  auto output_quantized_type =
      quant::QuantizedType::getQuantizedElementType(output_type);

  // If the quantized type is the same, we don't need to do anything.
  if (input_quantized_type == output_quantized_type) return failure();

  auto new_output_type =
      RankedTensorType::get(output_type.getShape(), input_quantized_type);
  auto new_mean_op =
      rewriter.create<TFL::MeanOp>(mean_op->getLoc(), new_output_type, input,
                                   mean_op.getAxis(), mean_op.getKeepDims());

  // Insert a requant op.
  rewriter.replaceOpWithNewOp<TFL::QuantizeOp>(
      mean_op, output_type, new_mean_op, mlir::TypeAttr::get(output_type));
  return success();
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
