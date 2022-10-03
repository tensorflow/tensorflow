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

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/lite/kernels/padding.h"

namespace mlir::quant {
namespace {

Value GetDimValue(OpBuilder &builder, Location loc, Value shape_value,
                  int32_t dim) {
  Type attribute_type = builder.getI64Type();
  return builder.create<TF::StridedSliceOp>(
      loc,
      RankedTensorType::get(
          {},
          shape_value.getType().template cast<ShapedType>().getElementType()),
      /*input=*/shape_value,
      /*begin=*/Create1DConstValue<int32_t>(builder, loc, {dim}),
      /*end=*/Create1DConstValue<int32_t>(builder, loc, {dim + 1}),
      /*strides=*/Create1DConstValue<int32_t>(builder, loc, {1}),
      /*begin_mask=*/builder.getIntegerAttr(attribute_type, 0),
      /*end_mask=*/builder.getIntegerAttr(attribute_type, 0),
      /*ellipsis_mask=*/builder.getIntegerAttr(attribute_type, 0),
      /*new_axis_mask=*/builder.getIntegerAttr(attribute_type, 0),
      /*shrink_axis_mask=*/builder.getIntegerAttr(attribute_type, 1));
}

// Given Value input_size, and known numbers filter_sz, dilation_rate, stride,
// calculate padding_low and padding_high for SAME padding.
void GetSamePaddingValues(OpBuilder &builder, Location loc, Value input_size,
                          int64_t filter_sz, int64_t dilation_rate,
                          int64_t stride, Value &padding_low,
                          Value &padding_high) {
  Value zero = CreateScalarConstValue<int32_t>(builder, loc, 0);
  Value one = CreateScalarConstValue<int32_t>(builder, loc, 1);
  Value two = CreateScalarConstValue<int32_t>(builder, loc, 2);
  Value filter_size = CreateScalarConstValue<int32_t>(builder, loc, filter_sz);
  Type int32_scalar_type = zero.getType();

  auto scalar_add = [&](Value lhs, Value rhs) {
    return builder.create<TF::AddOp>(loc, int32_scalar_type, lhs, rhs);
  };
  auto scalar_mul = [&](Value lhs, Value rhs) {
    return builder.create<TF::MulOp>(loc, int32_scalar_type, lhs, rhs);
  };
  auto scalar_sub = [&](Value lhs, Value rhs) {
    return builder.create<TF::SubOp>(loc, int32_scalar_type, lhs, rhs);
  };
  auto scalar_div = [&](Value lhs, Value rhs) {
    return builder.create<TF::DivOp>(loc, int32_scalar_type, lhs, rhs);
  };

  // effective_filter_size = (filter_size - 1) * dilation_rate + 1
  Value stride_value = CreateScalarConstValue<int32_t>(builder, loc, stride);
  Value dilation_rate_value =
      CreateScalarConstValue<int32_t>(builder, loc, dilation_rate);

  Value effective_filter_size_op = scalar_add(
      scalar_mul(dilation_rate_value, scalar_sub(filter_size, one)), one);

  // output_size = (input_size + stride - 1) / stride
  Value output_size = scalar_div(
      scalar_add(input_size, scalar_sub(stride_value, one)), stride_value);
  // padding_needed = std::max(
  //     0,
  //     (output_size - 1) * stride + effective_filter_size - input_size)
  Value padding_needed = scalar_sub(
      scalar_add(effective_filter_size_op,
                 scalar_mul(stride_value, scalar_sub(output_size, one))),
      input_size);
  padding_needed = builder.create<TF::MaximumOp>(loc, padding_needed, zero);
  padding_low = scalar_div(padding_needed, two);
  padding_high = scalar_sub(padding_needed, padding_low);
}

Value PadForDynamicShapedInputSamePadding(
    OpBuilder &builder, Location loc, Value input, Value filter,
    int8_t input_zp_value, ArrayAttr strides, ArrayAttr dilations,
    StringAttr conv_padding, Value &padding) {
  ShapedType filter_shape = filter.getType().template cast<ShapedType>();
  const int stride_h = strides[1].cast<IntegerAttr>().getInt();
  const int stride_w = strides[2].cast<IntegerAttr>().getInt();
  const int dilation_h = dilations[1].cast<IntegerAttr>().getInt();
  const int dilation_w = dilations[2].cast<IntegerAttr>().getInt();
  const int filter_h = filter_shape.getDimSize(0);
  const int filter_w = filter_shape.getDimSize(1);

  Value input_shape_value = builder.create<TF::ShapeOp>(
      loc, RankedTensorType::get({4}, builder.getI32Type()), input);
  Value input_size_h = GetDimValue(builder, loc, input_shape_value, 1);
  Value pad_h_low, pad_h_high;
  GetSamePaddingValues(builder, loc, input_size_h, filter_h, dilation_h,
                       stride_h, pad_h_low, pad_h_high);
  Value input_size_w = GetDimValue(builder, loc, input_shape_value, 2);
  Value pad_w_low, pad_w_high;
  GetSamePaddingValues(builder, loc, input_size_w, filter_w, dilation_w,
                       stride_w, pad_w_low, pad_w_high);
  padding = CreateConstValue<int32_t>(builder, loc, {2, 2}, {0, 0, 0, 0});
  Value zero = CreateScalarConstValue(builder, loc, 0);
  Value zero_rank1 = CreateConstValue<int32_t>(builder, loc, {1}, {0});
  auto reshape_op = [&](Value value, const SmallVector<int64_t> &shape) {
    const int64_t rank = shape.size();
    return builder.create<TF::ReshapeOp>(
        loc, RankedTensorType::get(shape, builder.getI32Type()), value,
        CreateConstValue<int64_t>(builder, loc, {rank}, shape));
  };
  auto scalar_to_rank1 = [&](Value value) { return reshape_op(value, {1}); };
  Value temp_padding_rank1 = builder.create<TF::ConcatOp>(
      loc, RankedTensorType::get({8}, builder.getI32Type()), zero,
      ArrayRef<Value>({zero_rank1, zero_rank1, scalar_to_rank1(pad_h_low),
                       scalar_to_rank1(pad_h_high), scalar_to_rank1(pad_w_low),
                       scalar_to_rank1(pad_w_high), zero_rank1, zero_rank1}));
  Value temp_padding = reshape_op(temp_padding_rank1, {4, 2});
  return builder.create<TF::PadV2Op>(
      loc, input.getType(), input, temp_padding,
      CreateScalarConstValue<int8_t>(builder, loc, input_zp_value));
}

}  // namespace

// If input spatial sizes are dynamic (unknown) and padding is same, add ops to
// dynamically calculate padding size and add input_zp value Pad op with the
// padding.
// Otherwise, calculates padding with known numbers, and only for non-zero
// padding (input_zp != 0), adds Pad op before convolution.
Value CalculatePaddingAndPadIfNeeded(
    OpBuilder &builder, Location loc, Value input, Value filter,
    int8_t input_zp_value, ArrayAttr strides, ArrayAttr dilations,
    StringAttr conv_padding, ArrayAttr explicit_paddings, Value &padding) {
  ShapedType input_shape = input.getType().template cast<ShapedType>();

  if (conv_padding.strref().equals("SAME") &&
      (input_shape.isDynamicDim(1) || input_shape.isDynamicDim(2))) {
    return PadForDynamicShapedInputSamePadding(
        builder, loc, input, filter, input_zp_value, strides, dilations,
        conv_padding, padding);
  }

  ShapedType filter_shape = filter.getType().template cast<ShapedType>();

  int padding_h_before, padding_h_after, padding_w_before, padding_w_after;
  if (conv_padding.strref().equals("EXPLICIT")) {
    if (explicit_paddings.size() != 8) {
      emitError(loc, "explicit_paddings are expected to be 8-element arrays");
      return {};
    }
    padding_h_before = explicit_paddings[2].cast<IntegerAttr>().getInt();
    padding_h_after = explicit_paddings[3].cast<IntegerAttr>().getInt();
    padding_w_before = explicit_paddings[4].cast<IntegerAttr>().getInt();
    padding_w_after = explicit_paddings[5].cast<IntegerAttr>().getInt();
  } else if (conv_padding.strref().equals("VALID")) {
    padding_h_before = 0;
    padding_h_after = 0;
    padding_w_before = 0;
    padding_w_after = 0;
  } else {
    // conv_padding is "SAME".
    int output_height, output_width;
    const int stride_h = strides[1].cast<IntegerAttr>().getInt();
    const int stride_w = strides[2].cast<IntegerAttr>().getInt();
    const int dilation_h = dilations[1].cast<IntegerAttr>().getInt();
    const int dilation_w = dilations[2].cast<IntegerAttr>().getInt();
    TfLitePaddingValues padding_values = tflite::ComputePaddingHeightWidth(
        stride_h, stride_w, dilation_h, dilation_w,
        /*in_height=*/input_shape.getDimSize(1),
        /*in_width=*/input_shape.getDimSize(2),
        /*filter_height=*/filter_shape.getDimSize(0),
        /*filter_width=*/filter_shape.getDimSize(1), kTfLitePaddingSame,
        &output_height, &output_width);
    padding_h_before = padding_values.height;
    padding_h_after = padding_values.height + padding_values.height_offset;
    padding_w_before = padding_values.width;
    padding_w_after = padding_values.width + padding_values.width_offset;
  }

  if (input_zp_value == 0 || (padding_h_before == 0 && padding_h_after == 0 &&
                              padding_w_before == 0 && padding_w_after == 0)) {
    padding = CreateConstValue<int32_t>(
        builder, loc, {2, 2},
        {padding_h_before, padding_h_after, padding_w_before, padding_w_after});
    return input;
  }
  padding = CreateConstValue<int32_t>(builder, loc, {2, 2}, {0, 0, 0, 0});

  Value temp_padding =
      CreateConstValue<int32_t>(builder, loc, {4, 2},
                                {0, 0, padding_h_before, padding_h_after,
                                 padding_w_before, padding_w_after, 0, 0});
  SmallVector<int64_t> output_shape(input_shape.getShape().begin(),
                                    input_shape.getShape().end());
  output_shape[1] += padding_h_before + padding_h_after;
  output_shape[2] += padding_w_before + padding_w_after;

  return builder.create<TF::PadV2Op>(
      loc, RankedTensorType::get(output_shape, builder.getI8Type()), input,
      temp_padding,
      CreateScalarConstValue<int8_t>(builder, loc, input_zp_value));
}

// Pack value using following formula:
// Consider value of rank=4, pack_dim=1 for example.
//
// if value.shape[1] % 2:
//   value = pad(value, [0, 1, 0, 0])
//
// slice_shape = value.shape
// slice_shape[1] /= 2
//
// packed_low = slice(value, [0, 0, 0, 0], slice_shape)
// packed_low = bitwise_and(packed_low, 0x0F)
//
// packed_high = slice(value, [0, value.shape[1] / 2, 0, 0], slice_shape)
// packed_high = left_shift(packed_high, 4)
//
// packed_value = bitwise_or(packed_low, packed_high)
Value PackOperand(OpBuilder &builder, Location loc, Value value, int pack_dim) {
  ShapedType value_type = value.getType().cast<ShapedType>();
  const int rank = value_type.getRank();

  SmallVector<int64_t> packed_shape(value_type.getShape().begin(),
                                    value_type.getShape().end());
  RankedTensorType shape_type =
      RankedTensorType::get({rank}, builder.getI64Type());
  Value shape_value = builder.create<TF::ShapeOp>(loc, shape_type, value);

  // It is guaranteed that packed_shape[pack_dim] is known.
  if (packed_shape[pack_dim] % 2 != 0) {
    packed_shape[pack_dim] += 1;
    llvm::SmallVector<int32_t> padding(rank * 2, 0);
    padding[pack_dim * 2 + 1] = 1;
    Value padding_value =
        CreateConstValue<int32_t>(builder, loc, {rank, 2}, padding);
    value = builder.create<TF::PadV2Op>(
        loc, RankedTensorType::get(packed_shape, builder.getI8Type()), value,
        padding_value, CreateScalarConstValue<int8_t>(builder, loc, 0));

    llvm::SmallVector<int64_t> shape_add(rank, 0);
    shape_add[pack_dim] = 1;
    shape_value = builder.create<TF::AddOp>(
        loc, shape_type, shape_value,
        CreateConstValue<int64_t>(builder, loc, {rank}, shape_add));
  }
  packed_shape[pack_dim] /= 2;
  llvm::SmallVector<int64_t> divisor(rank, 1);
  divisor[pack_dim] = 2;

  RankedTensorType packed_output_type =
      RankedTensorType::get(packed_shape, builder.getI8Type());
  Value packed_shape_value = builder.create<TF::DivOp>(
      loc, shape_type, shape_value,
      CreateConstValue<int64_t>(builder, loc, {rank}, divisor));

  Value packed_low_begin_value = CreateConstValue<int64_t>(
      builder, loc, {rank}, llvm::SmallVector<int64_t>(rank, 0));
  Value packed_low_value =
      builder.create<TF::SliceOp>(loc, packed_output_type, value,
                                  packed_low_begin_value, packed_shape_value);
  packed_low_value = builder.create<TF::BitwiseAndOp>(
      loc, packed_output_type, packed_low_value,
      CreateScalarConstValue<int8_t>(builder, loc, 0x0F));

  llvm::SmallVector<int64_t> packed_high_begin(rank, 0);
  packed_high_begin[pack_dim] = packed_shape[pack_dim];
  Value packed_high_begin_value =
      CreateConstValue<int64_t>(builder, loc, {rank}, packed_high_begin);
  Value packed_high_value =
      builder.create<TF::SliceOp>(loc, packed_output_type, value,
                                  packed_high_begin_value, packed_shape_value);
  packed_high_value = builder.create<TF::LeftShiftOp>(
      loc, packed_output_type, packed_high_value,
      CreateScalarConstValue<int8_t>(builder, loc, 4));

  Operation *packed = builder.create<TF::BitwiseOrOp>(
      loc, packed_output_type, packed_low_value, packed_high_value);
  return ConstantFoldOpIfPossible(packed).front();
}

}  // namespace mlir::quant
