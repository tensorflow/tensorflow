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

// This is the quantization pattern definition file for TensorFlow Lite.

include "mlir/IR/OpBase.td"
include "mlir/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

// Quantize attribute $0 by using quantization parameter from %1.
def QuantizeByQuantizedType : NativeCodeCall<"Quantize($0, $1.getValue())">;

// Call the generic builder of `op`. Use the result type of $0 in the new op.
class ReplaceWith<string op> : NativeCodeCall<"$_builder.create<" # op #
  ">($0->getLoc(), $0->getResult(0)->getType(), $1, $2, $3)">;

// Squash tfl.dequantize and tfl.quantize pairs.
// TODO(fengliuai): Compare the scale of input and output. This can also be
// squashed to a requantize op if the scales are different.
def : Pat<(TFL_QuantizeOp (TFL_DequantizeOp $in), $qt), (replaceWithValue $in)>;

// Quantize the value of a constant op if the quantization parameters have been
// propagated to the output.
def : Pat<(TFL_QuantizeOp
              (ConstantOp ElementsAttr:$value),
              $qtype),
          (TFL_QConstOp
              $qtype,
              (QuantizeByQuantizedType $value, $qtype))>;

// Quantize the AddOp if both inputs are dequantized and the output is
// quantized.
def : Pat<(TFL_QuantizeOp:$q
              (TFL_AddOp (TFL_DequantizeOp $lhs), (TFL_DequantizeOp $rhs),
                  $fused_activation_function),
              $output_type),
          (ReplaceWith<"TFL::AddOp"> $q, $lhs, $rhs,
              $fused_activation_function)>;

// Quantize the Conv2DOp if the input and weight are dequantized. The scale of
// the bias input is determined by the scales of input and weight operands.
def : Pat<(TFL_QuantizeOp
              (TFL_Conv2DOp
                  (TFL_DequantizeOp $in),
                  (TFL_DequantizeOp $weight),
                  (TFL_DequantizeOp $bias),
                  $dilation_h_factor,
                  $dilation_w_factor,
                  $fused_activation_function,
                  $padding,
                  $stride_h,
                  $stride_w),
              $output_type),
          (TFL_Conv2DOp
              $in,
              $weight,
              $bias,
              $dilation_h_factor,
              $dilation_w_factor,
              $fused_activation_function,
              $padding,
              $stride_h,
              $stride_w)>;

// Quantize the DepthwiseConv2DOp if the input and weight are dequantized. The
// scale of the bias input is determined by the scales of input and weight
// operands.
def : Pat<(TFL_QuantizeOp
              (TFL_DepthwiseConv2DOp
                  (TFL_DequantizeOp $in),
                  (TFL_DequantizeOp $weight),
                  (TFL_DequantizeOp $bias),
                  $dilation_h_factor,
                  $dilation_w_factor,
                  $fused_activation_function,
                  $padding,
                  $stride_h,
                  $stride_w,
                  $multiplier),
              $output_type),
          (TFL_DepthwiseConv2DOp
              $in,
              $weight,
              $bias,
              $dilation_h_factor,
              $dilation_w_factor,
              $fused_activation_function,
              $padding,
              $stride_h,
              $stride_w,
              $multiplier)>;

// Quantize the ReshapeOp if the input is dequantized and output is quantized.
// The pre-quantize pass can guarantee both quantization parameters are the
// same.
def : Pat<(TFL_QuantizeOp (TFL_ReshapeOp (TFL_DequantizeOp $in)), $output_type),
          (TFL_ReshapeOp $in)>;

// Quantize the ReshapeOp if the input is dequantized and output is quantized.
// The pre-quantize pass has set the output quantization parameters to a
// pre-defined value.
def : Pat<(TFL_QuantizeOp (TFL_SoftmaxOp (TFL_DequantizeOp $in), $beta),
              $output_type),
          (TFL_SoftmaxOp $in, $beta)>;

// Quantize the AveragePool2DOp if the input is dequantized and output is
// quantized. The pre-quantize pass can guarantee both quantization parameters
// are the same.
def : Pat<(TFL_QuantizeOp (TFL_AveragePool2DOp (TFL_DequantizeOp $in),
              $filter_height, $filter_width, $fused_activation_function,
              $padding, $stride_h, $stride_w), $output_type),
          (TFL_AveragePool2DOp $in,
              $filter_height, $filter_width, $fused_activation_function,
              $padding, $stride_h, $stride_w)>;

// Quantize the MaxPool2DOp if the input is dequantized and output is
// quantized. The pre-quantize pass can guarantee both quantization parameters
// are the same.
def : Pat<(TFL_QuantizeOp (TFL_MaxPool2DOp (TFL_DequantizeOp $in),
              $padding, $stride_w, $tride_h, $stride_width, $stride_height,
              $fused_activation_function), $output_type),
          (TFL_MaxPool2DOp $in,
              $padding, $stride_w, $tride_h, $stride_width, $stride_height,
              $fused_activation_function)>;
