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

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.td"

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

class ExtractVariadic<int N> : NativeCodeCall<"*($0.begin() + "# N #")">;

def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">;

class IsFusedOpEndsWith<string OpName> : AttrConstraint<
    CPred<"!$_self.cast<ArrayAttr>().empty() && "
          "$_self.cast<ArrayAttr>()[$_self.cast<ArrayAttr>().size() - 1]."
          "cast<::mlir::StringAttr>().str() == \"" # OpName # "\"">,
    "Matching fused '" # OpName # "' op at the end">;

//===----------------------------------------------------------------------===//
// Pattern rules for lifting as functions
//===----------------------------------------------------------------------===//

def LiftFusedConv2DRelu6 : Pat<
  (TF__FusedConv2DOp:$res
    $input, $filter, $bias, $strides, $padding,
    $explicit_padding, IsDataFormatNHWC:$data_format, $dilations, $use_cudnn,
    IsFusedOpEndsWith<"Relu6">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_conv2d_relu6_fn">
     (ArgumentList $input, $filter, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInLiftedFunc $res)]>;

def LiftFusedConv2DRelu : Pat<
  (TF__FusedConv2DOp:$res
    $input, $filter, $bias, $strides, $padding,
    $explicit_padding, IsDataFormatNHWC:$data_format, $dilations, $use_cudnn,
    IsFusedOpEndsWith<"Relu">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_conv2d_relu_fn">
     (ArgumentList $input, $filter, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInLiftedFunc $res)]>;

def LiftFusedConv2D : Pat<
  (TF__FusedConv2DOp:$res
    $input, $filter, $bias, $strides, $padding,
    $explicit_padding, IsDataFormatNHWC:$data_format, $dilations, $use_cudnn,
    IsFusedOpEndsWith<"BiasAdd">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_conv2d_fn">
     (ArgumentList $input, $filter, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInLiftedFunc $res)]>;

def LiftFusedMatMulRelu6 : Pat<
  (TF__FusedMatMulOp:$res
    $input, $weight, $bias, $transpose_a, $transpose_b,
    IsFusedOpEndsWith<"Relu6">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_matmul_relu6_fn">
     (ArgumentList $input, $weight, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $transpose_a, $transpose_b)),
  [(IsNotInLiftedFunc $res)]>;

def LiftFusedMatMulRelu : Pat<
  (TF__FusedMatMulOp:$res
    $input, $weight, $bias, $transpose_a, $transpose_b,
    IsFusedOpEndsWith<"Relu">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_matmul_relu_fn">
     (ArgumentList $input, $weight, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $transpose_a, $transpose_b)),
  [(IsNotInLiftedFunc $res)]>;

def LiftFusedMatMul : Pat<
  (TF__FusedMatMulOp:$res
    $input, $weight, $bias, $transpose_a, $transpose_b,
    IsFusedOpEndsWith<"BiasAdd">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_matmul_fn">
     (ArgumentList $input, $weight, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $transpose_a, $transpose_b)),
  [(IsNotInLiftedFunc $res)]>;
