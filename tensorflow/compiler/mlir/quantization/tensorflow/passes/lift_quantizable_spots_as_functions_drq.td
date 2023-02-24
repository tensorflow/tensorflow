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
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.td"
include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.td"

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops as functions
//===----------------------------------------------------------------------===//

def LiftConv : Pat<
  (TF_Conv2DOp:$res $input, $filter, $strides, $use_cudnn_on_gpu, $padding,
    $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations),
  (LiftAsFunctionCall<"composite_conv2d_fn">
    (ArgumentList $input, $filter),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"strides"> $strides),
      (NamedAttr<"use_cudnn_on_gpu"> $use_cudnn_on_gpu),
      (NamedAttr<"padding"> $padding),
      (NamedAttr<"explicit_paddings"> $explicit_paddings),
      (NamedAttr<"dilations"> $dilations))),
  [(IsNotInLiftedFunc $res), (IsConstTensor $filter)], (addBenefit 1)>;

def LiftDepthwiseConv : Pat<
  (TF_DepthwiseConv2dNativeOp:$res $input, $filter, $strides, $padding,
    $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations),
  (LiftAsFunctionCall<"composite_depthwise_conv2d_fn">
    (ArgumentList $input, $filter),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"strides"> $strides),
      (NamedAttr<"padding"> $padding),
      (NamedAttr<"explicit_paddings"> $explicit_paddings),
      (NamedAttr<"dilations"> $dilations))),
  [(IsNotInLiftedFunc $res), (IsConstTensor $filter)], (addBenefit 1)>;

def LiftMatMul : Pat<
  (TF_MatMulOp:$res $a, $b, $transpose_a, $transpose_b),
  (LiftAsFunctionCall<"composite_matmul_fn">
    (ArgumentList $a, $b),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"transpose_a"> $transpose_a),
      (NamedAttr<"transpose_b"> $transpose_b))),
  [(IsNotInLiftedFunc $res), (IsConstTensor $b)], (addBenefit 1)>;

def LiftGather : Pat<
  (TF_GatherV2Op:$res $params, $indices, $axis, $batch_dims),
  (LiftAsFunctionCall<"composite_gather_fn">
    (ArgumentList $params, $indices, $axis),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"batch_dims"> $batch_dims))),
  [(IsNotInLiftedFunc $res), (IsConstTensor $params)], (addBenefit 1)>;

def LiftConv3D : Pat<
  (TF_Conv3DOp:$res $input, $filter, $strides, $padding,
    IsDataFormatNDHWC:$data_format, $dilations),
  (LiftAsFunctionCall<"composite_conv3d_fn">
    (ArgumentList $input, $filter),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"strides"> $strides),
      (NamedAttr<"padding"> $padding),
      (NamedAttr<"dilations"> $dilations))),
  [(IsNotInLiftedFunc $res), (IsConstTensor $filter)], (addBenefit 1)>;

def LiftBatchMatMul : Pat<
  (TF_BatchMatMulV2Op:$res $x, $y, $adj_x, $adj_y),
  (LiftAsFunctionCall<"composite_batch_matmul_fn">
    (ArgumentList $x, $y),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"adj_x"> $adj_x),
      (NamedAttr<"adj_y"> $adj_y))),
  [(IsNotInLiftedFunc $res), (IsConstTensor $y)], (addBenefit 1)>;
