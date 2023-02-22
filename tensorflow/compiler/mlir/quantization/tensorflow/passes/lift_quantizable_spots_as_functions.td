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
// Helper functions.
//===----------------------------------------------------------------------===//

class IsFusedOpEndsWith<string OpName> : AttrConstraint<
  CPred<"!$_self.cast<ArrayAttr>().empty() && "
        "$_self.cast<ArrayAttr>()[$_self.cast<ArrayAttr>().size() - 1]."
        "cast<::mlir::StringAttr>().str() == \"" # OpName # "\"">,
  "Matching fused '" # OpName # "' op at the end">;

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
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

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
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

def LiftMatMul : Pat<
  (TF_MatMulOp:$res $a, $b, $transpose_a, $transpose_b),
  (LiftAsFunctionCall<"composite_matmul_fn">
    (ArgumentList $a, $b),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"transpose_a"> $transpose_a),
      (NamedAttr<"transpose_b"> $transpose_b))),
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

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
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

def LiftBatchMatMul : Pat<
  (TF_BatchMatMulV2Op:$res $x, $y, $adj_x, $adj_y),
  (LiftAsFunctionCall<"composite_batch_matmul_fn">
    (ArgumentList $x, $y),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"adj_x"> $adj_x),
      (NamedAttr<"adj_y"> $adj_y))),
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops with bias as functions
//===----------------------------------------------------------------------===//

def LiftDepthwiseConv2dNativeWithBias : Pat<
  (TF_BiasAddOp:$res
    (TF_DepthwiseConv2dNativeOp $input, $filter, $strides, $padding,
      $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations),
    $bias, IsDataFormatNHWC:$bias_data_format),
  (LiftAsFunctionCall<"composite_depthwise_conv2d_with_bias_fn">
    (ArgumentList $input, $filter, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"strides"> $strides),
      (NamedAttr<"padding"> $padding),
      (NamedAttr<"explicit_paddings"> $explicit_paddings),
      (NamedAttr<"dilations"> $dilations))),
  [(IsNotInLiftedFunc $res)], (addBenefit 5)>;

def LiftConv2dWithBias : Pat<
  (TF_BiasAddOp:$res
    (TF_Conv2DOp $input, $filter, $strides, $use_cudnn_on_gpu, $padding,
      $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations),
    $bias, IsDataFormatNHWC:$bias_data_format),
  (LiftAsFunctionCall<"composite_conv2d_with_bias_fn">
    (ArgumentList $input, $filter, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"strides"> $strides),
      (NamedAttr<"use_cudnn_on_gpu"> $use_cudnn_on_gpu),
      (NamedAttr<"padding"> $padding),
      (NamedAttr<"explicit_paddings"> $explicit_paddings),
      (NamedAttr<"dilations"> $dilations))),
  [(IsNotInLiftedFunc $res)], (addBenefit 5)>;

def LiftMatmulWithBias : Pat<
  (TF_BiasAddOp:$res
    (TF_MatMulOp $a, $b, $transpose_a, $transpose_b),
    $bias, IsDataFormatNHWC:$bias_data_format),
  (LiftAsFunctionCall<"composite_matmul_with_bias_fn">
    (ArgumentList $a, $b, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"transpose_a"> $transpose_a),
      (NamedAttr<"transpose_b"> $transpose_b))),
  [(IsNotInLiftedFunc $res)], (addBenefit 5)>;

def LiftConv3dWithBias : Pat<
  (TF_BiasAddOp:$res
    (TF_Conv3DOp $input, $filter, $strides, $padding,
      IsDataFormatNDHWC:$data_format, $dilations),
    $bias, $bias_data_format),
  (LiftAsFunctionCall<"composite_conv3d_with_bias_fn">
    (ArgumentList $input, $filter, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"strides"> $strides),
      (NamedAttr<"padding"> $padding),
      (NamedAttr<"dilations"> $dilations))),
  [(IsNotInLiftedFunc $res)], (addBenefit 5)>;

def LiftBatchMatMulWithBias : Pat<
  (TF_BiasAddOp:$res
    (TF_BatchMatMulV2Op $x, $y, $adj_x, $adj_y),
    $bias, IsDataFormatNHWC:$bias_data_format),
  (LiftAsFunctionCall<"composite_batch_matmul_with_bias_fn">
    (ArgumentList $x, $y, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"adj_x"> $adj_x),
      (NamedAttr<"adj_y"> $adj_y))),
  [(IsNotInLiftedFunc $res)], (addBenefit 5)>;

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops with bias and activation as functions
//===----------------------------------------------------------------------===//

multiclass LiftCompositeOpsWithActivation<Op ActivationOp, string ActivationName> {
  def LiftConvWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_Conv2DOp $input, $filter, $strides, $use_cudnn_on_gpu, $padding,
        $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations)),
    (LiftAsFunctionCall<"composite_conv2d_with_"# ActivationName #"_fn">
      (ArgumentList $input, $filter),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"strides"> $strides),
        (NamedAttr<"use_cudnn_on_gpu"> $use_cudnn_on_gpu),
        (NamedAttr<"padding"> $padding),
        (NamedAttr<"explicit_paddings"> $explicit_paddings),
        (NamedAttr<"dilations"> $dilations))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftConv2dWithBiasAnd#LastFusedOp : Pat<
    (ActivationOp:$res
      (TF_BiasAddOp
        (TF_Conv2DOp $input, $filter, $strides, $use_cudnn_on_gpu, $padding,
          $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations),
        $bias, IsDataFormatNHWC:$bias_data_format)),
    (LiftAsFunctionCall<"composite_conv2d_with_bias_and_"# ActivationName #"_fn">
      (ArgumentList $input, $filter, $bias),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"strides"> $strides),
        (NamedAttr<"use_cudnn_on_gpu"> $use_cudnn_on_gpu),
        (NamedAttr<"padding"> $padding),
        (NamedAttr<"explicit_paddings"> $explicit_paddings),
        (NamedAttr<"dilations"> $dilations))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftDepthwiseConv2dNativeWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_DepthwiseConv2dNativeOp $input, $filter, $strides, $padding,
        $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations)),
    (LiftAsFunctionCall<"composite_depthwise_conv2d_with_"# ActivationName #"_fn">
      (ArgumentList $input, $filter),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"strides"> $strides),
        (NamedAttr<"padding"> $padding),
        (NamedAttr<"explicit_paddings"> $explicit_paddings),
        (NamedAttr<"dilations"> $dilations))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftDepthwiseConv2dNativeWithBiasAnd#LastFusedOp : Pat<
    (ActivationOp:$res
      (TF_BiasAddOp
        (TF_DepthwiseConv2dNativeOp $input, $filter, $strides, $padding,
          $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations),
        $bias, IsDataFormatNHWC:$bias_data_format)),
    (LiftAsFunctionCall<"composite_depthwise_conv2d_with_bias_and_"# ActivationName #"_fn">
      (ArgumentList $input, $filter, $bias),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"strides"> $strides),
        (NamedAttr<"padding"> $padding),
        (NamedAttr<"explicit_paddings"> $explicit_paddings),
        (NamedAttr<"dilations"> $dilations))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftMatmulWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_MatMulOp $a, $b, $transpose_a, $transpose_b)),
    (LiftAsFunctionCall<"composite_matmul_with_"# ActivationName #"_fn">
      (ArgumentList $a, $b),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"transpose_a"> $transpose_a),
        (NamedAttr<"transpose_b"> $transpose_b))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftMatmulWithBiasAnd#LastFusedOp : Pat<
    (ActivationOp:$res
      (TF_BiasAddOp
        (TF_MatMulOp $a, $b, $transpose_a, $transpose_b),
      $bias, IsDataFormatNHWC:$bias_data_format)),
    (LiftAsFunctionCall<"composite_matmul_with_bias_and_"# ActivationName #"_fn">
      (ArgumentList $a, $b, $bias),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"transpose_a"> $transpose_a),
        (NamedAttr<"transpose_b"> $transpose_b))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftConv3dWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_Conv3DOp $input, $filter, $strides, $padding,
        IsDataFormatNDHWC:$data_format, $dilations)),
    (LiftAsFunctionCall<"composite_conv3d_with_"# ActivationName #"_fn">
      (ArgumentList $input, $filter),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"strides"> $strides),
        (NamedAttr<"padding"> $padding),
        (NamedAttr<"dilations"> $dilations))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftConv3dWithBiasAnd#LastFusedOp : Pat<
    (ActivationOp:$res
      (TF_BiasAddOp
        (TF_Conv3DOp $input, $filter, $strides, $padding,
          IsDataFormatNDHWC:$data_format, $dilations),
        $bias, $bias_data_format)),
    (LiftAsFunctionCall<"composite_conv3d_with_bias_and_"# ActivationName #"_fn">
      (ArgumentList $input, $filter, $bias),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"strides"> $strides),
        (NamedAttr<"padding"> $padding),
        (NamedAttr<"dilations"> $dilations))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftBatchMatMulWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_BatchMatMulV2Op $x, $y, $adj_x, $adj_y)),
    (LiftAsFunctionCall<"composite_batch_matmul_with_"# ActivationName #"_fn">
      (ArgumentList $x, $y),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"adj_x"> $adj_x),
        (NamedAttr<"adj_y"> $adj_y))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftBatchMatMulWithBiasAnd#LastFusedOp : Pat<
    (ActivationOp:$res
      (TF_BiasAddOp
        (TF_BatchMatMulV2Op $x, $y, $adj_x, $adj_y),
        $bias, IsDataFormatNHWC:$bias_data_format)),
    (LiftAsFunctionCall<"composite_batch_matmul_with_bias_and_"# ActivationName #"_fn">
      (ArgumentList $x, $y, $bias),
      (ResultList $res),
      (NamedAttributeList
        (NamedAttr<"adj_x"> $adj_x),
        (NamedAttr<"adj_y"> $adj_y))),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;
}
defm : LiftCompositeOpsWithActivation<TF_ReluOp, "relu">;
defm : LiftCompositeOpsWithActivation<TF_Relu6Op, "relu6">;

def LiftGather : Pat<
  (TF_GatherV2Op:$res $params, $indices, $axis, $batch_dims),
  (LiftAsFunctionCall<"composite_gather_fn">
    (ArgumentList $params, $indices, $axis),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"batch_dims"> $batch_dims))),
  [(IsNotInLiftedFunc $res), (IsConstTensor $params)], (addBenefit 1)>;
