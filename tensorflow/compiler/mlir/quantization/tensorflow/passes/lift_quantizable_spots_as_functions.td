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
  (TF_Conv2DOp:$res $input, $filter, $strides, $use_cudnn, $padding,
     $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
  (LiftAsFunctionCall<"fused_conv2d_fn">
     (ArgumentList $input, $filter),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

def LiftDepthwiseConv : Pat<
  (TF_DepthwiseConv2dNativeOp:$res $input, $filter, $strides, $padding,
     $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
  (LiftAsFunctionCall<"fused_depthwise_conv2d_fn">
     (ArgumentList $input, $filter),
     (ResultList $res),
     (AttributeList $strides, $padding, $explicit_padding, $dilations)),
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

def LiftMatMul : Pat<
  (TF_MatMulOp:$res $a, $b, $transpose_a, $transpose_b),
  (LiftAsFunctionCall<"fused_matmul_fn">
     (ArgumentList $a, $b),
     (ResultList $res),
     (AttributeList $transpose_a, $transpose_b)),
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops with bias as functions
//===----------------------------------------------------------------------===//

def LiftDepthwiseConv2dNativeWithBias : Pat<
    (TF_BiasAddOp:$res
      (TF_DepthwiseConv2dNativeOp $input, $filter, $strides, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
      $bias, IsDataFormatNHWC:$bias_data_format),
  (LiftAsFunctionCall<"fused_depthwise_conv2d_with_bias_fn">
     (ArgumentList $input, $filter, $bias),
     (ResultList $res),
     (AttributeList $strides, $padding, $explicit_padding, $dilations)),
  [(IsNotInLiftedFunc $res)], (addBenefit 5)>;

def LiftConv2dWithBias : Pat<
  (TF_BiasAddOp:$res
      (TF_Conv2DOp $input, $filter, $strides, $use_cudnn, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
       $bias, IsDataFormatNHWC:$bias_data_format),
  (LiftAsFunctionCall<"fused_conv2d_with_bias_fn">
     (ArgumentList $input, $filter, $bias),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
    [(IsNotInLiftedFunc $res)], (addBenefit 5)>;

def LiftMatmulWithBias : Pat<
  (TF_BiasAddOp:$res
      (TF_MatMulOp $a, $b, $transpose_a, $transpose_b),
      $bias, IsDataFormatNHWC:$bias_data_format),
  (LiftAsFunctionCall<"fused_matmul_with_bias_fn">
     (ArgumentList $a, $b, $bias),
     (ResultList $res),
     (AttributeList $transpose_a, $transpose_b)),
  [(IsNotInLiftedFunc $res)], (addBenefit 5)>;

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops with bias and activation as functions
//===----------------------------------------------------------------------===//

multiclass LiftCompositeOpsWithActivation<Op ActivationOp, string ActivationName> {
  def LiftConvWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_Conv2DOp $input, $filter, $strides, $use_cudnn, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations)),
    (LiftAsFunctionCall<"fused_conv2d_with_"# ActivationName #"_fn">
       (ArgumentList $input, $filter),
       (ResultList $res),
       (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftConv2dWithBiasAnd#LastFusedOp : Pat<
    (ActivationOp:$res
     (TF_BiasAddOp
        (TF_Conv2DOp $input, $filter, $strides, $use_cudnn, $padding,
           $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
         $bias, IsDataFormatNHWC:$bias_data_format)),
    (LiftAsFunctionCall<"fused_conv2d_with_bias_and_"# ActivationName #"_fn">
       (ArgumentList $input, $filter, $bias),
       (ResultList $res),
       (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftDepthwiseConv2dNativeWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_DepthwiseConv2dNativeOp $input, $filter, $strides, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations)),
    (LiftAsFunctionCall<"fused_depthwise_conv2d_with_"# ActivationName #"_fn">
       (ArgumentList $input, $filter),
       (ResultList $res),
       (AttributeList $strides, $padding, $explicit_padding, $dilations)),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftDepthwiseConv2dNativeWithBiasAnd#LastFusedOp : Pat<
      (ActivationOp:$res
        (TF_BiasAddOp
          (TF_DepthwiseConv2dNativeOp $input, $filter, $strides, $padding,
             $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
          $bias, IsDataFormatNHWC:$bias_data_format)),
      (LiftAsFunctionCall<"fused_depthwise_conv2d_with_bias_and_"# ActivationName #"_fn">
         (ArgumentList $input, $filter, $bias),
         (ResultList $res),
         (AttributeList $strides, $padding, $explicit_padding, $dilations)),
      [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftMatmulWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_MatMulOp $a, $b, $transpose_a, $transpose_b)),
    (LiftAsFunctionCall<"fused_matmul_with_"# ActivationName #"_fn">
       (ArgumentList $a, $b),
       (ResultList $res),
       (AttributeList $transpose_a, $transpose_b)),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

  def LiftMatmulWithBiasAnd#LastFusedOp : Pat<
    (ActivationOp:$res
     (TF_BiasAddOp
        (TF_MatMulOp $a, $b, $transpose_a, $transpose_b),
      $bias, IsDataFormatNHWC:$bias_data_format)),
    (LiftAsFunctionCall<"fused_matmul_with_bias_and_"# ActivationName #"_fn">
       (ArgumentList $a, $b, $bias),
       (ResultList $res),
       (AttributeList $transpose_a, $transpose_b)),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;

}
defm : LiftCompositeOpsWithActivation<TF_ReluOp, "relu">;
defm : LiftCompositeOpsWithActivation<TF_Relu6Op, "relu6">;
