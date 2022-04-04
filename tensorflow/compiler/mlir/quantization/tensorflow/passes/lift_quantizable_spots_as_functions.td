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

// TODO(b/227709630): Rename the term `fused` to `composite`.
multiclass LiftCompositeOpsWithBiasAndActivation<string LastFusedOp, string FusedOps> {
def LiftFusedConv2DWith#LastFusedOp : Pat<
  (TF__FusedConv2DOp:$res
    $input, $filter, $bias, $strides, $padding,
    $explicit_padding, IsDataFormatNHWC:$data_format, $dilations, $use_cudnn,
    IsFusedOpEndsWith<LastFusedOp>:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_conv2d_with_"# FusedOps #"_fn">
     (ArgumentList $input, $filter, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInLiftedFunc $res)]>;

def LiftFusedMatMulWith#LastFusedOp : Pat<
  (TF__FusedMatMulOp:$res
    $input, $weight, $bias, $transpose_a, $transpose_b,
    IsFusedOpEndsWith<LastFusedOp>:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_matmul_with_"# FusedOps #"_fn">
     (ArgumentList $input, $weight, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $transpose_a, $transpose_b)),
  [(IsNotInLiftedFunc $res)]>;
}
defm : LiftCompositeOpsWithBiasAndActivation<"BiasAdd", "bias">;
defm : LiftCompositeOpsWithBiasAndActivation<"Relu", "bias_and_relu">;
defm : LiftCompositeOpsWithBiasAndActivation<"Relu6", "bias_and_relu6">;

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

  def LiftMatmulWith#ActivationOp : Pat<
    (ActivationOp:$res
      (TF_MatMulOp $a, $b, $transpose_a, $transpose_b)),
    (LiftAsFunctionCall<"fused_matmul_with_"# ActivationName #"_fn">
       (ArgumentList $a, $b),
       (ResultList $res),
       (AttributeList $transpose_a, $transpose_b)),
    [(IsNotInLiftedFunc $res)], (addBenefit 10)>;
}
defm : LiftCompositeOpsWithActivation<TF_ReluOp, "relu">;
defm : LiftCompositeOpsWithActivation<TF_Relu6Op, "relu6">;

def LiftConv : Pat<
  (TF_Conv2DOp:$res $input, $filter, $strides, $use_cudnn, $padding,
     $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
  (LiftAsFunctionCall<"fused_conv2d_fn">
     (ArgumentList $input, $filter),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;

def LiftMatMul : Pat<
  (TF_MatMulOp:$res $a, $b, $transpose_a, $transpose_b),
  (LiftAsFunctionCall<"fused_matmul_fn">
     (ArgumentList $a, $b),
     (ResultList $res),
     (AttributeList $transpose_a, $transpose_b)),
  [(IsNotInLiftedFunc $res)], (addBenefit 1)>;
