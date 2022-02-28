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
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//
class LiftAsFunctionCall<string func_name> :
  NativeCodeCall<"LiftAsFunctionCall($_builder, $_loc, "
                    "\""# func_name #"\", $0...)">;

class ExtractVariadic<int N> : NativeCodeCall<"*($0.begin() + "# N #")">;

def ArgumentList : NativeCodeCall<"llvm::SmallVector<Value>{$0...}">;

def ResultList : NativeCodeCall<"llvm::SmallVector<Value>{$0...}">;

def AttributeList : NativeCodeCall<"llvm::SmallVector<Attribute>{$0...}">;

def IsNotInFusedFunc :
      Constraint<CPred<"!IsInFusedFunc($0.getDefiningOp())">>;

def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">;

class IsFusedOpEndsWith<string OpName> : AttrConstraint<
    CPred<"!$_self.cast<ArrayAttr>().empty() && "
          "$_self.cast<ArrayAttr>()[$_self.cast<ArrayAttr>().size() - 1]."
          "cast<::mlir::StringAttr>().str() == \"" # OpName # "\"">,
    "Matching fused '" # OpName # "' op at the end">;

//===----------------------------------------------------------------------===//
// Pattern rules for lifting as functions
//===----------------------------------------------------------------------===//

def LiftAdd : Pat<
  (TF_AddV2Op:$res $x, $y),
  (LiftAsFunctionCall<"fused_add_fn">
     (ArgumentList $x, $y),
     (ResultList $res)),
  [(IsNotInFusedFunc $res)]>;

def LiftFusedConv2DRelu6 : Pat<
  (TF__FusedConv2DOp:$res
    $input, $filter, $bias, $strides, $padding,
    $explicit_padding, IsDataFormatNHWC:$data_format, $dilations, $use_cudnn,
    IsFusedOpEndsWith<"Relu6">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_conv2d_relu6_fn">
     (ArgumentList $input, $filter, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInFusedFunc $res)]>;

def LiftFusedConv2DRelu : Pat<
  (TF__FusedConv2DOp:$res
    $input, $filter, $bias, $strides, $padding,
    $explicit_padding, IsDataFormatNHWC:$data_format, $dilations, $use_cudnn,
    IsFusedOpEndsWith<"Relu">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_conv2d_relu_fn">
     (ArgumentList $input, $filter, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInFusedFunc $res)]>;

def LiftFusedConv2D : Pat<
  (TF__FusedConv2DOp:$res
    $input, $filter, $bias, $strides, $padding,
    $explicit_padding, IsDataFormatNHWC:$data_format, $dilations, $use_cudnn,
    IsFusedOpEndsWith<"BiasAdd">:$fused_ops, $epsilon, $leakyrelu_alpha),
  (LiftAsFunctionCall<"fused_conv2d_fn">
     (ArgumentList $input, $filter, (ExtractVariadic<0> $bias)),
     (ResultList $res),
     (AttributeList $strides, $use_cudnn, $padding, $explicit_padding, $dilations)),
  [(IsNotInFusedFunc $res)]>;
