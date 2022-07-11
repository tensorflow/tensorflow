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
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.td"

def CreateXLAConvOpFromTFConv2DOp : NativeCodeCall<
  "CreateXLAConvOpFromTFConv2DOp($_builder, $_loc, $0...)">;

def ConvertTFConv2DToXLAConvOp : Pat<
  (TF_Conv2DOp:$conv
     (TF_SubOp (TF_CastOp $input, $truncate), $input_zp),
     (TF_ConstOp:$filter $filter_value),
     $strides, $use_cudnn, $padding, $explicit_padding,
    IsDataFormatNHWC:$data_format, $dilations),
  (CreateXLAConvOpFromTFConv2DOp
    $input, $filter, $input_zp, $conv, $strides,
    $dilations, $padding, $explicit_padding),
  [(IsInt8ElementType $input),
   (IsInt32ElementType $filter),
   (IsInt32ElementType $conv),
   (HasStaticShapeConstraint $filter),
   (HasStaticShapeAtDimsConstraint<"1, 2, 3"> $input)],
  (addBenefit 10)>;
