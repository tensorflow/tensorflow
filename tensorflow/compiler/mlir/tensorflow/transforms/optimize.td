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
include "mlir/IR/OpBase.td"
include "mlir/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "NHWC">;
def BroadcastableElements :
    Constraint<CPred<"TFL::IsBroadcastableElementsAttrs($0, $1)">>;
def F32ElementsAttr : ElementsAttrBase<
    CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// If we see a Conv2D op followed by Mul, then multiply the filter
// with the value in Mul.
def FuseMulAndConv2D : Pat<(TF_MulOp (TF_Conv2DOp $input,
                          (ConstantOp F32ElementsAttr:$filter),
                          $strides,
                          $use_cudnn,
                          $padding, $explicit_padding,
                          IsDataFormatNHWC:$data_format,
                          $dilations),
           (ConstantOp F32ElementsAttr:$value)),
// TODO(karimnosseir): Add check for output is of rank 4.
          (TF_Conv2DOp $input,
                       (TF_MulOp (ConstantOp $filter),
                                 (ConstantOp $value)),
                       $strides,
                       $use_cudnn,
                       $padding, $explicit_padding, $data_format,
                       $dilations),
          [(BroadcastableElements $filter, $value)]>;
