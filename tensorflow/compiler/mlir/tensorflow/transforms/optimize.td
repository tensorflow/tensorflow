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
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "NHWC">;
def BroadcastableElements :
    Constraint<CPred<"TFL::IsBroadcastableElementsAttrs($0, $1)">>;
def F32ElementsAttr : ElementsAttrBase<
    CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;
def DefinedByConv2D : Constraint<CPred<"llvm::isa_and_nonnull<mlir::TF::Conv2DOp>($0.getDefiningOp())">>;

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

// This rule does the following pattern match and rewrite:
//
//       input     bias                    input  value  bias  value
//          \      /                =>       \    /        \    /
//          BiasAdd    value                   Mul          Mul
//                \    /                           \       /
//                  Mul                              AddV2
// This is to enable the FuseMulAndConv2D pattern.
// Here, root of the result is AddV2 instead of BiasAdd because the value may
// not have rank one and therefore the second operand may not have rank one
// that is required by the BiasAdd. BiasAdd with 'NHWC' data format equivalent
// to AddV2 op.
def PassthroughMulAndBiasAdd :
  Pat<(TF_MulOp
        (TF_BiasAddOp $input,
          (ConstantOp F32ElementsAttr:$bias), IsDataFormatNHWC:$format),
        (ConstantOp F32ElementsAttr:$value)),
      (TF_AddV2Op
          (TF_MulOp $input, (ConstantOp $value)),
          (TF_MulOp (ConstantOp $bias), (ConstantOp $value))),
      [(DefinedByConv2D $input)]>;


// This rule does the following pattern match and rewrite:
//
//       input     bias                    input  value  bias  value
//          \      /                =>       \    /        \    /
//           AddV2    value                   Mul          Mul
//                \    /                           \       /
//                  Mul                             AddV2
// This is to enable the FuseMulAndConv2D pattern.
def PassthroughMulAndAddV2 :
  Pat<(TF_MulOp
        (TF_AddV2Op $input, (ConstantOp F32ElementsAttr:$bias)),
        (ConstantOp F32ElementsAttr:$value)),
      (TF_AddV2Op
          (TF_MulOp $input, (ConstantOp $value)),
          (TF_MulOp (ConstantOp $bias), (ConstantOp $value))),
      [(DefinedByConv2D $input)]>;
