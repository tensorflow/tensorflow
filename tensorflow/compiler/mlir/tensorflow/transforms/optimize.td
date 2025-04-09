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
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">;

// Get the last dimension size as a 1-d single element attr.
def GetLastDimSizeAsI32 : NativeCodeCall<
  "DenseElementsAttr::get(RankedTensorType::get({1}, $_builder.getIntegerType(32)), "
  "static_cast<int32_t>($0.getType().cast<RankedTensorType>().getDimSize(  "
  "  $0.getType().cast<RankedTensorType>().getRank() - 1)))">;

// Check whether the tensor is ranked and whether its last dim is static.
def IsRankedShapeLastDimStatic : Constraint<And<[
  CPred<"$0.getType().isa<RankedTensorType>()">,
  CPred<"!$0.getType().cast<ShapedType>().isDynamicDim( "
  "  $0.getType().cast<RankedTensorType>().getRank() - 1)">]>>;

def IsNotComplexType : Constraint<And<[
  CPred<"$0.getType().isa<RankedTensorType>()">,
  CPred<"!$0.getType().cast<ShapedType>().getElementType().isa<ComplexType>()">
]>>;

// Only fuse multiplier if all dimensions other than the channel dimension
// are equal to 1.
def CanFuseMulAndConv2D :
    Constraint<CPred<"IsBroadcastableElementsAttrs($0, $1) && IsDimensionsDegenerateExceptLastOne($1)">>;

def F32ElementsAttr : ElementsAttrBase<
    CPred<"$_self.cast<ElementsAttr>().getShapedType().getElementType().isF32()">, "float constant tensor">;
def DefinedByConv2D : Constraint<CPred<"llvm::isa_and_nonnull<mlir::TF::Conv2DOp>($0.getDefiningOp())">>;
// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

// If we see a Conv2D op followed by Mul, then multiply the filter
// with the value in Mul.
def FuseMulAndConv2D :
  Pat<(TF_MulOp:$mul (TF_Conv2DOp:$conv $input,
                          (Arith_ConstantOp:$filter F32ElementsAttr:$filter_value),
                          $strides, $use_cudnn, $padding, $explicit_padding,
                          IsDataFormatNHWC:$data_format, $dilations),
                     (Arith_ConstantOp:$multiplier F32ElementsAttr:$mul_value)),
// TODO(karimnosseir): Add check for $conv is of rank 4.
      (TF_Conv2DOp $input,
          (TF_MulOp (Arith_ConstantOp $filter_value, (location $filter)),
                    (Arith_ConstantOp $mul_value, (location $multiplier)),
                    (location $mul)),
          $strides, $use_cudnn, $padding, $explicit_padding, $data_format,
          $dilations, (location $conv)),
      [(CanFuseMulAndConv2D $filter_value, $mul_value), (HasOneUse $conv)]>;

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
        (TF_BiasAddOp:$output $input,
          (Arith_ConstantOp F32ElementsAttr:$bias), IsDataFormatNHWC:$format),
        (Arith_ConstantOp F32ElementsAttr:$value)),
      (TF_AddV2Op
          (TF_MulOp $input, (Arith_ConstantOp $value)),
          (TF_MulOp (Arith_ConstantOp $bias), (Arith_ConstantOp $value))),
      [(DefinedByConv2D $input), (HasOneUse $output)]>;


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
        (TF_AddV2Op:$output $input, (Arith_ConstantOp F32ElementsAttr:$bias)),
        (Arith_ConstantOp F32ElementsAttr:$value)),
      (TF_AddV2Op
          (TF_MulOp $input, (Arith_ConstantOp $value)),
          (TF_MulOp (Arith_ConstantOp $bias), (Arith_ConstantOp $value))),
      [(DefinedByConv2D $input), (HasOneUse $output)]>;

// input -> cast -> FFT  => input -> RFFT
def ConvertCastComplexFFTToRFFT: Pat<
  (TF_FFTOp (TF_CastOp $input, ConstBoolAttrFalse)),
  (TF_RFFTOp $input,
             (Arith_ConstantOp (GetLastDimSizeAsI32 $input))),
  [(IsRankedShapeLastDimStatic $input), (IsNotComplexType $input)]>;
