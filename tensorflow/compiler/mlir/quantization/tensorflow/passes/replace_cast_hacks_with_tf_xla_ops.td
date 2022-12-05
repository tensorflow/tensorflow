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
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.td"

def CreateXLAConvOpFromTFConv2DOp : NativeCodeCall<
  "CreateXlaConvOpFromTfConv2dOp($_builder, $_loc, $0...)">;

def CreateXLAConvOpFromTFDepthwiseConv2DOp : NativeCodeCall<
  "CreateXlaConvOpFromTfDepthwiseConv2dOp($_builder, $_loc, $0...)">;

def CreateXlaDotV2OpFromTfMatMulOp : NativeCodeCall<
  "CreateXlaDotV2OpFromTfMatMulOp($_builder, $_loc, $0...)">;

def CreateXLAConvOpFromTFConv3DOp : NativeCodeCall<
  "CreateXlaConvOpFromTfConv3dOp($_builder, $_loc, $0...)">;

def CreateXlaDotV2OpFromTfBatchMatMulOp : NativeCodeCall<
  "CreateXlaDotV2OpFromTfBatchMatMulOp($_builder, $_loc, $0...)">;

// Converts inlined Conv2D pattern to TF XlaConvV2 op. This pattern doesn't
// support non-constant weights.
def ConvertTFConv2DToXLAConvOp : Pat<
  (TF_Conv2DOp:$conv
    (TF_SubOp (TF_CastOp $input, $truncate), $input_zp),
    (TF_CastOp (TF_IdentityOp $filter), $truncate1),
    $strides, $use_cudnn, $padding, $explicit_padding,
    IsDataFormatNHWC:$data_format, $dilations),
  (CreateXLAConvOpFromTFConv2DOp
    $input, $filter, $input_zp, $conv, $strides,
    $dilations, $padding, $explicit_padding),
  [(IsInt8ElementType $input),
   (IsInt8ElementType $filter),
   (IsConstTensor $filter),
   (IsInt32ElementType $conv),
   (HasStaticShapeConstraint $filter),
   (HasStaticShapeAtDimsConstraint<"3"> $input)],
  (addBenefit 10)>;

// Same as ConvertTFConv2DToXLAConvOp but handles the case where input zero
// point is 0 and the Sub op has been folded.
def ConvertTFConv2DWithNoZeroPointToXLAConvOp : Pat<
  (TF_Conv2DOp:$conv
    (TF_CastOp $input, $truncate),
    (TF_CastOp (TF_IdentityOp $filter), $truncate1),
    $strides, $use_cudnn, $padding, $explicit_padding,
    IsDataFormatNHWC:$data_format, $dilations),
  (CreateXLAConvOpFromTFConv2DOp
    $input, $filter, /*input_zp=*/(CreateScalarIntegerConst<"int32_t", "0">),
    $conv, $strides, $dilations, $padding, $explicit_padding),
  [(IsInt8ElementType $input),
   (IsInt8ElementType $filter),
   (IsConstTensor $filter),
   (IsInt32ElementType $conv),
   (HasStaticShapeConstraint $filter),
   (HasStaticShapeAtDimsConstraint<"3"> $input)],
  (addBenefit 10)>;

// Converts inlined DepthwiseConv2D pattern to TF XlaConvV2 op. This pattern
// doesn't support non-constant weights.
def ConvertTFDepthwiseConv2DToXLAConvOp : Pat<
  (TF_CastOp:$conv
    (TF_DepthwiseConv2dNativeOp
      (TF_CastOp:$casted_input
        (TF_SubOp (TF_CastOp $input, $truncate1), $input_zp), $truncate2),
      (TF_CastOp
        (TF_CastOp (TF_IdentityOp $filter), $truncate3), $truncate4),
      $strides, $padding, $explicit_padding,
      IsDataFormatNHWC:$data_format, $dilations), $truncate5),
  (CreateXLAConvOpFromTFDepthwiseConv2DOp
    $input, $filter, $input_zp, $conv, $strides,
    $dilations, $padding, $explicit_padding),
  [(IsInt8ElementType $input),
   (IsF32ElementType $casted_input),
   (IsInt8ElementType $filter),
   (IsConstTensor $filter),
   (IsInt32ElementType $conv),
   (HasStaticShapeConstraint $filter),
   (HasStaticShapeAtDimsConstraint<"3"> $input)],
  (addBenefit 10)>;

// Same as ConvertTFDepthwiseConv2DToXLAConvOp but handles the case where input
// zero point is 0 and the Sub op has been folded.
def ConvertTFDepthwiseConv2DWithNoZeroPointToXLAConvOp : Pat<
  (TF_CastOp:$conv
    (TF_DepthwiseConv2dNativeOp
      (TF_CastOp:$casted_input
        (TF_CastOp $input, $truncate1), $truncate2),
      (TF_CastOp
        (TF_CastOp (TF_IdentityOp $filter), $truncate3), $truncate4),
      $strides, $padding, $explicit_padding,
      IsDataFormatNHWC:$data_format, $dilations), $truncate5),
  (CreateXLAConvOpFromTFDepthwiseConv2DOp
    $input, $filter, /*input_zp=*/(CreateScalarIntegerConst<"int32_t", "0">),
    $conv, $strides, $dilations, $padding, $explicit_padding),
  [(IsInt8ElementType $input),
   (IsF32ElementType $casted_input),
   (IsInt8ElementType $filter),
   (IsConstTensor $filter),
   (IsInt32ElementType $conv),
   (HasStaticShapeConstraint $filter),
   (HasStaticShapeAtDimsConstraint<"3"> $input)],
  (addBenefit 10)>;


// Converts inlined MatMul pattern to TF XlaDotV2 op. This pattern doesn't
// support non-constant weights.
def ConvertTFMatMulToXLADotV2Op : Pat<
  (TF_MatMulOp:$matmul
    (TF_SubOp (TF_CastOp $input, $truncate), $input_zp),
    (TF_CastOp (TF_IdentityOp $weight), $truncate1),
    $transpose_a, $transpose_b),
  (CreateXlaDotV2OpFromTfMatMulOp
    $input, $weight, $input_zp, $matmul, $transpose_a, $transpose_b),
  [(IsInt8ElementType $input),
   (IsInt8ElementType $weight),
   (IsConstTensor $weight),
   (IsInt32ElementType $matmul),
   (HasStaticShapeConstraint $weight)],
  (addBenefit 10)>;

// Same as ConvertTFMatMulToXLADotV2Op but handles the case where input
// zero point is 0 and the Sub op has been folded.
def ConvertTFMatMulWithNoZeroPointToXLADotV2Op : Pat<
  (TF_MatMulOp:$matmul
    (TF_CastOp $input, $truncate),
    (TF_CastOp (TF_IdentityOp $weight), $truncate1),
    $transpose_a, $transpose_b),
  (CreateXlaDotV2OpFromTfMatMulOp
    $input, $weight, /*input_zp=*/(CreateScalarIntegerConst<"int32_t", "0">),
    $matmul, $transpose_a, $transpose_b),
  [(IsInt8ElementType $input),
   (IsInt8ElementType $weight),
   (IsConstTensor $weight),
   (IsInt32ElementType $matmul),
   (HasStaticShapeConstraint $weight)],
  (addBenefit 10)>;

// Converts inlined Conv3D pattern to TF XlaConvV2 op. This pattern
// doesn't support non-constant weights.
def ConvertTFConv3DToXLAConvOp : Pat<
  (TF_CastOp:$conv
    (TF_Conv3DOp
      (TF_CastOp:$casted_input
        (TF_SubOp (TF_CastOp $input, $truncate1), $input_zp), $truncate2),
      (TF_CastOp
        (TF_CastOp (TF_IdentityOp $filter), $truncate3), $truncate4),
      $strides, $padding, IsDataFormatNDHWC:$data_format, $dilations),
    $truncate5),
  (CreateXLAConvOpFromTFConv3DOp
    $input, $filter, $input_zp, $conv, $strides, $dilations, $padding),
  [(IsInt8ElementType $input),
   (IsF32ElementType $casted_input),
   (IsInt8ElementType $filter),
   (IsConstTensor $filter),
   (IsInt32ElementType $conv),
   (HasStaticShapeConstraint $filter),
   (HasStaticShapeAtDimsConstraint<"4"> $input)],
  (addBenefit 10)>;

// Same as ConvertTFConv3DToXLAConvOp but handles the case where input
// zero point is 0 and the Sub op has been folded.
def ConvertTFConv3DWithNoZeroPointToXLAConvOp : Pat<
  (TF_CastOp:$conv
    (TF_Conv3DOp
      (TF_CastOp:$casted_input
        (TF_CastOp $input, $truncate1), $truncate2),
      (TF_CastOp
        (TF_CastOp (TF_IdentityOp $filter), $truncate3), $truncate4),
      $strides, $padding, IsDataFormatNDHWC:$data_format, $dilations),
    $truncate5),
  (CreateXLAConvOpFromTFConv3DOp
    $input, $filter, /*input_zp=*/(CreateScalarIntegerConst<"int32_t", "0">),
    $conv, $strides, $dilations, $padding),
  [(IsInt8ElementType $input),
   (IsF32ElementType $casted_input),
   (IsInt8ElementType $filter),
   (IsConstTensor $filter),
   (IsInt32ElementType $conv),
   (HasStaticShapeConstraint $filter),
   (HasStaticShapeAtDimsConstraint<"4"> $input)],
  (addBenefit 10)>;

// Converts inlined BatchMatMul pattern to TF XlaDotV2 op. This pattern doesn't
// support non-constant weights.
def ConvertTFBatchMatMulToXLADotV2Op : Pat<
  (TF_BatchMatMulV2Op:$batch_matmul
    (TF_SubOp (TF_CastOp $input, $truncate), $input_zp),
    (TF_CastOp (TF_IdentityOp $weight), $truncate1),
    $adj_x, $adj_y),
  (CreateXlaDotV2OpFromTfBatchMatMulOp
    $input, $weight, $input_zp, $batch_matmul, $adj_x, $adj_y),
  [(IsInt8ElementType $input),
   (HasRank $input),
   (IsInt8ElementType $weight),
   (IsConstTensor $weight),
   (IsInt32ElementType $batch_matmul),
   (HasStaticShapeConstraint $weight)],
  (addBenefit 10)>;

// Same as ConvertTFBatchMatMulToXLADotV2Op but handles the case where input
// zero point is 0 and the Sub op has been folded.
def ConvertTFBatchMatMulWithNoZeroPointToXLADotV2Op : Pat<
  (TF_BatchMatMulV2Op:$batch_matmul
    (TF_CastOp $input, $truncate),
    (TF_CastOp (TF_IdentityOp $weight), $truncate1),
    $adj_x, $adj_y),
  (CreateXlaDotV2OpFromTfBatchMatMulOp
    $input, $weight, /*input_zp=*/(CreateScalarIntegerConst<"int32_t", "0">),
    $batch_matmul, $adj_x, $adj_y),
  [(IsInt8ElementType $input),
   (HasRank $input),
   (IsInt8ElementType $weight),
   (IsConstTensor $weight),
   (IsInt32ElementType $batch_matmul),
   (HasStaticShapeConstraint $weight)],
  (addBenefit 10)>;
