/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

//===----------------------------------------------------------------------===//
// Pattern rules for converting bfloat16 operations to fp32 conversions.
//===----------------------------------------------------------------------===//

// Remove unneeded redundant cast ops like (f32 -> bf16 -> f32).
def RemoveUnneededCastOps : Pat<
  (TF_CastOp:$output
    (TF_CastOp
      $input, $truncate_0), $truncate_1),
  (replaceWithValue $input),
  [(AreTheSameElementType $input, $output)]>;

// Cast BF16 Conv2D ops to FP32 Conv2D ops. Inputs and
// filters will be casted to fp32 as well, and unused
// BF16 constant values will be removed by the compiler.
def CastBFloat16ConvToFloat32 : Pat<
  (TF_Conv2DOp:$res
    $input, $filter, $strides, $use_cudnn_on_gpu, $padding,
    $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations),
  (TF_CastOp
    (TF_Conv2DOp
      (TF_CastOp
        $input, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $input))),
      (TF_CastOp
        $filter, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $filter))),
      $strides, $use_cudnn_on_gpu, $padding,
      $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations,
      (returnType (CloneTypeWithF32ElementType $res))),
    /*truncate=*/ConstBoolAttrFalse),
  [(IsBF16ElementType $input),
   (IsBF16ElementType $filter)],
  (addBenefit 1)>;

// Casts BF16 BiasAdd ops to F32 to optimize quantizable ops followed by
// BiasAdd ops. This cast will cover Conv + BiasAdd, MatMul + BiasAdd,
// etc.
def CastBFloat16BiasAddToFloat32 : Pat<
  (TF_BiasAddOp:$res
    $input, $bias, IsDataFormatNHWC:$bias_data_format),
  (TF_CastOp
    (TF_BiasAddOp
      (TF_CastOp
        $input, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $input))),
      (TF_CastOp
        $bias, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $bias))),
      IsDataFormatNHWC:$bias_data_format,
      (returnType (CloneTypeWithF32ElementType $res))),
    /*truncate=*/ConstBoolAttrFalse),
  [(IsBF16ElementType $input),
   (IsBF16ElementType $bias)],
  (addBenefit 1)>;

def CastBFloat16AvgPoolToFloat32 : Pat<
  (TF_AvgPoolOp:$res
    $input, $ksize, $strides, $padding,
    IsDataFormatNHWC:$data_format),
  (TF_CastOp
    (TF_AvgPoolOp
      (TF_CastOp
        $input, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $input))),
      $ksize, $strides, $padding,
      IsDataFormatNHWC:$data_format,
      (returnType (CloneTypeWithF32ElementType $res))),
    /*truncate=*/ConstBoolAttrFalse),
  [(IsBF16ElementType $input)],
  (addBenefit 1)>;

def CastBFloat16MatMulToFloat32 : Pat<
  (TF_MatMulOp:$res
    $input, $filter, $transpose_a, $transpose_b),
  (TF_CastOp
    (TF_MatMulOp
      (TF_CastOp
        $input, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $input))),
      (TF_CastOp
        $filter, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $filter))),
      $transpose_a, $transpose_b,
      (returnType (CloneTypeWithF32ElementType $res))),
    /*truncate=*/ConstBoolAttrFalse),
  [(IsBF16ElementType $input),
   (IsBF16ElementType $filter)],
  (addBenefit 1)>;

def CastBFloat16BatchMatMulV2ToFloat32 : Pat<
  (TF_BatchMatMulV2Op:$res
    $input, $filter, $adj_x, $adj_y),
  (TF_CastOp
    (TF_BatchMatMulV2Op
      (TF_CastOp
        $input, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $input))),
      (TF_CastOp
        $filter, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $filter))),
      $adj_x, $adj_y,
      (returnType (CloneTypeWithF32ElementType $res))),
    /*truncate=*/ConstBoolAttrFalse),
  [(IsBF16ElementType $input),
   (IsBF16ElementType $filter)],
  (addBenefit 1)>;

def CastBFloat16DepthwiseConvToFloat32 : Pat<
  (TF_DepthwiseConv2dNativeOp:$res
    $input, $filter, $strides, $padding,
    $explicit_paddings, IsDataFormatNHWC:$data_format, $dilations),
  (TF_CastOp
    (TF_DepthwiseConv2dNativeOp
      (TF_CastOp
        $input, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $input))),
      (TF_CastOp
        $filter, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $filter))),
      $strides, $padding, $explicit_paddings,
      IsDataFormatNHWC:$data_format, $dilations,
      (returnType (CloneTypeWithF32ElementType $res))),
    /*truncate=*/ConstBoolAttrFalse),
  [(IsBF16ElementType $input),
   (IsBF16ElementType $filter)],
  (addBenefit 1)>;

def CastBFloat16GatherToFloat32 : Pat<
  (TF_GatherV2Op:$res
    $params, $indices, $axis, $batch_dims),
  (TF_CastOp
    (TF_GatherV2Op
      (TF_CastOp
        $params, /*truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $params))),
      $indices, $axis, $batch_dims,
      (returnType (CloneTypeWithF32ElementType $res))),
    /*truncate=*/ConstBoolAttrFalse),
  [(IsBF16ElementType $params),
    (IsConstTensor $params)],
  (addBenefit 1)>;

// Converts an AddV2 op accepting two bfloat16 operands into the one taking two
// float32 operands.
def CastBFloat16AddV2ToFloat32 : Pat<
  (TF_AddV2Op:$res $x, $y),
  (TF_CastOp
    (TF_AddV2Op
      (TF_CastOp $x, /*Truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $x))),
      (TF_CastOp $y, /*Truncate=*/ConstBoolAttrFalse,
        (returnType (CloneTypeWithF32ElementType $y)))),
    /*Truncate=*/ConstBoolAttrFalse),
  [(IsBF16ElementType $x),
   (IsBF16ElementType $y),
   (IsBF16ElementType $res)]>;
