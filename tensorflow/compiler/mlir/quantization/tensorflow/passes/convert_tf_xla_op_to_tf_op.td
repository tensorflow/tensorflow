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
include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_generated_ops.td"

// Only handles the case where precision config is default.
def IsPrecisionEmpty :
  Constraint<CPred<"IsPrecisionEmpty($0)">>;

// Creates Einsum Op from XlaDotV2 Op by generating equation.
def CreateEinsumOpFromXlaDotV2Op : NativeCodeCall<
  "CreateEinsumOpFromXlaDotV2Op($_builder, $_loc, $0...)">;

// Convert XlaDotV2 Op to Einsum Op with above two functions.
def ConvertXlaDotV2OpToEinsumOp : Pat<
  (TF_XlaDotV2Op:$dot $lhs, $rhs, $dot_dimension_numbers, $precision_config),
  (CreateEinsumOpFromXlaDotV2Op $lhs, $rhs, $dot, $dot_dimension_numbers),
  [(IsPrecisionEmpty $precision_config)]>;

// Only handles the case where batch_dimension is empty.
def IsXlaGatherWithoutBatch :
  Constraint<CPred<"IsXlaGatherWithoutBatch($0, $1)">>;

// Create Slice op from XlaGather op without batch dimension.
def CreateSliceAndReshapeOpFromXlaGatherOpWithoutBatch : NativeCodeCall<
  "CreateSliceAndReshapeOpFromXlaGatherOpWithoutBatch($_builder, $_loc, $0...)">;

// Convert XlaGather op without batch to Slice op with above two functions.
def ConvertXlaGatherOpWithoutBatch : Pat<
  (TF_XlaGatherOp:$gather $operand,
    $start_indices, $slice_sizes, $dimension_numbers, $indices_are_sorted),
  (CreateSliceAndReshapeOpFromXlaGatherOpWithoutBatch $operand,
    $start_indices, $slice_sizes, $gather, $dimension_numbers),
  [(IsXlaGatherWithoutBatch $operand, $start_indices)]>;
