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
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

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

