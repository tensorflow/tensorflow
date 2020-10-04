/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Canonicalization patterns for the MHLO dialect.

include "mlir/Dialect/Shape/IR/ShapeOps.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

def EqualBinaryOperands : Constraint<CPred<"$0 == $1">>;

// Canonicalization patterns.

def DynamicBroadcastToOwnShape_1 : Pat<
   (HLO_DynamicBroadcastInDimOp:$op $arg0,
       (Shape_ToExtentTensorOp (Shape_ShapeOfOp $arg1)), $attr),
  (replaceWithValue $arg0), [(EqualBinaryOperands $arg0, $arg1)]>;
def DynamicBroadcastToOwnShape_2 : Pat<
  (HLO_DynamicBroadcastInDimOp:$op $arg0, (Shape_ShapeOfOp $arg1), $attr),
  (replaceWithValue $arg0), [(EqualBinaryOperands $arg0, $arg1)]>;

