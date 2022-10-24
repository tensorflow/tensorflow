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
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.td"

def HasSameType : Constraint<CPred<"$0.getType() == $1.getType()">>;

// Canonicalization patterns.

def DynamicBroadcastToOwnShape_1 : Pat<
  (HLO_DynamicBroadcastInDimOp:$op $x,
    (Shape_ToExtentTensorOp (Shape_ShapeOfOp $x)), $_, $_, $_),
  (replaceWithValue $x)>;
def DynamicBroadcastToOwnShape_2 : Pat<
  (HLO_DynamicBroadcastInDimOp:$op $x,
    (Shape_ShapeOfOp $x), $attr0, $attr1, $attr2),
  (replaceWithValue $x)>;
def DynamicBroadcastToOwnShape_3 : Pat<
  (HLO_DynamicBroadcastInDimOp:$op $x,
    (Tensor_CastOp (Shape_ToExtentTensorOp (Shape_ShapeOfOp $x))),
    $attr0, $attr1, $attr2),
  (Tensor_CastOp $x)>;
def DynamicBroadcastToOwnShape_4 : Pat<
  (HLO_DynamicBroadcastInDimOp:$op $x,
    (Tensor_CastOp (Shape_ShapeOfOp $x)), $attr0, $attr1, $attr2),
  (Tensor_CastOp $x)>;

def ShapeOfDynamicReshape : Pat<
  (Shape_ShapeOfOp:$op (HLO_DynamicReshapeOp $x, $shape)),
  (replaceWithValue $shape),
  [(HasSameType $shape, $op)]>;

def IdentityBroadcastReshape : Pat<
  (HLO_ReshapeOp:$op (HLO_BroadcastOp $input, $dims)),
  (replaceWithValue $input),
  [(HasSameType $input, $op)]>;

def IdentityBroadcastInDimReshape : Pat<
  (HLO_ReshapeOp:$op (HLO_BroadcastInDimOp $input, $dims)),
  (replaceWithValue $input),
  [(HasSameType $input, $op)]>;

def EliminateIdentityConvert : Pat<
  (HLO_ConvertOp:$res $src),
  (replaceWithValue $src),
  [(HasSameType $res, $src)]>;

def EliminateRedundantReshape : Pat<
  (HLO_ReshapeOp:$res (HLO_ReshapeOp $src)),
  (replaceWithValue $src),
  [(HasSameType $res, $src)]>;

def EliminateIdentityReshape : Pat<
  (HLO_ReshapeOp:$res $src),
  (replaceWithValue $src),
  [(HasSameType $res, $src)]>;
