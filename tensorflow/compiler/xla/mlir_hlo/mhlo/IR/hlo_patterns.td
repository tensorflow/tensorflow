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
include "mhlo/IR/hlo_ops.td"

def HasSameType : Constraint<CPred<"$0.getType() == $1.getType()">>;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

// Canonicalization patterns.

def DynamicBroadcastToOwnShape_1 : Pat<
  (MHLO_DynamicBroadcastInDimOp:$op $x,
    (Shape_ToExtentTensorOp (Shape_ShapeOfOp $x)), $_, $_, $_),
  (replaceWithValue $x)>;
def DynamicBroadcastToOwnShape_2 : Pat<
  (MHLO_DynamicBroadcastInDimOp:$op $x,
    (Shape_ShapeOfOp $x), $attr0, $attr1, $attr2),
  (replaceWithValue $x)>;
def DynamicBroadcastToOwnShape_3 : Pat<
  (MHLO_DynamicBroadcastInDimOp:$op $x,
    (Tensor_CastOp (Shape_ToExtentTensorOp (Shape_ShapeOfOp $x))),
    $attr0, $attr1, $attr2),
  (Tensor_CastOp $x)>;
def DynamicBroadcastToOwnShape_4 : Pat<
  (MHLO_DynamicBroadcastInDimOp:$op $x,
    (Tensor_CastOp (Shape_ShapeOfOp $x)), $attr0, $attr1, $attr2),
  (Tensor_CastOp $x)>;

def ShapeOfDynamicReshape : Pat<
  (Shape_ShapeOfOp:$op (MHLO_DynamicReshapeOp $x, $shape)),
  (replaceWithValue $shape),
  [(HasSameType $shape, $op)]>;

def IdentityBroadcastReshape : Pat<
  (MHLO_ReshapeOp:$op (MHLO_BroadcastOp $input, $dims)),
  (replaceWithValue $input),
  [(HasSameType $input, $op)]>;

def IdentityBroadcastInDimReshape : Pat<
  (MHLO_ReshapeOp:$op (MHLO_BroadcastInDimOp $input, $dims)),
  (replaceWithValue $input),
  [(HasSameType $input, $op)]>;

def EliminateIdentityConvert : Pat<
  (MHLO_ConvertOp:$res $src),
  (replaceWithValue $src),
  [(HasSameType $res, $src)]>;

def EliminateRedundantReshape : Pat<
  (MHLO_ReshapeOp:$res (MHLO_ReshapeOp $src)),
  (replaceWithValue $src),
  [(HasSameType $res, $src)]>;

def EliminateIdentityReshape : Pat<
  (MHLO_ReshapeOp:$res $src),
  (replaceWithValue $src),
  [(HasSameType $res, $src)]>;

// select(not(p), t, f) => select(p, f, t)
def FusePredNegIntoSelect : Pat<
  (MHLO_SelectOp (MHLO_NotOp $pred), $on_true, $on_false),
  (MHLO_SelectOp $pred, $on_false, $on_true)>;

// select(broadcast(not(p)), t, f) => select(broadcast(p), f, t)
def FuseBroadcastedPredNegIntoSelect : Pat<
  (MHLO_SelectOp
    (MHLO_BroadcastInDimOp:$b (MHLO_NotOp $pred), $broadcast_dimensions),
    $on_true, $on_false),
  (MHLO_SelectOp
    (MHLO_BroadcastInDimOp $pred, $broadcast_dimensions, (returnType $b)),
    $on_false, $on_true),
  [(HasOneUse $b)]>;
