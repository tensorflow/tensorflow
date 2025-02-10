/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// Patterns to optimize explicit broadcasting-like patterns, because TFLite Ops
// support implicit broadcasting.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/utils/utils.td"

////////////////////////////////////////////////////////////////////////////////
// Patterns on TFL::Select*Op to optimize explicit broadcasting-like patterns.
////////////////////////////////////////////////////////////////////////////////

// Pattern for fusing splat const into select op.
multiclass FuseSplatConstIntoSelectOp<Op SelectOp> {
  def FuseSplatConstLhsInto#SelectOp : Pat<
    (SelectOp:$result
      AnyStaticShapeTensor:$input1,
      (Arith_ConstantOp:$constant_value SplatElementsAttr:$constant_attr),
      AnyStaticShapeTensor:$input2),
    (TFL_SelectV2Op
      $input1,
      (Arith_ConstantOp (GetScalarElementsAttrFromSplat $constant_attr)),
      $input2),
    // Check if condition or rhs will promote the required broadcasting.
    [(HasRankAtLeast<2> $constant_attr),
     (OperandsBroadcastToOutputType $input1, $input2, $result),
     (HasRankAtMost<5> $constant_value)]>;

  def FuseSplatConstRhsInto#SelectOp : Pat<
    (SelectOp:$result
      AnyStaticShapeTensor:$input1,
      AnyStaticShapeTensor:$input2,
      (Arith_ConstantOp:$constant_value SplatElementsAttr:$constant_attr)),
    (TFL_SelectV2Op
      $input1, $input2,
      (Arith_ConstantOp (GetScalarElementsAttrFromSplat $constant_attr))),
    // Check if condition or lhs will promote the required broadcasting.
    [(HasRankAtLeast<2> $constant_attr),
     (OperandsBroadcastToOutputType $input1, $input2, $result),
     (HasRankAtMost<5> $constant_value)]>;
}

// Pattern for skipping FillOp if it is mainly for broadcasting and the
// Op is already supporting broadcasting.
multiclass FuseFillOpBroadcastIntoFollowingSelectOp<Op SelectOp> {
  def FoldFillOpIntoSelectOpRHS#SelectOp : Pat<
    (SelectOp:$result
      AnyStaticShapeTensor:$input1,
      AnyStaticShapeTensor:$input2,
      (TFL_FillOp:$fill_output $fill_dims,
        (Arith_ConstantOp:$fill_value $val))),
    (TFL_SelectV2Op $input1, $input2, $fill_value),
  [(OperandsBroadcastToOutputType $input1, $input2, $result),
   (HasRankAtMost<5> $result),
   (IsRankLessThanEqualTo $fill_output, $result)]>;

  def FoldFillOpIntoSelectOpLHS#SelectOp : Pat<
    (SelectOp:$result
      AnyStaticShapeTensor:$input1,
      (TFL_FillOp:$fill_output $fill_dims, (Arith_ConstantOp:$fill_value $val)),
      AnyStaticShapeTensor:$input2),
    (TFL_SelectV2Op $input1, $fill_value, $input2),
  [(OperandsBroadcastToOutputType $input1, $input2, $result),
   (HasRankAtMost<5> $result),
   (IsRankLessThanEqualTo $fill_output, $result)]>;
}

multiclass FuseBroadcastToIntoSelectOp<Op SelectOp> {
    // Fuse select(broadcast_to(input, shape), x, y) -> selectV2(input, x, y)
  // Also, fuse selectv2(broadcast_to(input, shape), x, y) -> selectV2(input, x, y)
  // It is safe to perform this transform here because-
  // the shapes of `pre_broadcast` and `dim` must be broadcast
  // compatible for the `broadcast_to` op to be valid.
  // And considering, `shape(post_broadcast)` == `shape(%input1)`,
  // `post_broadcast` is broadcast compatible with `input1`.
  def FuseBroadcastConditionInto#SelectOp : Pat<
    (SelectOp:$result
      (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim),
      AnyStaticShapeTensor:$input1, AnyStaticShapeTensor:$input2),
    (TFL_SelectV2Op $pre_broadcast, $input1, $input2),
    [(IsRankLessThanEqualTo $post_broadcast, $result),
     (OperandsBroadcastToOutputType $input1, $input2, $result),
     (HasRankAtMost<5> $post_broadcast)]>;

  def FuseBroadcastLhsInto#SelectOp : Pat<
    (SelectOp:$result
      AnyStaticShapeTensor:$input1,
      (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim),
      AnyStaticShapeTensor:$input2),
    (TFL_SelectV2Op $input1, $pre_broadcast, $input2),
    [(IsRankLessThanEqualTo $post_broadcast, $result),
     (OperandsBroadcastToOutputType $input1, $input2, $result),
     (HasRankAtMost<5> $post_broadcast)]>;

  def FuseBroadcastRhsInto#SelectOp : Pat<
    (SelectOp:$result
      AnyStaticShapeTensor:$input1,
      AnyStaticShapeTensor:$input2,
      (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim)),
    (TFL_SelectV2Op $input1, $input2, $pre_broadcast),
    [(IsRankLessThanEqualTo $post_broadcast, $result),
     (OperandsBroadcastToOutputType $input1, $input2, $result),
     (HasRankAtMost<5> $post_broadcast)]>;
}

foreach SelectOp = [TFL_SelectOp, TFL_SelectV2Op] in {
  // Fuse splat const into select op.
  defm : FuseSplatConstIntoSelectOp<SelectOp>;

  // Fuse FillOp broadcast into following select op.
  defm : FuseFillOpBroadcastIntoFollowingSelectOp<SelectOp>;

  // Fuse broadcast to into select op.
  defm : FuseBroadcastToIntoSelectOp<SelectOp>;
}
