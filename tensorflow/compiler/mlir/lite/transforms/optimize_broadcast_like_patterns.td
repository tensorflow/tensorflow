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

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

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

// Checks if the value has only one use or used by elementwise op.
def HasOneUseOrUsedByElementwiseOp : Constraint<CPred<
  "($0.hasOneUse() || llvm::all_of($0.getUsers(), [](Operation* user){"
  "  return llvm::isa<TFL::AddOp, TFL::SubOp, TFL::MulOp, TFL::DivOp, "
  "                   TFL::MinimumOp, TFL::MaximumOp, TFL::LessOp, "
  "                   TFL::LessEqualOp, TFL::GreaterOp, TFL::GreaterEqualOp, "
  "                   TFL::NotEqualOp, TFL::EqualOp, TFL::PowOp, "
  "                   TFL::SquaredDifferenceOp, TFL::FloorDivOp, "
  "                   TFL::FloorModOp, TFL::AbsOp, TFL::NegOp, TFL::SqrtOp, "
  "                   TFL::RsqrtOp, TFL::SquareOp, TFL::LogicalNotOp, "
  "                   TFL::LogicalAndOp, TFL::LogicalOrOp, TFL::ExpOp, "
  "                   TFL::SelectOp, TFL::SelectV2Op, TFL::CeilOp, "
  "                   TFL::FloorOp, TFL::RoundOp, TFL::SinOp, TFL::CosOp, "
  "                   TFL::TanhOp, TFL::LogOp>(user);"
  "}))"
  >>;

////////////////////////////////////////////////////////////////////////////////
// Patterns on TFL::<BinaryOp> to optimize explicit broadcast_to patterns.
////////////////////////////////////////////////////////////////////////////////

// ConvertResultsBroadcastableShapeOp pattern in this pass fuses the
// broadcast_to op into the TFL ops that support implicit broadcasting.
// These Patterns below aims to handle all other broadcast_to ops that remain,
// by moving the broadcast_to op after the binary op. This way, the
// broadcast_to op can get the opportunity to be fused into the consumer of the
// binary op.

// TFL_DivOp needs to be handled separately because it supports implicit
// broadcasting only for rank<=5.
def ReorderBroadcastToOpAndDivOpLhs : Pat<
  (TFL_DivOp:$result
      (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim),
      AnyStaticShapeTensor:$input2, $act_fn2),
    (TFL_BroadcastToOp
      (TFL_DivOp $pre_broadcast, $input2, $act_fn2), $dim),
    [(IsNotQuantized $post_broadcast),
     (OperandsDontBroadcastToOutputType $input2, $pre_broadcast, $post_broadcast),
     (HasSameStaticShapes $post_broadcast, $result),
     (HasOneUse $post_broadcast),
     (HasRankAtMost<5> $post_broadcast)]>;

def ReorderBroadcastToOpAndDivOpRhs : Pat<
  (TFL_DivOp:$result
      AnyStaticShapeTensor:$input1,
      (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim), $act_fn2),
    (TFL_BroadcastToOp
      (TFL_DivOp $input1, $pre_broadcast, $act_fn2), $dim),
    [(IsNotQuantized $post_broadcast),
     (OperandsDontBroadcastToOutputType $input1, $pre_broadcast, $post_broadcast),
     (HasSameStaticShapes $post_broadcast, $result),
     (HasOneUse $post_broadcast),
     (HasRankAtMost<5> $post_broadcast)]>;

def ReorderBroadcastToOpAndDivOpWithSplatLhs : Pat<
  (TFL_DivOp:$result
      (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim),
      (Arith_ConstantOp:$constant_value SplatElementsAttr:$constant_attr), $act_fn2),
    (TFL_BroadcastToOp
      (TFL_DivOp $pre_broadcast,
        (Arith_ConstantOp (GetScalarElementsAttrFromSplat $constant_attr)), $act_fn2),
      $dim),
        [(IsNotQuantized $post_broadcast),
        (OperandsDontBroadcastToOutputType $constant_value, $pre_broadcast, $post_broadcast),
         (HasSameStaticShapes $post_broadcast, $result),
         (HasOneUse $post_broadcast),
         (HasRankAtMost<5> $post_broadcast)]>;

    def ReorderBroadcastToOpAndDivOpWithSplat2Rhs : Pat<
      (TFL_DivOp:$result
          (Arith_ConstantOp:$constant_value SplatElementsAttr:$constant_attr),
          (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim), $act_fn2),
        (TFL_BroadcastToOp
          (TFL_DivOp (Arith_ConstantOp (GetScalarElementsAttrFromSplat $constant_attr)), $pre_broadcast, $act_fn2),
          $dim),
        [(IsNotQuantized $post_broadcast),
         (OperandsDontBroadcastToOutputType $constant_value, $pre_broadcast, $post_broadcast),
         (HasSameStaticShapes $post_broadcast, $result),
         (HasOneUse $post_broadcast),
         (HasRankAtMost<5> $post_broadcast)]>;


multiclass ReorderBroadcastToOpAndBinaryOpWithActFn<Op BinaryOp> {
  def ReorderBroadcastToOpAnd#BinaryOp#Lhs : Pat<
    (BinaryOp:$result
      (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim),
      AnyStaticShapeTensor:$input2, $act_fn2),
    (TFL_BroadcastToOp
      (BinaryOp $pre_broadcast, $input2, $act_fn2), $dim),
    [(IsNotQuantized $post_broadcast),
     (OperandsDontBroadcastToOutputType $input2, $pre_broadcast, $post_broadcast),
     (HasSameStaticShapes $post_broadcast, $result),
     (HasOneUse $post_broadcast),
     (HasRankAtMost<6> $post_broadcast)]>;


  def ReorderBroadcastToOpAnd#BinaryOp#Rhs : Pat<
    (BinaryOp:$result
        AnyStaticShapeTensor:$input1,
        (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim), $act_fn2),
      (TFL_BroadcastToOp
        (BinaryOp $input1, $pre_broadcast, $act_fn2), $dim),
      [(IsNotQuantized $post_broadcast),
       (OperandsDontBroadcastToOutputType $input1, $pre_broadcast, $post_broadcast),
       (HasSameStaticShapes $post_broadcast, $result),
       (HasOneUse $post_broadcast),
       (HasRankAtMost<6> $post_broadcast)]>;

  def ReorderBroadcastToOpAnd#BinaryOp#WithSplatLhs : Pat<
    (BinaryOp:$result
        (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim),
        (Arith_ConstantOp:$constant_value SplatElementsAttr:$constant_attr), $act_fn2),
      (TFL_BroadcastToOp
        (BinaryOp $pre_broadcast,
          (Arith_ConstantOp (GetScalarElementsAttrFromSplat $constant_attr)), $act_fn2),
        $dim),
          [(IsNotQuantized $post_broadcast),
           (OperandsDontBroadcastToOutputType $constant_value, $pre_broadcast, $post_broadcast),
           (HasSameStaticShapes $post_broadcast, $result),
           (HasOneUse $post_broadcast),
           (HasRankAtMost<6> $post_broadcast)]>;

  def ReorderBroadcastToOpAnd#BinaryOp#WithSplat2Rhs : Pat<
    (BinaryOp:$result
        (Arith_ConstantOp:$constant_value SplatElementsAttr:$constant_attr),
        (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim), $act_fn2),
      (TFL_BroadcastToOp
        (BinaryOp (Arith_ConstantOp (GetScalarElementsAttrFromSplat $constant_attr)), $pre_broadcast, $act_fn2),
        $dim),
      [(IsNotQuantized $post_broadcast),
       (OperandsDontBroadcastToOutputType $constant_value, $pre_broadcast, $post_broadcast),
       (HasSameStaticShapes $post_broadcast, $result),
       (HasOneUse $post_broadcast),
       (HasRankAtMost<6> $post_broadcast)]>;
}

foreach BinaryOp = [TFL_AddOp, TFL_SubOp, TFL_MulOp] in {
  // Reorder broadcast to after binary op.
  defm : ReorderBroadcastToOpAndBinaryOpWithActFn<BinaryOp>;
}

multiclass ReorderBroadcastToOpAndBinaryOpWithoutActFn<Op BinaryOp> {
  def ReorderBroadcastToOpAnd#BinaryOp#Lhs : Pat<
    (BinaryOp:$result
      (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim),
      AnyStaticShapeTensor:$input2),
    (TFL_BroadcastToOp
      (BinaryOp $pre_broadcast, $input2), $dim),
    [(IsNotQuantized $post_broadcast),
     (OperandsDontBroadcastToOutputType $input2, $pre_broadcast, $post_broadcast),
     (HasSameStaticShapes $post_broadcast, $result),
     (HasOneUse $post_broadcast),
     (HasRankAtMost<4> $post_broadcast)]>;

  def ReorderBroadcastToOpAnd#BinaryOp#Rhs : Pat<
    (BinaryOp:$result
        AnyStaticShapeTensor:$input1,
        (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim)),
      (TFL_BroadcastToOp
        (BinaryOp $input1, $pre_broadcast), $dim),
      [(IsNotQuantized $post_broadcast),
       (OperandsDontBroadcastToOutputType $input1, $pre_broadcast, $post_broadcast),
       (HasSameStaticShapes $post_broadcast, $result),
       (HasOneUse $post_broadcast),
       (HasRankAtMost<4> $post_broadcast)]>;

  def ReorderBroadcastToOpAnd#BinaryOp#WithSplatLhs : Pat<
    (BinaryOp:$result
        (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim),
        (Arith_ConstantOp:$constant_value SplatElementsAttr:$constant_attr)),
      (TFL_BroadcastToOp
        (BinaryOp $pre_broadcast,
          (Arith_ConstantOp (GetScalarElementsAttrFromSplat $constant_attr))),
        $dim),
          [(IsNotQuantized $post_broadcast),
           (OperandsDontBroadcastToOutputType $constant_value, $pre_broadcast, $post_broadcast),
           (HasSameStaticShapes $post_broadcast, $result),
           (HasOneUse $post_broadcast),
           (HasRankAtMost<4> $post_broadcast)]>;

  def ReorderBroadcastToOpAnd#BinaryOp#WithSplat2Rhs : Pat<
    (BinaryOp:$result
        (Arith_ConstantOp:$constant_value SplatElementsAttr:$constant_attr),
        (TFL_BroadcastToOp:$post_broadcast AnyStaticShapeTensor:$pre_broadcast, $dim)),
      (TFL_BroadcastToOp
        (BinaryOp (Arith_ConstantOp (GetScalarElementsAttrFromSplat $constant_attr)), $pre_broadcast),
        $dim),
      [(IsNotQuantized $post_broadcast),
       (OperandsDontBroadcastToOutputType $constant_value, $pre_broadcast, $post_broadcast),
       (HasSameStaticShapes $post_broadcast, $result),
       (HasOneUse $post_broadcast),
       (HasRankAtMost<4> $post_broadcast)]>;
}

foreach BinaryOp = [TFL_MinimumOp, TFL_MaximumOp, TFL_LessOp,
                    TFL_LessEqualOp, TFL_GreaterOp,
                    TFL_GreaterEqualOp, TFL_NotEqualOp, TFL_EqualOp, TFL_PowOp,
                    TFL_SquaredDifferenceOp, TFL_FloorDivOp, TFL_FloorModOp] in {
  // Reorder broadcast to after binary op without act fn.
  defm : ReorderBroadcastToOpAndBinaryOpWithoutActFn<BinaryOp>;
}

////////////////////////////////////////////////////////////////////////////////
// Reorder TFL::<UnaryOp> with the TFL::broadcast_to operator.
////////////////////////////////////////////////////////////////////////////////
multiclass ReorderBroadcastToAndUnaryOp<Op UnaryOp> {
  def ReorderBroadcastToOf#UnaryOp : Pat<
    (UnaryOp (TFL_BroadcastToOp AnyStaticShapeTensor:$input, $dim)),
    (TFL_BroadcastToOp (UnaryOp $input), $dim)>;
}

// TFL_CastOp of requires special handling due to not having a builder, it's
// implemented in native code in ReorderBroadcastToCast.
foreach UnaryOp = [TFL_AbsOp, TFL_CeilOp, TFL_ComplexAbsOp, TFL_CosOp,
                   TFL_DequantizeOp, TFL_EluOp, TFL_ExpOp, TFL_FloorOp,
                   TFL_HardSwishOp, TFL_ImagOp, TFL_LogOp, TFL_LogicalNotOp,
                   TFL_LogisticOp, TFL_NegOp, TFL_RealOp, TFL_Relu0To1Op,
                   TFL_Relu1Op, TFL_Relu6Op, TFL_ReluOp, TFL_RoundOp,
                   TFL_RsqrtOp, TFL_SignOp, TFL_SinOp, TFL_SqrtOp, TFL_SquareOp,
                   TFL_TanhOp, TFL_ZerosLikeOp] in {
  defm : ReorderBroadcastToAndUnaryOp<UnaryOp>;
}

////////////////////////////////////////////////////////////////////////////////
// Remove redundant broadcast_to op.
////////////////////////////////////////////////////////////////////////////////
def RemoveRedundantBroadcastToOp : Pat<
  (TFL_BroadcastToOp:$result AnyStaticShapeTensor:$pre_broadcast, $dim),
  (replaceWithValue $pre_broadcast),
  [(HasSameStaticShapes $pre_broadcast, $result)]>;

////////////////////////////////////////////////////////////////////////////////
// Reorder TFL::SumOp with the TFL::broadcast_to operator.
////////////////////////////////////////////////////////////////////////////////

def HasDistinctBroadcastAndReduceAxes : Constraint<CPred<
    "AreBroadcastAndReductionAxesIndependent($0, $1, $2)">>;

// Pattern to transform tfl.sum(tfl.broadcast_to(input, shape=S1), axis=B, keep_dims=true)
// into tfl.broadcast_to(tfl.sum(input, axis=B, keep_dims=true), shape=S2)
// where S1 is intermediate_target_shape_val, B is reduction_indices_val,
// and S2 is the computed final_target_shape_val (shape of original sum).
def ReorderBroadcastToAfterSumOp : Pat<
  (TFL_SumOp:$original_sum
    (TFL_BroadcastToOp:$intermediate_broadcast
      AnyStaticShapeTensor:$original_input,
      (Arith_ConstantOp $intermediate_target_shape_val)),
    (Arith_ConstantOp I32ElementsAttr:$reduction_indices_val),
    $keep_dims),
  (TFL_BroadcastToOp
    (TFL_SumOp
      $original_input,
      (Arith_ConstantOp $reduction_indices_val),
      $keep_dims),
    (Arith_ConstantOp (GetShapeAttr $original_sum))),
  [(HasOneUse $intermediate_broadcast),
   (HasDistinctBroadcastAndReduceAxes
      $original_input, $reduction_indices_val, $intermediate_target_shape_val),
   ]>;
