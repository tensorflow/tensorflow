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

// This is the legalization pattern definition file for HLO to TF.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops.td"

def : Pat<(HLO_ConstOp $value), (TF_ConstOp $value)>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

// Check that two values can be broadcasted together
// TODO(jpienaar): Move somewhere more general
def AreBroadcastCompatible : Constraint<CPred<"AreBroadcastCompatible($0, $1)">,
    "types must be broadcastable">;

foreach fromToBinPair = [[HLO_AddOp, TF_AddV2Op],
                         [HLO_DivOp, TF_DivOp],
                         [HLO_ShiftLeftOp, TF_LeftShiftOp],
                         [HLO_MaxOp, TF_MaximumOp],
                         [HLO_MinOp, TF_MinimumOp],
                         [HLO_MulOp, TF_MulOp],
                         [HLO_PowOp, TF_PowOp],
                         [HLO_SubOp, TF_SubOp],
                         [HLO_Atan2Op, TF_Atan2Op],
                         [HLO_RemOp, TF_ModOp]] in
  def : Pat<(fromToBinPair[0] $l, $r, $_), (fromToBinPair[1] $l, $r),
            [(AreBroadcastCompatible $l, $r)]>;

foreach pair  = [[HLO_AndOp, TF_BitwiseAndOp],
                 [HLO_OrOp, TF_BitwiseOrOp],
                 [HLO_XorOp, TF_BitwiseXorOp]] in
  def : Pat<(pair[0] TF_IntTensor:$l, TF_IntTensor:$r, $_), (pair[1] $l, $r),
            [(AreBroadcastCompatible $l, $r)]>;

foreach pair  = [[HLO_AndOp, TF_LogicalAndOp],
                 [HLO_OrOp, TF_LogicalOrOp]] in
  def : Pat<(pair[0] I1Tensor:$l, I1Tensor:$r, $_), (pair[1] $l, $r),
            [(AreBroadcastCompatible $l, $r)]>;

def : Pat<(HLO_ShiftRightArithmeticOp $l, $r, $_), (TF_RightShiftOp $l, $r),
          [(AreBroadcastCompatible $l, $r)]>;
def : Pat<(HLO_ShiftRightLogicalOp $l, $r, $_), (TF_RightShiftOp $l, $r),
          [(AreBroadcastCompatible $l, $r)]>;

def : Pat<(HLO_FloorOp (HLO_DivOp $l, $r, $_)), (TF_FloorDivOp $l, $r),
          [(AreBroadcastCompatible $l, $r)]>;

def : Pat<(HLO_ComplexOp $r, $i), (TF_ComplexOp $r, $i)>;
//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(HLO_ConvertOp HLO_Tensor:$operand),
          (TF_CastOp $operand, ConstBoolAttrFalse)>;

foreach Mapping = [[HLO_AbsOp, TF_AbsOp],
                   [HLO_BitcastConvertOp, TF_BitcastOp],
                   [HLO_CeilOp, TF_CeilOp],
                   [HLO_CosOp, TF_CosOp],
                   [HLO_ExpOp, TF_ExpOp],
                   [HLO_Expm1Op, TF_Expm1Op],
                   [HLO_FloorOp, TF_FloorOp],
                   [HLO_ImagOp, TF_ImagOp],
                   [HLO_IsFiniteOp, TF_IsFiniteOp],
                   [HLO_LogOp, TF_LogOp],
                   [HLO_Log1pOp, TF_Log1pOp],
                   [HLO_NotOp, TF_LogicalNotOp],
                   [HLO_NegOp, TF_NegOp],
                   [HLO_RealOp, TF_RealOp],
                   [HLO_RsqrtOp, TF_RsqrtOp],
                   [HLO_SinOp, TF_SinOp],
                   [HLO_SignOp, TF_SignOp],
                   [HLO_SqrtOp, TF_SqrtOp],
                   [HLO_TanhOp, TF_TanhOp]] in
 def : Pat<(Mapping[0] TF_IntOrFpTensor:$input), (Mapping[1] $input)>;

def : Pat<(HLO_AbsOp TF_ComplexTensor:$arg), (TF_ComplexAbsOp $arg)>;

def : Pat<(HLO_BroadcastOp $arg, $shape),
          (TF_BroadcastToOp $arg, (TF_ConstOp $shape))>;
def : Pat<(HLO_TransposeOp $arg, $permutation),
          (TF_TransposeOp $arg, (TF_ConstOp $permutation))>;
def : Pat<(HLO_ReverseOp $op, $dims), (TF_ReverseV2Op $op, (TF_ConstOp $dims))>;

//===----------------------------------------------------------------------===//
// Ternary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(HLO_ClampOp $min, $arg, $max),
          (TF_MaximumOp (TF_MinimumOp $arg, $max), $min)>;
def : Pat<(HLO_SelectOp $cond, $t, $e), (TF_SelectOp $cond, $t, $e)>;

//===----------------------------------------------------------------------===//
// Variadic op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(HLO_ConcatenateOp $inputs, $dim),
          (TF_ConcatV2Op $inputs, (TF_ConstOp $dim))>;

//===----------------------------------------------------------------------===//
// Compare op patterns.
//===----------------------------------------------------------------------===//

foreach p = [[TF_EqualOp, HLO_COMPARISON_DIRECTION_EQ],
             [TF_NotEqualOp, HLO_COMPARISON_DIRECTION_NE]] in
  def : Pat<(HLO_CompareOp $l, $r, $_, p[1]), (p[0] $l, $r, ConstBoolAttrTrue),
            [(AreBroadcastCompatible $l, $r)]>;

foreach pair = [[TF_GreaterEqualOp, HLO_COMPARISON_DIRECTION_GE],
                [TF_GreaterOp, HLO_COMPARISON_DIRECTION_GT],
                [TF_LessEqualOp, HLO_COMPARISON_DIRECTION_LE],
                [TF_LessOp, HLO_COMPARISON_DIRECTION_LT]] in
  def : Pat<(HLO_CompareOp $l, $r, $_, pair[1]), (pair[0] $l, $r),
            [(AreBroadcastCompatible $l, $r)]>;
