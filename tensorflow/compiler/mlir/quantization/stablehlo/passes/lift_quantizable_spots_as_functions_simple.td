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
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "stablehlo/dialect/StablehloOps.td"
include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.td"
include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.td"

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops as functions
//===----------------------------------------------------------------------===//

def LiftConv : Pat<
  (StableHLO_ConvolutionOp:$res $lhs, $rhs, $window_strides, $padding,
      $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
      $feature_group_count, $batch_group_count, $precision_config),
  (LiftAsTFXlaCallModule<"composite_conv_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"window_strides"> (DefaultOrNullAttr $window_strides)),
      (NamedAttr<"padding"> (DefaultOrNullAttr $padding)),
      (NamedAttr<"lhs_dilation"> (DefaultOrNullAttr $lhs_dilation)),
      (NamedAttr<"rhs_dilation"> (DefaultOrNullAttr $rhs_dilation)),
      (NamedAttr<"window_reversal"> (DefaultOrNullAttr $window_reversal)),
      (NamedAttr<"dimension_numbers"> $dimension_numbers),
      (NamedAttr<"feature_group_count"> $feature_group_count),
      (NamedAttr<"batch_group_count"> $batch_group_count),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)))),
  [(IsNotInLiftedFunc $res)], [], (addBenefit 1)>;

def LiftDotGeneral : Pat<
  (StableHLO_DotGeneralOp:$res
      $lhs, $rhs, $dot_dimension_numbers, $precision_config),
  (LiftAsTFXlaCallModule<"composite_dot_general_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)))),
  [(IsNotInLiftedFunc $res)], [], (addBenefit 1)>;

def LiftGather : Pat<
  (StableHLO_GatherOp:$res
      $operand, $start_indices, $dimension_numbers, $slice_sizes, $indices_are_sorted),
  (LiftAsTFXlaCallModule<"composite_gather_fn">
    (ArgumentList $operand, $start_indices),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dimension_numbers"> $dimension_numbers),
      (NamedAttr<"slice_sizes"> $slice_sizes),
      (NamedAttr<"indices_are_sorted"> (DefaultOrNullAttr $indices_are_sorted)))),
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $operand)], [], (addBenefit 1)>;

def LiftAbs : Pat<
  (StableHLO_AbsOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_abs_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftAdd : Pat<
  (StableHLO_AddOp:$res
      $lhs, $rhs),
  (LiftAsTFXlaCallModule<"composite_add_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftCbrt : Pat<
  (StableHLO_CbrtOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_cbrt_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftCeil : Pat<
  (StableHLO_CeilOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_ceil_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftCosine : Pat<
  (StableHLO_CosineOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_cosine_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftDiv : Pat<
  (StableHLO_DivOp:$res
      $lhs, $rhs),
  (LiftAsTFXlaCallModule<"composite_div_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftExp : Pat<
  (StableHLO_ExpOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_exp_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftExpm1 : Pat<
  (StableHLO_Expm1Op:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_expm1_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftFloor : Pat<
  (StableHLO_FloorOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_floor_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftLog : Pat<
  (StableHLO_LogOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_log_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftLog1p : Pat<
  (StableHLO_Log1pOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_log1p_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftLogistic : Pat<
  (StableHLO_LogisticOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_logistic_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftMul : Pat<
  (StableHLO_MulOp:$res
      $lhs, $rhs),
  (LiftAsTFXlaCallModule<"composite_mul_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftNeg : Pat<
  (StableHLO_NegOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_neg_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;


def LiftSign : Pat<
  (StableHLO_SignOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_sign_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftSine : Pat<
  (StableHLO_SineOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_sine_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftSqrt : Pat<
  (StableHLO_SqrtOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_sqrt_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftSubtract : Pat<
  (StableHLO_SubtractOp:$res
      $lhs, $rhs),
  (LiftAsTFXlaCallModule<"composite_subtract_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

def LiftTanh : Pat<
  (StableHLO_TanhOp:$res
      $operand),
  (LiftAsTFXlaCallModule<"composite_tanh_fn">
    (ArgumentList $operand),
    (ResultList $res)),
  [(IsNotInLiftedFunc $res), (IsNotInStableHloOpRegion $res)], [], (addBenefit 1)>;

