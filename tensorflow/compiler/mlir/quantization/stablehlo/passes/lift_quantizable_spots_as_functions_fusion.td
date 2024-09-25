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

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Shape/IR/ShapeOps.td"
include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "stablehlo/dialect/StablehloOps.td"
include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.td"
include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.td"

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops with bias as functions
//===----------------------------------------------------------------------===//

def LiftDotGeneralWithBiasSameShape : Pat<
  (StableHLO_AddOp:$res
    (StableHLO_DotGeneralOp
        $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
    $bias),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_same_shape_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias)], [], (addBenefit 5)>;

def LiftConvWithBiasSameShape : Pat<
  (StableHLO_AddOp:$res
    (StableHLO_ConvolutionOp $lhs, $rhs, $window_strides, $padding,
        $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
        $feature_group_count, $batch_group_count, $precision_config),
    $bias),
  (LiftAsTFXlaCallModule<"composite_conv_with_bias_same_shape_fn">
    (ArgumentList $lhs, $rhs, $bias),
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
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias)], [], (addBenefit 5)>;


def LiftConvWithBias : Pat<
  (StableHLO_AddOp:$res
    (StableHLO_ConvolutionOp $lhs, $rhs, $window_strides, $padding,
        $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
        $feature_group_count, $batch_group_count, $precision_config),
    (StableHLO_BroadcastInDimOp $bias, $dims)),
  (LiftAsTFXlaCallModule<"composite_conv_with_bias_fn">
    (ArgumentList $lhs, $rhs, $bias),
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
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias)], [], (addBenefit 5)>;

def LiftDotGeneralWithBias : Pat<
  (StableHLO_AddOp:$res
    (StableHLO_DotGeneralOp
        $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
    (StableHLO_BroadcastInDimOp $bias, $dims)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias)], [], (addBenefit 5)>;

def LiftConvWithBiasDynamic : Pat<
  (StableHLO_AddOp:$res
    (StableHLO_ConvolutionOp:$conv_0 $lhs, $rhs, $window_strides, $padding,
        $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
        $feature_group_count, $batch_group_count, $precision_config),
    (StableHLO_DynamicBroadcastInDimOp
      $bias,
      (Shape_ShapeOfOp $conv_1), $_, $_, $_)),
  (LiftAsTFXlaCallModule<"composite_conv_with_bias_dynamic_fn">
    (ArgumentList $lhs, $rhs, $bias),
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
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias), (AreTheSameValue $conv_0, $conv_1)], [], (addBenefit 10)>;

def LiftDotGeneralWithBiasDynamic : Pat<
  (StableHLO_AddOp:$res
    (StableHLO_DotGeneralOp:$dot_general_0 $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
    (StableHLO_DynamicBroadcastInDimOp
      $bias,
      (Shape_ShapeOfOp $dot_general_1), $_, $_, $_)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_dynamic_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias), (AreTheSameValue $dot_general_0, $dot_general_1)], [], (addBenefit 10)>;

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops with activation as functions
//===----------------------------------------------------------------------===//

def LiftConvWithRelu : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_ConvolutionOp $lhs, $rhs, $window_strides, $padding,
        $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
        $feature_group_count, $batch_group_count, $precision_config),
    (StableHLO_ConstantOp $cst)),
  (LiftAsTFXlaCallModule<"composite_conv_with_relu_fn">
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
  [(IsNotInLiftedFunc $res), (FloatValueEquals<"0"> $cst)], [], (addBenefit 10)>;

def LiftDotGeneralWithRelu : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_DotGeneralOp
        $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
    (StableHLO_ConstantOp $cst)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_relu_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst)], [], (addBenefit 10)>;

def LiftConvWithReluDynamic : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_ConvolutionOp:$conv_0 $lhs, $rhs, $window_strides, $padding,
        $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
        $feature_group_count, $batch_group_count, $precision_config),
    (StableHLO_DynamicBroadcastInDimOp
      (StableHLO_ConstantOp $cst),
      (Shape_ShapeOfOp $conv_1), $_, $_, $_)),
  (LiftAsTFXlaCallModule<"composite_conv_with_relu_dynamic_fn">
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
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst), (AreTheSameValue $conv_0, $conv_1)], [], (addBenefit 15)>;

def LiftDotGeneralWithReluDynamic : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_DotGeneralOp:$dot_general_0 $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
    (StableHLO_DynamicBroadcastInDimOp
      (StableHLO_ConstantOp $cst),
      (Shape_ShapeOfOp $dot_general_1), $_, $_, $_)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_relu_dynamic_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst), (AreTheSameValue $dot_general_0, $dot_general_1)], [], (addBenefit 15)>;

def LiftConvWithRelu6 : Pat<
  (StableHLO_ClampOp:$res
    (StableHLO_ConstantOp $cst_0),
    (StableHLO_ConvolutionOp $lhs, $rhs, $window_strides, $padding,
        $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
        $feature_group_count, $batch_group_count, $precision_config),
    (StableHLO_ConstantOp $cst_1)),
  (LiftAsTFXlaCallModule<"composite_conv_with_relu6_fn">
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
  [(IsNotInLiftedFunc $res), (FloatValueEquals<"0"> $cst_0), (FloatValueEquals<"6"> $cst_1)], [], (addBenefit 10)>;

def LiftDotGeneralWithRelu6 : Pat<
  (StableHLO_ClampOp:$res
    (StableHLO_ConstantOp $cst_0),
    (StableHLO_DotGeneralOp
        $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
    (StableHLO_ConstantOp $cst_1)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_relu6_fn">
    (ArgumentList $lhs, $rhs),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res), (FloatValueEquals<"0"> $cst_0), (FloatValueEquals<"6"> $cst_1)], [], (addBenefit 10)>;

//===----------------------------------------------------------------------===//
// Pattern rules for lifting ops with bias and activation as functions
//===----------------------------------------------------------------------===//

def LiftDotGeneralWithBiasSameShapeAndRelu : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_AddOp
      (StableHLO_DotGeneralOp
          $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
      $bias),
    (StableHLO_ConstantOp $cst)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_same_shape_and_relu_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst), (IsStableHLOConstantOp $bias)], [], (addBenefit 10)>;

def LiftConvWithBiasSameShapeAndRelu : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_AddOp
      (StableHLO_ConvolutionOp $lhs, $rhs, $window_strides, $padding,
          $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
          $feature_group_count, $batch_group_count, $precision_config),
      $bias),
    (StableHLO_ConstantOp $cst)),
  (LiftAsTFXlaCallModule<"composite_conv_with_bias_same_shape_and_relu_fn">
    (ArgumentList $lhs, $rhs, $bias),
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
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst), (IsStableHLOConstantOp $bias)], [], (addBenefit 10)>;


def LiftConvWithBiasAndRelu : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_AddOp
      (StableHLO_ConvolutionOp $lhs, $rhs, $window_strides, $padding,
          $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
          $feature_group_count, $batch_group_count, $precision_config),
      (StableHLO_BroadcastInDimOp $bias, $dims)),
    (StableHLO_ConstantOp $cst)),
  (LiftAsTFXlaCallModule<"composite_conv_with_bias_and_relu_fn">
    (ArgumentList $lhs, $rhs, $bias),
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
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst), (IsStableHLOConstantOp $bias)], [], (addBenefit 10)>;

def LiftDotGeneralWithBiasAndRelu : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_AddOp
      (StableHLO_DotGeneralOp
          $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
      (StableHLO_BroadcastInDimOp $bias, $dims)),
    (StableHLO_ConstantOp $cst)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_and_relu_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst), (IsStableHLOConstantOp $bias)], [], (addBenefit 10)>;

def LiftConvWithBiasAndReluDynamic : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_AddOp:$add_0
      (StableHLO_ConvolutionOp:$conv_0 $lhs, $rhs, $window_strides, $padding,
          $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
          $feature_group_count, $batch_group_count, $precision_config),
      (StableHLO_DynamicBroadcastInDimOp
        $bias,
        (Shape_ShapeOfOp $conv_1), $_, $_, $_)),
    (StableHLO_DynamicBroadcastInDimOp
      (StableHLO_ConstantOp $cst),
      (Shape_ShapeOfOp $add_1), $_, $_, $_)),
  (LiftAsTFXlaCallModule<"composite_conv_with_bias_and_relu_dynamic_fn">
    (ArgumentList $lhs, $rhs, $bias),
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
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst), (IsStableHLOConstantOp $bias), (AreTheSameValue $conv_0, $conv_1), (AreTheSameValue $add_0, $add_1)], [], (addBenefit 15)>;

def LiftDotGeneralWithBiasAndReluDynamic : Pat<
  (StableHLO_MaxOp:$res
    (StableHLO_AddOp:$add_0
      (StableHLO_DotGeneralOp:$dot_general_0 $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
      (StableHLO_DynamicBroadcastInDimOp
        $bias,
        (Shape_ShapeOfOp $dot_general_1), $_, $_, $_)),
    (StableHLO_DynamicBroadcastInDimOp
      (StableHLO_ConstantOp $cst),
      (Shape_ShapeOfOp $add_1), $_, $_, $_)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_and_relu_dynamic_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res),
   (FloatValueEquals<"0"> $cst), (IsStableHLOConstantOp $bias), (AreTheSameValue $dot_general_0, $dot_general_1), (AreTheSameValue $add_0, $add_1)], [], (addBenefit 15)>;

def LiftDotGeneralWithBiasSameShapeAndRelu6 : Pat<
  (StableHLO_ClampOp:$res
    (StableHLO_ConstantOp $cst_0),
    (StableHLO_AddOp
      (StableHLO_DotGeneralOp
          $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
      $bias),
    (StableHLO_ConstantOp $cst_1)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_same_shape_and_relu6_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias), (FloatValueEquals<"0"> $cst_0), (FloatValueEquals<"6"> $cst_1)], [], (addBenefit 10)>;

def LiftConvWithBiasAndRelu6 : Pat<
  (StableHLO_ClampOp:$res
    (StableHLO_ConstantOp $cst_0),
    (StableHLO_AddOp
      (StableHLO_ConvolutionOp $lhs, $rhs, $window_strides, $padding,
          $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
          $feature_group_count, $batch_group_count, $precision_config),
      (StableHLO_BroadcastInDimOp $bias, $dims)),
    (StableHLO_ConstantOp $cst_1)),
  (LiftAsTFXlaCallModule<"composite_conv_with_bias_and_relu6_fn">
    (ArgumentList $lhs, $rhs, $bias),
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
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias), (FloatValueEquals<"0"> $cst_0), (FloatValueEquals<"6"> $cst_1)], [], (addBenefit 10)>;

def LiftDotGeneralWithBiasAndRelu6 : Pat<
  (StableHLO_ClampOp:$res
    (StableHLO_ConstantOp $cst_0),
    (StableHLO_AddOp
      (StableHLO_DotGeneralOp
          $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
      (StableHLO_BroadcastInDimOp $bias, $dims)),
    (StableHLO_ConstantOp $cst_1)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_and_relu6_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias), (FloatValueEquals<"0"> $cst_0), (FloatValueEquals<"6"> $cst_1)], [], (addBenefit 10)>;

def LiftConvWithBiasAndRelu6Dynamic : Pat<
  (StableHLO_ClampOp:$res
    (StableHLO_ConstantOp $cst_0),
    (StableHLO_AddOp
      (StableHLO_ConvolutionOp:$conv_0 $lhs, $rhs, $window_strides, $padding,
          $lhs_dilation, $rhs_dilation, $window_reversal, $dimension_numbers,
          $feature_group_count, $batch_group_count, $precision_config),
      (StableHLO_DynamicBroadcastInDimOp
        $bias,
        (Shape_ShapeOfOp $conv_1), $_, $_, $_)),
    (StableHLO_ConstantOp $cst_1)),
  (LiftAsTFXlaCallModule<"composite_conv_with_bias_and_relu6_dynamic_fn">
    (ArgumentList $lhs, $rhs, $bias),
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
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias), (FloatValueEquals<"0"> $cst_0), (FloatValueEquals<"6"> $cst_1), (AreTheSameValue $conv_0, $conv_1)], [], (addBenefit 15)>;

def LiftDotGeneralWithBiasAndRelu6Dynamic : Pat<
  (StableHLO_ClampOp:$res
    (StableHLO_ConstantOp $cst_0),
    (StableHLO_AddOp
      (StableHLO_DotGeneralOp:$dot_general_0
          $lhs, $rhs, $dot_dimension_numbers, $precision_config, $algorithm),
      (StableHLO_DynamicBroadcastInDimOp
        $bias,
        (Shape_ShapeOfOp $dot_general_1), $_, $_, $_)),
    (StableHLO_ConstantOp $cst_1)),
  (LiftAsTFXlaCallModule<"composite_dot_general_with_bias_and_relu6_dynamic_fn">
    (ArgumentList $lhs, $rhs, $bias),
    (ResultList $res),
    (NamedAttributeList
      (NamedAttr<"dot_dimension_numbers"> $dot_dimension_numbers),
      (NamedAttr<"precision_config"> (DefaultOrNullAttr $precision_config)),
      (NamedAttr<"algorithm"> (DefaultOrNullAttr $algorithm)))),
  [(IsNotInLiftedFunc $res), (IsStableHLOConstantOp $bias), (FloatValueEquals<"0"> $cst_0), (FloatValueEquals<"6"> $cst_1), (AreTheSameValue $dot_general_0, $dot_general_1)], [], (addBenefit 15)>;
