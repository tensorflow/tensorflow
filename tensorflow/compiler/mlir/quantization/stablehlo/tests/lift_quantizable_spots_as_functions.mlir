// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-lift-quantizable-spots-as-functions | FileCheck %s

// CHECK-LABEL: @conv_fn(
// CHECK-SAME:          %[[ARG_0:.*]]: tensor<1x3x3x4xf32>,
// CHECK-SAME:          %[[ARG_1:.*]]: tensor<3x3x4x4xf32>)
func.func @conv_fn(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  func.return %0: tensor<1x3x3x4xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_fn_1
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: return %[[CONV]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_fn(
// CHECK-SAME:                 %[[ARG_0:.*]]: tensor<1x1x167xf32>,
// CHECK-SAME:                 %[[ARG_1:.*]]: tensor<167x64xf32>
func.func @dot_general_fn(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %0 : tensor<1x1x64xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: return %[[DOT_GENERAL:.*]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<3x3x4x4xf32>,
// CHECK-SAME:                    %[[ARG_2:.*]]: tensor<1x3x3x4xf32>)
func.func @conv_with_bias_fn(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<3x3x4x4xf32>, %arg2: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<1x3x3x4xf32>
  func.return %1: tensor<1x3x3x4xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1, %arg2)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_fn_1
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %arg2
// CHECK: return %[[ADD]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x1x167xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<167x64xf32>
// CHECK-SAME:                    %[[ARG_2:.*]]: tensor<1x1x64xf32>)
func.func @dot_general_with_bias_fn(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>, %arg2: tensor<1x1x64xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<1x1x64xf32>
  func.return %1: tensor<1x1x64xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1, %arg2)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg2
// CHECK: return %[[ADD]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_relu_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<3x3x4x4xf32>)
func.func @conv_with_relu_fn(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x3x4xf32>
  %1 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %2 = stablehlo.maximum %1, %0 : tensor<1x3x3x4xf32>
  func.return %2: tensor<1x3x3x4xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_relu_fn_1
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[CONV]], %[[CONST]]
// CHECK: return %[[MAX]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_relu_fn(
// CHECK-SAME:                 %[[ARG_0:.*]]: tensor<1x1x167xf32>,
// CHECK-SAME:                 %[[ARG_1:.*]]: tensor<167x64xf32>
func.func @dot_general_with_relu_fn(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %1 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %2 = stablehlo.maximum %1, %0 : tensor<1x1x64xf32>
  return %2 : tensor<1x1x64xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_relu_fn_1
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[DOT_GENERAL]], %[[CONST]]
// CHECK: return %[[MAX:.*]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// The pattern should not match when the const value for relu is not 0.

// CHECK-LABEL: @conv_with_relu_wrong_const_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<3x3x4x4xf32>)
func.func @conv_with_relu_wrong_const_fn(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<1x3x3x4xf32>
  %1 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %2 = stablehlo.maximum %1, %0 : tensor<1x3x3x4xf32>
  func.return %2: tensor<1x3x3x4xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_fn_1

// -----

// CHECK-LABEL: @conv_with_relu6_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<3x3x4x4xf32>)
func.func @conv_with_relu6_fn(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x3x4xf32>
  %1 = stablehlo.constant dense<6.000000e+00> : tensor<1x3x3x4xf32>
  %2 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %3 = stablehlo.clamp %0, %2, %1 : tensor<1x3x3x4xf32>
  func.return %3: tensor<1x3x3x4xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_relu6_fn_1
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[CONV]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_relu6_fn(
// CHECK-SAME:                 %[[ARG_0:.*]]: tensor<1x1x167xf32>,
// CHECK-SAME:                 %[[ARG_1:.*]]: tensor<167x64xf32>
func.func @dot_general_with_relu6_fn(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %1 = stablehlo.constant dense<6.000000e+00> : tensor<1x1x64xf32>
  %2 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %3 = stablehlo.clamp %0, %2, %1 : tensor<1x1x64xf32>
  return %3 : tensor<1x1x64xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_relu6_fn_1
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[DOT_GENERAL]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_and_relu_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<3x3x4x4xf32>,
// CHECK-SAME:                    %[[ARG_2:.*]]: tensor<1x3x3x4xf32>)
func.func @conv_with_bias_and_relu_fn(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<3x3x4x4xf32>, %arg2: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x3x4xf32>
  %1 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %2 = stablehlo.add %1, %arg2 : tensor<1x3x3x4xf32>
  %3 = stablehlo.maximum %2, %0 : tensor<1x3x3x4xf32>
  func.return %3: tensor<1x3x3x4xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1, %arg2)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_and_relu_fn_1
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %arg2
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[CONST]]
// CHECK: return %[[MAX]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_and_relu_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x1x167xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<167x64xf32>
// CHECK-SAME:                    %[[ARG_2:.*]]: tensor<1x1x64xf32>)
func.func @dot_general_with_bias_and_relu_fn(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>, %arg2: tensor<1x1x64xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %1 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %2 = stablehlo.add %1, %arg2 : tensor<1x1x64xf32>
  %3 = stablehlo.maximum %2, %0 : tensor<1x1x64xf32>
  func.return %3: tensor<1x1x64xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1, %arg2)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_and_relu_fn_1
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg2
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[CONST]]
// CHECK: return %[[MAX]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_and_relu6_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<3x3x4x4xf32>,
// CHECK-SAME:                    %[[ARG_2:.*]]: tensor<1x3x3x4xf32>)
func.func @conv_with_bias_and_relu6_fn(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<3x3x4x4xf32>, %arg2: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x3x4xf32>
  %1 = stablehlo.constant dense<6.000000e+00> : tensor<1x3x3x4xf32>
  %2 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %3 = stablehlo.add %2, %arg2 : tensor<1x3x3x4xf32>
  %4 = stablehlo.clamp %0, %3, %1 : tensor<1x3x3x4xf32>
  func.return %4: tensor<1x3x3x4xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1, %arg2)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_and_relu6_fn_1
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %arg2
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[ADD]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_and_relu6_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x1x167xf32>,
// CHECK-SAME:                    %[[ARG_1:.*]]: tensor<167x64xf32>
// CHECK-SAME:                    %[[ARG_2:.*]]: tensor<1x1x64xf32>)
func.func @dot_general_with_bias_and_relu6_fn(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>, %arg2: tensor<1x1x64xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %1 = stablehlo.constant dense<6.000000e+00> : tensor<1x1x64xf32>
  %2 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %3 = stablehlo.add %2, %arg2 : tensor<1x1x64xf32>
  %4 = stablehlo.clamp %0, %3, %1 : tensor<1x1x64xf32>
  func.return %4: tensor<1x1x64xf32>
}
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %arg1, %arg2)
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_and_relu6_fn_1
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg2
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[ADD]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x1x64xf32>
// CHECK: }
