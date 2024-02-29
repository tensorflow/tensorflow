// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:   -stablehlo-test-pre-calibration-component | FileCheck %s

func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
  %0 = stablehlo.constant dense<1.0> : tensor<4x3xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}
// CHECK: @main(%[[ARG_0:.+]]: tensor<1x4xf32>) -> tensor<1x3xf32>
// CHECK-DAG: %[[XLA_CALL_MODULE_0:.+]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<4x3>], {{.*}}, module = "{{.+}}", platforms = ["CPU", "TPU"], version = 9 : i64}> : () -> tensor<4x3xf32>
// CHECK: %[[CUSTOM_AGGREGATOR_0:.+]] = "tf.CustomAggregator"(%[[ARG_0]]) <{id = "0"}> {calibration_method = 0 : i32, {{.*}}} : (tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK: %[[XLA_CALL_MODULE_1:.+]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_0]], %[[XLA_CALL_MODULE_0]]) <{Sout = [#tf_type.shape<1x3>], {{.*}}, module = "{{.+}}", platforms = ["CPU"], version = 9 : i64}> {_original_entry_function = "composite_dot_general_fn_1", _tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
// CHECK: %[[CUSTOM_AGGREGATOR_1:.+]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_1]]) <{id = "1"}> {calibration_method = 0 : i32, {{.*}}} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[CUSTOM_AGGREGATOR_1]] : tensor<1x3xf32>
// CHECK: }
// CHECK: }

// -----

// Tests that stablehlo op serialization also works for the "serving_default"
// function.

func.func @serving_default(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
  %0 = stablehlo.constant dense<1.0> : tensor<4x3xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}
// CHECK: @serving_default(%[[ARG_0:.+]]: tensor<1x4xf32>) -> tensor<1x3xf32>
// CHECK-DAG: %[[XLA_CALL_MODULE_0:.+]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<4x3>], {{.*}}, module = "{{.+}}", platforms = ["CPU", "TPU"], version = 9 : i64}> : () -> tensor<4x3xf32>
// CHECK: %[[CUSTOM_AGGREGATOR_0:.+]] = "tf.CustomAggregator"(%[[ARG_0]]) <{id = "0"}> {calibration_method = 0 : i32, {{.*}}} : (tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK: %[[XLA_CALL_MODULE_1:.+]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_0]], %[[XLA_CALL_MODULE_0]]) <{Sout = [#tf_type.shape<1x3>], {{.*}}, module = "{{.+}}", platforms = ["CPU"], version = 9 : i64}> {_original_entry_function = "composite_dot_general_fn_1", _tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
// CHECK: %[[CUSTOM_AGGREGATOR_1:.+]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_1]]) <{id = "1"}> {calibration_method = 0 : i32, {{.*}}} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[CUSTOM_AGGREGATOR_1]] : tensor<1x3xf32>
// CHECK: }
// CHECK: }

// -----

// Tests that `stablehlo.convolution` with NCHW format is converted to NHWC.

func.func @main(%arg0: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
  %0 = stablehlo.constant() {value = dense<3.000000e+00> : tensor<8x8x3x3xf32>} : () -> tensor<8x8x3x3xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x8x4x4xf32>, tensor<8x8x3x3xf32>) -> tensor<1x8x4x4xf32>
  return %2 : tensor<1x8x4x4xf32>
}
// CHECK: @main(%[[ARG:.+]]: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>

// Contains the `stablehlo.transpose` op of the arg (e.g. [b, f, 0, 1] to
// [b, 0, 1, f]).
// CHECK: %[[XLA_CALL_MODULE_0:.+]]:2 = "tf.XlaCallModule"(%[[ARG]]) {{.*}} : (tensor<1x8x4x4xf32>) -> (tensor<3x3x8x8xf32>, tensor<1x4x4x8xf32>)
// CHECK-DAG: %[[CUSTOM_AGGREGATOR_1:.+]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_0]]#1) {{.*}} : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
// CHECK-DAG: %[[CUSTOM_AGGREGATOR_0:.+]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_0]]#0) {{.*}} : (tensor<3x3x8x8xf32>) -> tensor<3x3x8x8xf32>

// Corresponds to the converted `stablehlo.convolution`. Note that the shapes
// correspond to the dimension numbers of: [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK: %[[XLA_CALL_MODULE_1:.+]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_1]], %[[CUSTOM_AGGREGATOR_0]]) {{.*}} : (tensor<1x4x4x8xf32>, tensor<3x3x8x8xf32>) -> tensor<1x4x4x8xf32>
// CHECK: %[[CUSTOM_AGGREGATOR_2:.+]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_1]]) {{.*}} : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>

// Corresponds to the `stablehlo.transpose` op that reverts the output tensor
// shape: [b, 0, 1, f] to [b, f, 0, 1].
// CHECK: %[[XLA_CALL_MODULE_2:.+]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_2]]) {{.*}} : (tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32>
// CHECK: return %[[XLA_CALL_MODULE_2]] : tensor<1x8x4x4xf32>
