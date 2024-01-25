// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics -stablehlo-test-pre-calibration-component | FileCheck %s

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
