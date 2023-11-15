// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-quantize -verify-each=false | FileCheck %s

// CHECK-LABEL: same_scale_after_composite
func.func @same_scale_after_composite() -> tensor<3x1xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[RESHAPE:.*]] = "stablehlo.reshape"(%[[CALL]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[RESHAPE]]) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  // CHECK: "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3x1xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  return %5 : tensor<3x1xf32>
}

// -----

// CHECK-LABEL: same_scale_indirectly_connected
func.func @same_scale_indirectly_connected() -> tensor<1x3xf32> {
  // CHECK: %[[CALL:.*]] = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-SAME: () -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[RESHAPE:.*]] = "stablehlo.reshape"(%[[CALL]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[TRANSPOSE:.*]] = "stablehlo.transpose"(%[[RESHAPE]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ:.*]] = "quantfork.dcast"(%[[TRANSPOSE]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  // CHECK: "func.return"(%[[DQ]])

  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3x1xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  %6 = "stablehlo.transpose"(%5) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<3x1xf32>) -> tensor<1x3xf32>
  %7 = "quantfork.qcast"(%6) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %8 = "quantfork.dcast"(%7) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  return %8 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: same_scale_not_connected_to_composite
func.func @same_scale_not_connected_to_composite() -> tensor<3x1xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  // CHECK: %[[Q1:.*]] = "quantfork.qcast"(%[[CST]]) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ1:.*]] = "quantfork.dcast"(%[[Q1]]) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[DQ1]]
  // CHECK: %[[Q2:.*]] = "quantfork.qcast"(%[[RESHAPE]]) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  // CHECK: %[[DQ2:.*]] = "quantfork.dcast"(%[[Q2]]) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  // CHECK: return %[[DQ2]]

  %0 = stablehlo.constant dense<1.000000e+00> : tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<1x3xf32>
  %3 = stablehlo.reshape %2 : (tensor<1x3xf32>) -> tensor<3x1xf32>
  %4 = "quantfork.qcast"(%3) {volatile} : (tensor<3x1xf32>) -> tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>
  %5 = "quantfork.dcast"(%4) : (tensor<3x1x!quant.uniform<i8:f32, 0.13170163023705575:-1>>) -> tensor<3x1xf32>
  return %5 : tensor<3x1xf32>
}
