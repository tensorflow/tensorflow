// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-quantize -verify-each=false | FileCheck %s

// CHECK-LABEL: quantize_simple_xla_call_module
func.func private @quantize_simple_xla_call_module(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<4x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %2 = "quantfork.dcast"(%1) : (tensor<4x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<4x3xf32>
  %3 = "quantfork.qcast"(%arg0) {volatile} : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
  %4 = "quantfork.dcast"(%3) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<1x4xf32>
  %5 = "tf.XlaCallModule"(%4, %2) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
  %6 = "quantfork.qcast"(%5) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %7 = "quantfork.dcast"(%6) : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
  return %7 : tensor<1x3xf32>
}

// Tests that the inputs and output of the tf.XlaCallModule op has been replaced
// by quantized types, and the corresponding quantfork.dcast ops that turned
// those quantized types back to float types are removed.
// CHECK: ^bb0(%[[ARG_0:.*]]: tensor<1x4xf32>):
// CHECK: %[[CONST_0:.*]] = "stablehlo.constant"() {value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> tensor<4x3xf32>
// CHECK-DAG: %[[QCAST_0:.*]] = "quantfork.qcast"(%[[CONST_0]]) {volatile} : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
// CHECK-DAG: %[[QCAST_1:.*]] = "quantfork.qcast"(%[[ARG_0]]) {volatile} : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
// CHECK: %[[XLACALLMODULE_0:.*]] = "tf.XlaCallModule"(%[[QCAST_1]], %[[QCAST_0]]) {{{.*}}} : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
// CHECK: %[[DCAST_0:.*]] = "quantfork.dcast"(%[[XLACALLMODULE_0]]) : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
// CHECK: "func.return"(%[[DCAST_0]]) : (tensor<1x3xf32>) -> ()

// -----

// CHECK-LABEL: quantize_simple_xla_call_module_no_operand
func.func private @quantize_simple_xla_call_module_no_operand() -> tensor<1x3xf32> {
  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
  return %2 : tensor<1x3xf32>
}

// Tests that the output of the tf.XlaCallModule op has been replaced by
// a quantized type, and the corresponding quantfork.qcast ops that turned
// the float output to a quantized type is removed.
// CHECK: %[[XLACALLMODULE_0:.*]] = "tf.XlaCallModule"() {{{.*}}} : () -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
// CHECK: %[[DCAST_0:.*]] = "quantfork.dcast"(%[[XLACALLMODULE_0]]) : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
// CHECK: "func.return"(%[[DCAST_0]]) : (tensor<1x3xf32>) -> ()
