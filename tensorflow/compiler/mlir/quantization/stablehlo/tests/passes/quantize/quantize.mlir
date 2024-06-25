// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-quantize -verify-each=false | FileCheck %s

// Tests for PopulateFusedGemmStylePatterns are handled in
// quantize_composite_functions for module-level evaluation of functions.

module attributes {tf_saved_model.semantics} {
// CHECK: quantize_simple_xla_call_module(%[[ARG_0:.+]]: tensor<1x4xf32>)
  func.func private @quantize_simple_xla_call_module(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<4x3xf32>
    %1 = "quantfork.qcast"(%0) {volatile} : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03, 5.000000e-03, 5.000000e-03}>>
    %2 = "quantfork.dcast"(%1) : (tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03, 5.000000e-03, 5.000000e-03}>>) -> tensor<4x3xf32>
    %3 = "quantfork.qcast"(%arg0) {volatile} : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    %4 = "quantfork.dcast"(%3) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<1x4xf32>
    %5 = "tf.XlaCallModule"(%4, %2) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %6 = "quantfork.qcast"(%5) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %7 = "quantfork.dcast"(%6) : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
    return %7 : tensor<1x3xf32>
  }
// Test that the inputs and output of the tf.XlaCallModule op has been replaced
// by quantized types, and the corresponding quantfork.dcast ops that turned
// those quantized types back to float types are removed.
// CHECK: %[[CONST_0:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<4x3xf32>
// CHECK-DAG: %[[QCAST_0:.+]] = "quantfork.qcast"(%[[CONST_0]]) {volatile} : (tensor<4x3xf32>) -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
// CHECK-DAG: %[[QCAST_1:.+]] = "quantfork.qcast"(%[[ARG_0]]) {volatile} : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
// CHECK: %[[CALL_0:.+]] = call @quantized_dot_general_fn(%[[QCAST_1]], %[[QCAST_0]])
// Test that the `Method` has been copied over.
// CHECK-SAME: {_quantization_method = "static_range_ptq { }"}
// CHECK-SAME: : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
// CHECK: %[[DCAST_0:.+]] = "quantfork.dcast"(%[[CALL_0]]) :  (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
// CHECK: return

  func.func private @composite_dot_general_fn(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
      return %0 : tensor<1x3xf32>
  }
}

// -----

// Tests that the output of the tf.XlaCallModule op has been replaced by
// a quantized type, and the corresponding quantfork.qcast ops that turned
// the float output to a quantized type is removed.

// CHECK-LABEL: quantize_simple_xla_call_module_no_operand
func.func private @quantize_simple_xla_call_module_no_operand() -> tensor<1x3xf32> {
  %0 = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : () -> tensor<1x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %2 = "quantfork.dcast"(%1) : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
  return %2 : tensor<1x3xf32>
}
// CHECK: %[[XLA_CALL_MODULE_0:.+]] = "tf.XlaCallModule"() <{{{.*}}}> {{{.*}}} : () -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
// CHECK: %[[DCAST_0:.+]] = "quantfork.dcast"(%[[XLA_CALL_MODULE_0]]) : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
// CHECK: "func.return"(%[[DCAST_0]]) : (tensor<1x3xf32>) -> ()

// -----

// Tests for emitting an error when there is no corresponding entry
// function to quantize (@composite_dot_general_fn).

module attributes {tf_saved_model.semantics} {
 func.func private @error_when_no_entry_function(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
   %0 = stablehlo.constant dense<1.000000e+00> : tensor<2x3xf32>
   %1 = "quantfork.qcast"(%0) {volatile} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
   %2 = "quantfork.dcast"(%1) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<2x3xf32>
   %3 = "quantfork.qcast"(%arg0) {volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>
   %4 = "quantfork.dcast"(%3) : (tensor<1x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<1x2xf32>
// expected-error @+2 {{Failed to find a valid entry function}}
// expected-error @+1 {{'tf.XlaCallModule' op operand #0 must be variadic of tensor of tf.dtype values}}
   %5 = "tf.XlaCallModule"(%4, %2) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
   %6 = "quantfork.qcast"(%5) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
   %7 = "quantfork.dcast"(%6) : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
   return %7 : tensor<1x3xf32>
 }
}
