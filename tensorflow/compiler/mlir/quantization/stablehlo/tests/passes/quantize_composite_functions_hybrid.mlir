// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:     -stablehlo-quantize-composite-functions=enable-weight-only=true | FileCheck --check-prefix=CHECK %s

// Test that hybrid quantized dot_general op is produced when hybrid-quantize
// is set to true.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_dot_general_fn(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %0 = stablehlo.constant dense<3.000000e-01> : tensor<2x3xf32>
    %1 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }

  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// CHECK-LABEL: quantize_dot_general_fn
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2xf32>
// CHECK: %[[CST:.+]] = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8:f32, 0.0011764706349840352:-128>>
// CHECK: %[[CALL:.+]] = call @quantized_dot_general_fn(%[[ARG0]], %[[CST]]) : (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8:f32, 0.0011764706349840352:-128>>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]]

// CHECK: quantized_dot_general_fn
// CHECK-SAME: (%[[ARG1:.+]]: tensor<1x2xf32>,  %[[ARG2:.+]]: tensor<2x3x!quant.uniform<i8:f32, 0.0011764706349840352:-128>>) -> tensor<1x3xf32>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[ARG1]], %[[ARG2]]
// CHECK-SAME: (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8:f32, 0.0011764706349840352:-128>>) -> tensor<1x3xf32>
// CHECK: return %[[DOT]]
