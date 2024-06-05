// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-quantize | FileCheck %s

// Test that hybrid quantized dot_general is produced when q/dq pair only exists
// for weight.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_dot_general_fn(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = stablehlo.constant dense<3.000000e-01> : tensor<2x3xf32>
    %0 = "quantfork.qcast"(%cst) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 6.000000e-03>>
    %1 = "quantfork.dcast"(%0) : (tensor<2x3x!quant.uniform<i8:f32, 6.000000e-03>>) -> tensor<2x3xf32>
    %2 = "tf.XlaCallModule"(%arg0, %1) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "weight_only_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }

  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// CHECK-LABEL: quantize_dot_general_fn
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2xf32>
// CHECK: %[[CST:.+]] = stablehlo.constant dense<3.000000e-01> : tensor<2x3xf32>
// CHECK: %[[Q:.+]] = "quantfork.qcast"(%[[CST]]) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 6.000000e-03>>
// CHECK: %[[CALL:.+]] = call @quantized_dot_general_fn(%[[ARG0]], %[[Q]])
// CHECK-SAME: {_quantization_method = "weight_only_ptq { }"} : (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8:f32, 6.000000e-03>>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]]

// CHECK: quantized_dot_general_fn
// CHECK-SAME: (%[[ARG1:.+]]: tensor<1x2xf32>,  %[[ARG2:.+]]: tensor<2x3x!quant.uniform<i8:f32, 6.000000e-03>>) -> tensor<1x3xf32>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[ARG1]], %[[ARG2]]
// CHECK-SAME: (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8:f32, 6.000000e-03>>) -> tensor<1x3xf32>
// CHECK: return %[[DOT]]

// -----

// Test that hybrid quantized convolution is produced when q/dq pair only exists
// for weight.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_fn(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = stablehlo.constant dense<3.000000e-01> : tensor<2x3x3x2xf32>
    %0 = "quantfork.qcast"(%cst) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    %1 = "quantfork.dcast"(%0) : (tensor<2x3x3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<2x3x3x2xf32>
    %2 = "tf.XlaCallModule"(%arg0, %1) <{Sout = [#tf_type.shape<1x3x4x2>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn", _quantization_method = "weight_only_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %2 : tensor<1x3x4x2xf32>
  }

  func.func private @composite_conv_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }
}

// CHECK-LABEL: quantize_conv_fn
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x3x4x3xf32>
// CHECK: %[[CST:.+]] = stablehlo.constant dense<3.000000e-01> : tensor<2x3x3x2xf32>
// CHECK: %[[Q:.+]] = "quantfork.qcast"(%[[CST]]) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>
// CHECK: %[[CALL:.+]] = call @quantized_conv_fn(%[[ARG0]], %[[Q]]) {_quantization_method = "weight_only_ptq { }"} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[CALL]]

// CHECK: quantized_conv_fn
// CHECK-SAME: (%[[ARG1:.+]]: tensor<1x3x4x3xf32>,  %[[ARG2:.+]]: tensor<2x3x3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<1x3x4x2xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[ARG1]], %[[ARG2]])
// CHECK-SAME: (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[CONV]]
