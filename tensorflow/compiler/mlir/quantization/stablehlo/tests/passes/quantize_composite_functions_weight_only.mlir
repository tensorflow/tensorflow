// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:     -stablehlo-quantize-composite-functions | FileCheck --check-prefix=CHECK %s

// Test that per-tensor weight-only quantized dot_general op is produced when
// empty `weight_only_ptq` is provided.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_dot_general_per_tensor(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %0 = stablehlo.constant dense<3.000000e-01> : tensor<2x3xf32>
    %1 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "weight_only_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }

  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// CHECK-LABEL: quantize_dot_general_per_tensor
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2xf32>
// CHECK: %[[CST:.+]] = stablehlo.constant() <{value = dense<127> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>
// CHECK: %[[CALL:.+]] = call @quantized_dot_general_fn(%[[ARG0]], %[[CST]]) {_quantization_method = "weight_only_ptq { }"} : (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]]

// CHECK: quantized_dot_general_fn
// CHECK-SAME: (%[[ARG1:.+]]: tensor<1x2xf32>,  %[[ARG2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3xf32>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[ARG1]], %[[ARG2]]
// CHECK-SAME: (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3xf32>
// CHECK: return %[[DOT]]

// -----

// Test that per-tensor weight-only quantized convolution op is produced when
// empty `weight_only_ptq` is provided.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_per_tensor(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %0 = stablehlo.constant dense<3.000000e-01> : tensor<2x3x3x2xf32>
    %1 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3x4x2>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn", _quantization_method = "weight_only_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %1 : tensor<1x3x4x2xf32>
  }

  func.func private @composite_conv_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }
}

// CHECK-LABEL: quantize_conv_per_tensor
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x3x4x3xf32>
// CHECK: %[[CST:.+]] = stablehlo.constant() <{value = dense<127> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>
// CHECK: %[[CALL:.+]] = call @quantized_conv_fn(%[[ARG0]], %[[CST]]) {_quantization_method = "weight_only_ptq { }"} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[CALL]]

// CHECK: quantized_conv_fn
// CHECK-SAME: (%[[ARG1:.+]]: tensor<1x3x4x3xf32>,  %[[ARG2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3x4x2xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[ARG1]], %[[ARG2]])
// CHECK-SAME: (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[CONV]]

// -----

// Test that per-channel weight-only quantized dot_general op is produced when
// `weight_only_ptq` with `dimension_specs` is provided.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_dot_general_per_channel(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %0 = stablehlo.constant dense<3.000000e-01> : tensor<2x3xf32>
    %1 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {dimension_specs {}}}}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }

  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// CHECK-LABEL: quantize_dot_general_per_channel
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2xf32>
// CHECK: %[[CST:.+]] = stablehlo.constant() <{value = dense<127> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {0.0023622048182750312,0.0023622048182750312,0.0023622048182750312}>>
// CHECK: %[[CALL:.+]] = call @quantized_dot_general_fn(%[[ARG0]], %[[CST]]) {_quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {dimension_specs {}}}}"}
// CHECK-SAME: (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {0.0023622048182750312,0.0023622048182750312,0.0023622048182750312}>>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]]

// CHECK: quantized_dot_general_fn
// CHECK-SAME: (%[[ARG1:.+]]: tensor<1x2xf32>,  %[[ARG2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {0.0023622048182750312,0.0023622048182750312,0.0023622048182750312}>>) -> tensor<1x3xf32>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[ARG1]], %[[ARG2]]
// CHECK-SAME: (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {0.0023622048182750312,0.0023622048182750312,0.0023622048182750312}>>) -> tensor<1x3xf32>
// CHECK: return %[[DOT]]

// -----

// Test that per-channel weight-only quantized convolution op is produced when
// `weight_only_ptq` with `dimension_specs` is provided.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_per_channel(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %0 = stablehlo.constant dense<3.000000e-01> : tensor<2x3x3x2xf32>
    %1 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3x4x2>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn", _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {dimension_specs {}}}}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %1 : tensor<1x3x4x2xf32>
  }

  func.func private @composite_conv_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }
}

// CHECK-LABEL: quantize_conv_per_channel
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x3x4x3xf32>
// CHECK: %[[CST:.+]] = stablehlo.constant() <{value = dense<127> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>
// CHECK: %[[CALL:.+]] = call @quantized_conv_fn(%[[ARG0]], %[[CST]]) {_quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {dimension_specs {}}}}"}
// CHECK-SAME: (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[CALL]]

// CHECK: quantized_conv_fn
// CHECK-SAME: (%[[ARG1:.+]]: tensor<1x3x4x3xf32>,  %[[ARG2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<1x3x4x2xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[ARG1]], %[[ARG2]])
// CHECK-SAME: (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[CONV]]
