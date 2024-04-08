// RUN: stablehlo-quant-opt %s -split-input-file \
// RUN:     -stablehlo-quantize-composite-functions="enable-weight-only=true enable-per-channel-quantized-weight=false" \
// RUN:     | FileCheck --check-prefix=CHECK-PER-TENSOR %s

// Test that weight-only quantized dot_general op is produced when
// enable-weight-only is set to true.

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

// CHECK-PER-CHANNEL: quantize_dot_general_fn(%[[ARG0:.+]]: tensor<1x2xf32>)
// CHECK-PER-CHANNEL: %[[CST:.+]] = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8:f32:1, {0.0011764706349840352:-128,0.0011764706349840352:-128,0.0011764706349840352:-128}>>
// CHECK-PER-CHANNEL: %[[CALL:.+]] = call @quantized_dot_general_fn(%[[ARG0]], %[[CST]]) : (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8:f32:1, {0.0011764706349840352:-128,0.0011764706349840352:-128,0.0011764706349840352:-128}>>) -> tensor<1x3xf32>
// CHECK-PER-CHANNEL: return %[[CALL]]

// CHECK-PER-CHANNEL: quantized_dot_general_fn
// CHECK-PER-CHANNEL-SAME: (%[[ARG1:.+]]: tensor<1x2xf32>,  %[[ARG2:.+]]: tensor<2x3x!quant.uniform<i8:f32:1, {0.0011764706349840352:-128,0.0011764706349840352:-128,0.0011764706349840352:-128}>>) -> tensor<1x3xf32>
// CHECK-PER-CHANNEL: %[[DOT:.+]] = stablehlo.dot_general %[[ARG1]], %[[ARG2]]
// CHECK-PER-CHANNEL-SAME: (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8:f32:1, {0.0011764706349840352:-128,0.0011764706349840352:-128,0.0011764706349840352:-128}>>) -> tensor<1x3xf32>
// CHECK-PER-CHANNEL: return %[[DOT]]

// CHECK-PER-TENSOR: quantize_dot_general_fn(%[[ARG0:.+]]: tensor<1x2xf32>)
// CHECK-PER-TENSOR: %[[CST:.+]] = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>
// CHECK-PER-TENSOR: %[[CALL:.+]] = call @quantized_dot_general_fn(%[[ARG0]], %[[CST]]) : (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3xf32>
// CHECK-PER-TENSOR: return %[[CALL]]

// CHECK-PER-TENSOR quantized_dot_general_fn
// CHECK-PER-TENSOR: (%[[ARG1:.+]]: tensor<1x2xf32>,  %[[ARG2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3xf32>
// CHECK-PER-TENSOR: %[[DOT:.+]] = stablehlo.dot_general %[[ARG1]], %[[ARG2]]
// CHECK-PER-TENSOR-SAME: (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3xf32>
// CHECK-PER-TENSOR: return %[[DOT]]

// -----

// Test that hybrid quantized convolution op is produced when enable-weight-only
// is set to true.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_fn(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %0 = stablehlo.constant dense<3.000000e-01> : tensor<2x3x3x2xf32>
    %1 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3x4x2>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %1 : tensor<1x3x4x2xf32>
  }

  func.func private @composite_conv_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }
}

// CHECK-PER-CHANNEL-LABEL: quantize_conv_fn
// CHECK-PER-CHANNEL-SAME: %[[ARG0:.+]]: tensor<1x3x4x3xf32>
// CHECK-PER-CHANNEL: %[[CST:.+]] = stablehlo.constant() {value = dense<127> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2x!quant.uniform<i8:f32:3, {0.0011764706349840352:-128,0.0011764706349840352:-128}>>
// CHECK-PER-CHANNEL: %[[CALL:.+]] = call @quantized_conv_fn(%[[ARG0]], %[[CST]]) : (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8:f32:3, {0.0011764706349840352:-128,0.0011764706349840352:-128}>>) -> tensor<1x3x4x2xf32>
// CHECK-PER-CHANNEL: return %[[CALL]]

// CHECK-PER-CHANNEL: quantized_conv_fn
// CHECK-PER-CHANNEL-SAME: (%[[ARG1:.+]]: tensor<1x3x4x3xf32>,  %[[ARG2:.+]]: tensor<2x3x3x2x!quant.uniform<i8:f32:3, {0.0011764706349840352:-128,0.0011764706349840352:-128}>>) -> tensor<1x3x4x2xf32>
// CHECK-PER-CHANNEL: %[[CONV:.+]] = stablehlo.convolution(%[[ARG1]], %[[ARG2]])
// CHECK-PER-CHANNEL-SAME: (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8:f32:3, {0.0011764706349840352:-128,0.0011764706349840352:-128}>>) -> tensor<1x3x4x2xf32>
// CHECK-PER-CHANNEL: return %[[CONV]]

// CHECK-PER-TENSOR-LABEL: quantize_conv_fn
// CHECK-PER-TENSOR-SAME: %[[ARG0:.+]]: tensor<1x3x4x3xf32>
// CHECK-PER-TENSOR: %[[CST:.+]] = stablehlo.constant() {value = dense<127> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>
// CHECK-PER-TENSOR: %[[CALL:.+]] = call @quantized_conv_fn(%[[ARG0]], %[[CST]]) : (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3x4x2xf32>
// CHECK-PER-TENSOR: return %[[CALL]]

// CHECK-PER-TENSOR: quantized_conv_fn
// CHECK-PER-TENSOR-SAME: (%[[ARG1:.+]]: tensor<1x3x4x3xf32>,  %[[ARG2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3x4x2xf32>
// CHECK-PER-TENSOR: %[[CONV:.+]] = stablehlo.convolution(%[[ARG1]], %[[ARG2]])
// CHECK-PER-TENSOR-SAME: (tensor<1x3x4x3xf32>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x3x4x2xf32>
// CHECK-PER-TENSOR: return %[[CONV]]
