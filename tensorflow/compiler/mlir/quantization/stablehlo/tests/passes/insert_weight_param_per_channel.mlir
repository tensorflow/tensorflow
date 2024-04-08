// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-insert-weight-param -verify-diagnostics | FileCheck %s

// Test that per-channel q/dq pair is inserted between constant and XlaCallModule op
// with quantizable trait and function name containing conv.

module attributes {tf_saved_model.semantics} {
  func.func private @qdq_for_convolution_per_channel(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "tf.XlaCallModule"(%arg0, %cst) {
        Sout = [#tf_type.shape<1x3x4x2>], dim_args_spec = [], disabled_checks = [],
        has_token_input_output = false, module = "", platforms = [], version = 5 : i64,
        _entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn",
        _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
        device = ""
      } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }
  // CHECK: func.func private @qdq_for_convolution_per_channel(%[[ARG0:.+]]: tensor<1x3x4x3xf32>)
  // CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
  // CHECK: %[[Q:.+]] = "quantfork.qcast"(%[[CST]]) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8:f32:3, {0.0011764706349840352:-128,0.0011764706349840352:-128}>>
  // CHECK: %[[DQ:.+]] = "quantfork.dcast"(%[[Q]]) : (tensor<2x3x3x2x!quant.uniform<i8:f32:3, {0.0011764706349840352:-128,0.0011764706349840352:-128}>>) -> tensor<2x3x3x2xf32>
  // CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG0]], %[[DQ]])
  // CHECK-SAME: (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
  // CHECK: return %[[CALL]]

  func.func private @composite_conv_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }
  // CHECK: func private @composite_conv_fn
  // CHECK: %[[CONV:.+]] = stablehlo.convolution
  // CHECK: return %[[CONV]]
}

// -----

// Test that per-channel q/dq pair is inserted between constant and XlaCallModule op
// with quantizable trait and function name containing dot_general.

module attributes {tf_saved_model.semantics} {
  func.func private @qdq_for_dot_general_per_channel(%arg0: tensor<4x3x6x5xf32>) -> tensor<4x3x6x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<4x3x5x2xf32>} : () -> tensor<4x3x5x2xf32>
    %0 = "tf.XlaCallModule"(%arg0, %cst) {
      Sout = [#tf_type.shape<4x3x6x2>], _entry_function = @composite_dot_general_fn,
      _original_entry_function = "composite_dot_general_fn",
      _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
      device = "", dim_args_spec = [], disabled_checks = [],
      has_token_input_output = false, module = "", platforms = [],
      version = 5 : i64
    } : (tensor<4x3x6x5xf32>, tensor<4x3x5x2xf32>) -> tensor<4x3x6x2xf32>
    return %0 : tensor<4x3x6x2xf32>
  }
  // CHECK: func.func private @qdq_for_dot_general_per_channel(%[[ARG0:.+]]: tensor<4x3x6x5xf32>)
  // CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<4x3x5x2xf32>}> : () -> tensor<4x3x5x2xf32>
  // CHECK: %[[Q:.+]] = "quantfork.qcast"(%[[CST]]) : (tensor<4x3x5x2xf32>) -> tensor<4x3x5x2x!quant.uniform<i8:f32:3, {0.0011764706349840352:-128,0.0011764706349840352:-128}>>
  // CHECK: %[[DQ:.+]] = "quantfork.dcast"(%[[Q]]) : (tensor<4x3x5x2x!quant.uniform<i8:f32:3, {0.0011764706349840352:-128,0.0011764706349840352:-128}>>) -> tensor<4x3x5x2xf32>
  // CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG0]], %[[DQ]])
  // CHECK-SAME: (tensor<4x3x6x5xf32>, tensor<4x3x5x2xf32>) -> tensor<4x3x6x2xf32>
  // CHECK: return %[[CALL]]

  func.func private @composite_dot_general_fn(%arg0: tensor<4x3x6x5xf32>, %arg1: tensor<4x3x5x2xf32>) -> tensor<4x3x6x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<4x3x6x5xf32>, tensor<4x3x5x2xf32>) -> tensor<4x3x6x2xf32>
    return %0 : tensor<4x3x6x2xf32>
  }
  // CHECK: func private @composite_dot_general_fn
  // CHECK: %[[DOT:.+]] = stablehlo.dot_general
  // CHECK: return %[[DOT]]
}

// -----

// Test that error is raised when rhs of dot_general does not have
// non-contracting, non-batching dimensions.

module attributes {tf_saved_model.semantics} {
  func.func private @dot_general_per_channel_error(%arg0: tensor<4x3x6x5xf32>) -> tensor<4x3x6xf32> attributes {tf._original_func_name = "main_0"} {
    // expected-error @+1 {{Failed to get quantization dimension for weight.}}
    %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<4x3x5xf32>} : () -> tensor<4x3x5xf32>
    // expected-error @+1 {{dot_general op does not have non-contracting, non-batching dimension.}}
    %0 = "tf.XlaCallModule"(%arg0, %cst) {
      Sout = [#tf_type.shape<4x3x6>], _entry_function = @composite_dot_general_fn,
      _original_entry_function = "composite_dot_general_fn",
      _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
      device = "", dim_args_spec = [], disabled_checks = [],
      has_token_input_output = false, module = "", platforms = [],
      version = 5 : i64
    } : (tensor<4x3x6x5xf32>, tensor<4x3x5xf32>) -> tensor<4x3x6xf32>
    return %0 : tensor<4x3x6xf32>
  }

  func.func private @composite_dot_general_fn(%arg0: tensor<4x3x6x5xf32>, %arg1: tensor<4x3x5xf32>) -> tensor<4x3x6xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<4x3x6x5xf32>, tensor<4x3x5xf32>) -> tensor<4x3x6xf32>
    return %0 : tensor<4x3x6xf32>
  }
}
