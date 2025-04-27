// RUN: stablehlo-quant-opt %s -split-input-file -tf-stablehlo-insert-weight-param | FileCheck %s

// Test that q/dq pair with per-tensor quantization parameter is inserted
// between constant and XlaCallModule op with empty `weight_only_ptq` method
// and function name containing conv.

func.func @qdq_for_conv_weight_empty(%arg0: tensor<1x3x2x3xf32>) -> tensor<1x2x2x2xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x2x2x2>], _entry_function = @composite_conv_fn,
    _original_entry_function = "composite_conv_fn",
    _stablehlo_module_attrs = {}, _quantization_method = "weight_only_ptq { }",
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @qdq_for_conv_weight_empty
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x3x2x3xf32>) -> tensor<1x2x2x2xf32>
// CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
// CHECK: %[[Q:.+]] = "quantization.qcast"(%[[CST]]) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>
// CHECK: %[[DQ:.+]] = "quantization.dcast"(%[[Q]]) : (tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<2x3x3x2xf32>
// CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG_0]], %[[DQ]])
// CHECK-SAME: _entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn", _quantization_method = "weight_only_ptq { }"
// CHECK-SAME: (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
// CHECK: return %[[CALL]] : tensor<1x2x2x2xf32>

// -----

// Test that q/dq pair with per-tensor quantization parameter is inserted
// between constant and XlaCallModule op with empty `weight_only_ptq` method and
// function name containing dot_general.

func.func @qdq_for_dot_general_weight_empty(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn,
    _original_entry_function = "composite_dot_general_fn",
    _quantization_method = "weight_only_ptq { }", _stablehlo_module_attrs = {},
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: func.func @qdq_for_dot_general_weight_empty
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32>
// CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
// CHECK: %[[Q:.+]] = "quantization.qcast"(%[[CST]]) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>
// CHECK: %[[DQ:.+]] = "quantization.dcast"(%[[Q]]) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<2x3xf32>
// CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG_0]], %[[DQ]])
// CHECK-SAME: _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "weight_only_ptq { }"
// CHECK-SAME: (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]] : tensor<1x3xf32>

// -----

// Test that q/dq pair with per-tensor quantization parameter is inserted
// between constant and XlaCallModule op with `weight_only_ptq` method of
// `per_tensor` and function name containing conv.

func.func @qdq_for_conv_weight_per_tensor(%arg0: tensor<1x3x2x3xf32>) -> tensor<1x2x2x2xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x2x2x2>], _entry_function = @composite_conv_fn,
    _original_entry_function = "composite_conv_fn",
    _stablehlo_module_attrs = {}, _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {per_tensor {}}}}",
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @qdq_for_conv_weight_per_tensor
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x3x2x3xf32>) -> tensor<1x2x2x2xf32>
// CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
// CHECK: %[[Q:.+]] = "quantization.qcast"(%[[CST]]) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>
// CHECK: %[[DQ:.+]] = "quantization.dcast"(%[[Q]]) : (tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<2x3x3x2xf32>
// CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG_0]], %[[DQ]])
// CHECK-SAME: _entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn", _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {per_tensor {}}}}"
// CHECK-SAME: (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
// CHECK: return %[[CALL]] : tensor<1x2x2x2xf32>

// -----

// Test that q/dq pair with per-tensor quantization parameter is inserted
// between constant and XlaCallModule op with `weight_only_ptq` method of
// `per_tensor` and function name containing dot_general.

func.func @qdq_for_dot_general_weight_per_tensor(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn,
    _original_entry_function = "composite_dot_general_fn",
    _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {per_tensor {}}}}", _stablehlo_module_attrs = {},
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: func.func @qdq_for_dot_general_weight_per_tensor
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32>
// CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
// CHECK: %[[Q:.+]] = "quantization.qcast"(%[[CST]]) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>
// CHECK: %[[DQ:.+]] = "quantization.dcast"(%[[Q]]) : (tensor<2x3x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<2x3xf32>
// CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG_0]], %[[DQ]])
// CHECK-SAME: _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {per_tensor {}}}}"
// CHECK-SAME: (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]] : tensor<1x3xf32>

// -----

// Test that q/dq pair with per-channel quantization parameter is inserted
// between constant and XlaCallModule op with `weight_only_ptq` method of
// `quatized_type` without specified quantization dimension and function name
// containing conv.

module attributes {tf_saved_model.semantics} {
  func.func private @qdq_for_conv_weight_per_channel_default(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "tf.XlaCallModule"(%arg0, %cst) {
        Sout = [#tf_type.shape<1x3x4x2>], dim_args_spec = [], disabled_checks = [],
        has_token_input_output = false, module = "", platforms = [], version = 5 : i64,
        _entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn",
        _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {dimension_specs {}}}}",
        _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
        device = ""
      } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }

  // CHECK: func.func private @qdq_for_conv_weight_per_channel_default(%[[ARG0:.+]]: tensor<1x3x4x3xf32>)
  // CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
  // CHECK: %[[Q:.+]] = "quantization.qcast"(%[[CST]]) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>
  // CHECK: %[[DQ:.+]] = "quantization.dcast"(%[[Q]]) : (tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<2x3x3x2xf32>
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

// Test that q/dq pair with per-channel quantization parameter is inserted
// between constant and XlaCallModule op with `weight_only_ptq` method of
// `quatized_type` without specified quantization dimension and function name
// containing dot_general.

module attributes {tf_saved_model.semantics} {
  func.func private @qdq_for_dot_general_weight_per_channel_default(%arg0: tensor<4x3x6x5xf32>) -> tensor<4x3x6x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<4x3x5x2xf32>} : () -> tensor<4x3x5x2xf32>
    %0 = "tf.XlaCallModule"(%arg0, %cst) {
      Sout = [#tf_type.shape<4x3x6x2>], _entry_function = @composite_dot_general_fn,
      _original_entry_function = "composite_dot_general_fn",
      _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {dimension_specs {}}}}",
      _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
      device = "", dim_args_spec = [], disabled_checks = [],
      has_token_input_output = false, module = "", platforms = [],
      version = 5 : i64
    } : (tensor<4x3x6x5xf32>, tensor<4x3x5x2xf32>) -> tensor<4x3x6x2xf32>
    return %0 : tensor<4x3x6x2xf32>
  }
  // CHECK: func.func private @qdq_for_dot_general_weight_per_channel_default(%[[ARG0:.+]]: tensor<4x3x6x5xf32>)
  // CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<4x3x5x2xf32>}> : () -> tensor<4x3x5x2xf32>
  // CHECK: %[[Q:.+]] = "quantization.qcast"(%[[CST]]) : (tensor<4x3x5x2xf32>) -> tensor<4x3x5x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>
  // CHECK: %[[DQ:.+]] = "quantization.dcast"(%[[Q]]) : (tensor<4x3x5x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<4x3x5x2xf32>
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

// Test that q/dq pair with per-channel quantization parameter is inserted
// between constant and XlaCallModule op with `weight_only_ptq` method of
// `quatized_type` with specified quantization dimension and function name
// containing conv.

module attributes {tf_saved_model.semantics} {
  func.func private @qdq_for_conv_weight_per_channel(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "tf.XlaCallModule"(%arg0, %cst) {
        Sout = [#tf_type.shape<1x3x4x2>], dim_args_spec = [], disabled_checks = [],
        has_token_input_output = false, module = "", platforms = [], version = 5 : i64,
        _entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn",
        _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
        _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
        device = ""
      } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }

  // CHECK: func.func private @qdq_for_conv_weight_per_channel(%[[ARG0:.+]]: tensor<1x3x4x3xf32>)
  // CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
  // CHECK: %[[Q:.+]] = "quantization.qcast"(%[[CST]]) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>
  // CHECK: %[[DQ:.+]] = "quantization.dcast"(%[[Q]]) : (tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<2x3x3x2xf32>
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

// Test that q/dq pair with per-channel quantization parameter is inserted
// between constant and XlaCallModule op with `weight_only_ptq` method of
// `quatized_type` with specified quantization dimension and function name
// containing dot_general.

module attributes {tf_saved_model.semantics} {
  func.func private @qdq_for_dot_general_weight_per_channel(%arg0: tensor<4x3x6x5xf32>) -> tensor<4x3x6x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<4x3x5x2xf32>} : () -> tensor<4x3x5x2xf32>
    %0 = "tf.XlaCallModule"(%arg0, %cst) {
      Sout = [#tf_type.shape<4x3x6x2>], _entry_function = @composite_dot_general_fn,
      _original_entry_function = "composite_dot_general_fn",
      _quantization_method = "weight_only_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
      _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
      device = "", dim_args_spec = [], disabled_checks = [],
      has_token_input_output = false, module = "", platforms = [],
      version = 5 : i64
    } : (tensor<4x3x6x5xf32>, tensor<4x3x5x2xf32>) -> tensor<4x3x6x2xf32>
    return %0 : tensor<4x3x6x2xf32>
  }
  // CHECK: func.func private @qdq_for_dot_general_weight_per_channel(%[[ARG0:.+]]: tensor<4x3x6x5xf32>)
  // CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<4x3x5x2xf32>}> : () -> tensor<4x3x5x2xf32>
  // CHECK: %[[Q:.+]] = "quantization.qcast"(%[[CST]]) : (tensor<4x3x5x2xf32>) -> tensor<4x3x5x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>
  // CHECK: %[[DQ:.+]] = "quantization.dcast"(%[[Q]]) : (tensor<4x3x5x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<4x3x5x2xf32>
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

// Test that q/dq pair is not inserted between constant and XlaCallModule op
// whose entry function name does not include conv nor dot_general.

func.func @no_qdq_except_conv_and_dot_general(%arg0: tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<3x4x2xf32>} : () -> tensor<3x4x2xf32>
  %0 = "tf.XlaCallModule"(%cst, %arg0) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_gather_fn,
    _original_entry_function = "composite_gather_fn", _quantization_method = "weight_only_ptq { }",
    _stablehlo_module_attrs = {}, device = "", dim_args_spec = [],
    disabled_checks = [], has_token_input_output = false, module = "",
    platforms = [], version = 5 : i64
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
  return %0 : tensor<2x3x2x2xf32>
}

// CHECK-LABEL: func.func @no_qdq_except_conv_and_dot_general
// CHECK-NOT: quantization.qcast
// CHECK-NOT: quantization.dcast

// -----

// Test that q/dq pair is not inserted for constant whose operand number is
// not 1.

func.func @no_qdq_for_non_weight_constant(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<4.000000e-02> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %arg1, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_with_bias_fn,
    _original_entry_function = "composite_dot_general_with_bias_fn",
    _stablehlo_module_attrs = {}, _quantization_method = "weight_only_ptq { }",
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<1x2xf32>, tensor<2x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: func.func @no_qdq_for_non_weight_constant
// CHECK-NOT: quantization.qcast
// CHECK-NOT: quantization.dcast

// -----

// Test that q/dq pair is not inserted between constant and XlaCallModule op
// without `weight_only_ptq` method.

func.func @no_qdq_for_not_quantizable_call(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn,
    _original_entry_function = "composite_dot_general_fn",
    _stablehlo_module_attrs = {}, device = "", dim_args_spec = [],
    disabled_checks = [], has_token_input_output = false, module = "",
    platforms = [], version = 5 : i64
  } : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: func.func @no_qdq_for_not_quantizable_call
// CHECK-NOT: quantization.qcast
// CHECK-NOT: quantization.dcast

// -----

// Test that q/dq pair is not inserted between constant and XlaCallModule op
// with different method.

func.func @no_qdq_for_not_quantizable_call(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn,
    _original_entry_function = "composite_dot_general_fn",
    _stablehlo_module_attrs = {}, device = "", dim_args_spec = [],
    disabled_checks = [], has_token_input_output = false, module = "",
    platforms = [], _quantization_method = "static_range_ptq { }", version = 5 : i64
  } : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: func.func @no_qdq_for_not_quantizable_call
// CHECK-NOT: quantization.qcast
// CHECK-NOT: quantization.dcast

// -----

// Test that q/dq pair is not inserted when constant has multiple users.

func.func @no_qdq_for_multiple_users(%arg0: tensor<2x2xf32>) -> tensor<2x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn,
    _original_entry_function = "composite_dot_general_fn",
    _stablehlo_module_attrs = {}, _quantization_method = "weight_only_ptq { }",
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = stablehlo.add %cst, %0 : tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @no_qdq_for_multiple_users
// CHECK-NOT: quantization.qcast
// CHECK-NOT: quantization.dcast
