// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-insert-weight-param | FileCheck %s

// Test that q/dq pair is inserted between constant and XlaCallModule op
// with quantizable trait and function name containing conv.

func.func @qdq_for_conv_weight(%arg0: tensor<1x3x2x3xf32>) -> tensor<1x2x2x2xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x2x2x2>], _entry_function = @composite_conv_fn,
    _original_entry_function = "composite_conv_fn",
    _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @qdq_for_conv_weight
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x3x2x3xf32>) -> tensor<1x2x2x2xf32>
// CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
// CHECK: %[[Q:.+]] = "quantfork.qcast"(%[[CST]]) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8:f32, 0.0011764706349840352:-128>>
// CHECK: %[[DQ:.+]] = "quantfork.dcast"(%[[Q]]) : (tensor<2x3x3x2x!quant.uniform<i8:f32, 0.0011764706349840352:-128>>) -> tensor<2x3x3x2xf32>
// CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG_0]], %[[DQ]]) <{Sout = [#tf_type.shape<1x2x2x2>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_conv_fn, _original_entry_function = "composite_conv_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
// CHECK: return %[[CALL]] : tensor<1x2x2x2xf32>

// -----

// Test that q/dq pair is inserted between constant and XlaCallModule op
// with quantizable trait and function name containing dot_general.

func.func @qdq_for_dot_general_weight(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn,
    _original_entry_function = "composite_dot_general_fn",
    _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: func.func @qdq_for_dot_general_weight
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32>
// CHECK: %[[CST:.+]] = "tf.Const"() <{value = dense<3.000000e-01> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
// CHECK: %[[Q:.+]] = "quantfork.qcast"(%[[CST]]) : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i8:f32, 0.0011764706349840352:-128>>
// CHECK: %[[DQ:.+]] = "quantfork.dcast"(%[[Q]]) : (tensor<2x3x!quant.uniform<i8:f32, 0.0011764706349840352:-128>>) -> tensor<2x3xf32>
// CHECK: %[[CALL:.+]] = "tf.XlaCallModule"(%[[ARG_0]], %[[DQ]]) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]] : tensor<1x3xf32>

// -----

// Test that q/dq pair is not inserted between constant and XlaCallModule op
// whose entry function name does not include conv nor dot_general.

func.func @no_qdq_except_conv_and_dot_general(%arg0: tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<3x4x2xf32>} : () -> tensor<3x4x2xf32>
  %0 = "tf.XlaCallModule"(%cst, %arg0) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_gather_fn,
    _original_entry_function = "composite_gather_fn",
    _stablehlo_module_attrs = {}, device = "", dim_args_spec = [],
    disabled_checks = [], has_token_input_output = false, module = "",
    platforms = [], version = 5 : i64
  } : (tensor<3x4x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
  return %0 : tensor<2x3x2x2xf32>
}

// CHECK-LABEL: func.func @no_qdq_except_conv_and_dot_general
// CHECK-NOT: quantfork.qcast
// CHECK-NOT: quantfork.dcast

// -----

// Test that q/dq pair is not inserted for constant whose operand number is
// not 1.

func.func @no_qdq_for_non_weight_constant(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<4.000000e-02> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %arg1, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_with_bias_fn,
    _original_entry_function = "composite_dot_general_with_bias_fn",
    _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<1x2xf32>, tensor<2x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}

// CHECK-LABEL: func.func @no_qdq_for_non_weight_constant
// CHECK-NOT: quantfork.qcast
// CHECK-NOT: quantfork.dcast

// -----

// Test that q/dq pair is not inserted between constant and XlaCallModule op
// without quantizable trait.

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
// CHECK-NOT: quantfork.qcast
// CHECK-NOT: quantfork.dcast

// -----

// Test that q/dq pair is not inserted when constant has multiple users.

func.func @no_qdq_for_multiple_users(%arg0: tensor<2x2xf32>) -> tensor<2x3xf32> attributes {tf._original_func_name = "main_0"} {
  %cst = "tf.Const"() {value = dense<3.000000e-01> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %0 = "tf.XlaCallModule"(%arg0, %cst) {
    Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn,
    _original_entry_function = "composite_dot_general_fn",
    _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",
    device = "", dim_args_spec = [], disabled_checks = [],
    has_token_input_output = false, module = "", platforms = [],
    version = 5 : i64
  } : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = stablehlo.add %cst, %0 : tensor<2x3xf32>
  return %2 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @no_qdq_for_multiple_users
// CHECK-NOT: quantfork.qcast
// CHECK-NOT: quantfork.dcast
