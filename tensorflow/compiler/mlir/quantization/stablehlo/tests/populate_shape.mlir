// RUN: stablehlo-quant-opt %s -split-input-file -populate-shape | FileCheck %s

// CHECK-LABEL: @populate_shape_for_custom_aggregator
func.func @populate_shape_for_custom_aggregator(%input: tensor<?x56x56x64xf32>) {
  // CHECK: %[[OUTPUT:.*]] = "tf.CustomAggregator"(%[[INPUT:.*]]) <{id = "49d53b0"}> {calibration_method = 1 : i64, device = "", initial_num_bins = 0 : i64, max = 6.000000e+00 : f32, max_percentile = 0.000000e+00 : f32, min = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<?x56x56x64xf32>) -> tensor<?x56x56x64xf32>
  %0 = "tf.CustomAggregator"(%input) <{id = "49d53b0"}> {calibration_method = 1 : i64, device = "", initial_num_bins = 0 : i64, max = 6.000000e+00 : f32, max_percentile = 0.000000e+00 : f32, min = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<?x56x56x64xf32>) -> tensor<*xf32>
  func.return
}

// ----

// CHECK-LABEL: @populate_shape_for_xla_call_module
func.func @populate_shape_for_xla_call_module(%input: tensor<?x56x56x256xf32>) {
  %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<1x1x64x256xf32>} : () -> tensor<1x1x64x256xf32>
  // CHECK: %[[OUTPUT:.*]] = "tf.XlaCallModule"(%[[INPUT:.*]], %[[CST:.*]]) <{Sout = [#tf_type.shape<?x56x56x256>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @main_9, _original_entry_function = "composite_conv_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<?x56x56x256xf32>, tensor<1x1x64x256xf32>) -> tensor<?x56x56x256xf32>
  %0 = "tf.XlaCallModule"(%input, %cst) <{Sout = [#tf_type.shape<?x56x56x256>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @main_9, _original_entry_function = "composite_conv_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<?x56x56x256xf32>, tensor<1x1x64x256xf32>) -> tensor<*xf32>
  func.return
}

// ----

// CHECK-LABEL: @populate_shape_for_chain_of_ops
func.func @populate_shape_for_chain_of_ops(%input: tensor<?x56x56x64xf32>) {
  %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<1x1x64x256xf32>} : () -> tensor<1x1x64x256xf32>
  // CHECK: %[[VAL_0:.*]] = "tf.CustomAggregator"(%[[INPUT:.*]]) <{id = "49d53b0"}> {calibration_method = 1 : i64, device = "", initial_num_bins = 0 : i64, max = 6.000000e+00 : f32, max_percentile = 0.000000e+00 : f32, min = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<?x56x56x64xf32>) -> tensor<?x56x56x64xf32>
  // CHECK: %[[VAL_1:.*]] = "tf.XlaCallModule"(%[[VAL_0:.*]], %[[CST:.*]]) <{Sout = [#tf_type.shape<?x56x56x256>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @main_9, _original_entry_function = "composite_conv_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<?x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<?x56x56x256xf32>
  // CHECK: %[[VAL_2:.*]] = "tf.CustomAggregator"(%[[VAL_1:.*]]) <{id = "49d53b1"}> {calibration_method = 1 : i64, device = "", initial_num_bins = 0 : i64, max = 6.000000e+00 : f32, max_percentile = 0.000000e+00 : f32, min = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<?x56x56x256xf32>) -> tensor<?x56x56x256xf32>
  %0 = "tf.CustomAggregator"(%input) <{id = "49d53b0"}> {calibration_method = 1 : i64, device = "", initial_num_bins = 0 : i64, max = 6.000000e+00 : f32, max_percentile = 0.000000e+00 : f32, min = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<?x56x56x64xf32>) -> tensor<*xf32>
  %1 = "tf.XlaCallModule"(%0, %cst) <{Sout = [#tf_type.shape<?x56x56x256>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @main_9, _original_entry_function = "composite_conv_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<*xf32>, tensor<1x1x64x256xf32>) -> tensor<*xf32>
  %2 = "tf.CustomAggregator"(%1) <{id = "49d53b1"}> {calibration_method = 1 : i64, device = "", initial_num_bins = 0 : i64, max = 6.000000e+00 : f32, max_percentile = 0.000000e+00 : f32, min = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
  func.return
}

// ----

// CHECK-LABEL: @populate_shape_for_xla_call_module_failure_not_single_output
func.func @populate_shape_for_xla_call_module_failure_not_single_output(%input: tensor<?x56x56x256xf32>) {
  %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<1x1x64x256xf32>} : () -> tensor<1x1x64x256xf32>
  // expected-error @+2 {{XlaCallModuleOp doesn't have 1 output.}}
  %0, %1 = "tf.XlaCallModule"(%input, %cst) <{Sout = [#tf_type.shape<?x56x56x256>, #tf_type.shape<?x56x56x256>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @main_9, _original_entry_function = "composite_conv_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<?x56x56x256xf32>, tensor<1x1x64x256xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  // expected-error @+1 {{XlaCallModuleOp doesn't have 1 output.}}
  "tf.XlaCallModule"(%input, %cst) <{Sout = [], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @main_9, _original_entry_function = "composite_conv_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<?x56x56x256xf32>, tensor<1x1x64x256xf32>) -> ()
  func.return
}
