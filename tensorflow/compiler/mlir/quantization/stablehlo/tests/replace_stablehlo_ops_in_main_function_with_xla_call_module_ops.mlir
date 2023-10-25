// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-replace-stablehlo-ops-in-main-function-with-xla-call-module-ops | FileCheck %s

// Modules with "main" or "serving_default" should properly run this pass and convert subgraphs into XLACallModuleOp.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {

  // CHECK: func private @_stablehlo_main_1
  // CHECK: %[[CONSTANT_0:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[CONSTANT_1:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1x3xf32>
  // CHECK: return
  // CHECK: }

  // CHECK: func private @_stablehlo_main_0
  // CHECK: %[[CONSTANT_0:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<3x64xf32>
  // CHECK: %[[CONSTANT_1:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1x64xf32>
  // CHECK: return
  // CHECK: }

  func.func @main(%arg0: tensor<1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x64xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %1 = stablehlo.constant dense<1.000000e+03> : tensor<1x3xf32>
    %2 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %3 = "tf.XlaCallModule"(%2, %0, %1) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x1024xf32>, tensor<1024x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %4 = "tf.CustomAggregator"(%3) {calibration_method = 1 : i32, id = "1", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %5 = stablehlo.constant dense<1.000000e+03> : tensor<3x64xf32>
    %6 = stablehlo.constant dense<1.000000e+03> : tensor<1x64xf32>
    %7 = "tf.CustomAggregator"(%4) {calibration_method = 1 : i32, id = "0", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %8 = "tf.XlaCallModule"(%7, %5, %6) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_with_relu_fn_1, _original_entry_function = "composite_dot_general_with_relu_fn_1", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x3xf32>, tensor<3x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
    %9 = "tf.CustomAggregator"(%6) {calibration_method = 1 : i32, id = "1", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x64xf32>) -> tensor<1x64xf32>
    return %9 : tensor<1x64xf32>
  }

  // CHECK: %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_0:.*]] = "tf.XlaCallModule"() {Sout = [#tf_type.shape<{{.*}}>, #tf_type.shape<{{.*}}>], _entry_function = @_stablehlo_main_1
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
  // CHECK: %[[XLA_CALL_MODULE_0:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_0]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_0:.*]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_0:.*]]) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_0]])
  // CHECK: %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_1:.*]] = "tf.XlaCallModule"() {Sout = [#tf_type.shape<{{.*}}>, #tf_type.shape<{{.*}}>], _entry_function = @_stablehlo_main_0
  // CHECK: %[[CUSTOM_AGGREGATOR_2:.*]] = "tf.CustomAggregator"(%[[CUSTOM_AGGREGATOR_1]]) {calibration_method = 1 : i32, id = "0", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  // CHECK: %[[XLA_CALL_MODULE_1:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_2]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_1:.*]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_1:.*]]) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_with_relu_fn_1, _original_entry_function = "composite_dot_general_with_relu_fn_1"
  // CHECK: %[[CUSTOM_AGGREGATOR_3:.*]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_1:.*]])
  // CHECK: return %[[CUSTOM_AGGREGATOR_3]] : tensor<1x64xf32>
  // CHECK: }

  // CHECK: @composite_dot_general_fn_1
  // CHECK-NOT: tf_quant.composite_function
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  // CHECK: @composite_dot_general_with_relu_fn_1
  // CHECK-NOT: tf_quant.composite_function
  func.func private @composite_dot_general_with_relu_fn_1(%arg0: tensor<1x3xf32>, %arg1: tensor<3x64xf32>, %arg2: tensor<1x64xf32>) -> tensor<1x64xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x64xf32>
    %1 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x3xf32>, tensor<3x64xf32>) -> tensor<1x64xf32>
    %2 = stablehlo.add %1, %arg2 : tensor<1x64xf32>
    %3 = stablehlo.maximum %2, %0 : tensor<1x64xf32>
    return %3 : tensor<1x64xf32>
  }
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK: func private @_stablehlo_main_
  // CHECK: %[[CONSTANT:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: return %[[CONSTANT:.*]]
  // CHECK: }

  // CHECK: @serving_default
  func.func @serving_default(%arg0: tensor<1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %1 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %2 = "tf.XlaCallModule"(%1, %0) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    %3 = "tf.CustomAggregator"(%2) {calibration_method = 1 : i32, id = "1", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }

  // CHECK: %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP:.*]] = "tf.XlaCallModule"() {Sout = [#tf_type.shape<1024x3>], _entry_function = @_stablehlo_main_
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR:.*]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP:.*]]) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
  // CHECK: return %[[CUSTOM_AGGREGATOR_1]]
  // CHECK: }

  // CHECK: @composite_dot_general_fn_1
  // CHECK-NOT: tf_quant.composite_function
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// -----

// Tests to confirm that the StableHLO graph is not replaced if "main" or "serving_default" function is in the module.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK-NOT: func private @_stablehlo_main_

  // CHECK-LABEL: @random_name
  func.func @random_name(%arg0: tensor<1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %1 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %2 = "tf.XlaCallModule"(%1, %0) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    %3 = "tf.CustomAggregator"(%2) {calibration_method = 1 : i32, id = "1", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }

  // CHECK: %[[CONSTANT:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR:.*]], %[[XLA_CALL_MODULE_EXTRACTED_FROM_SUBGRAPH:.*]]) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1"
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]] = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
  // CHECK: return %[[CUSTOM_AGGREGATOR_1]]
  // CHECK: }

  // CHECK: @composite_dot_general_fn_1
  // CHECK: tf_quant.composite_function
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}
