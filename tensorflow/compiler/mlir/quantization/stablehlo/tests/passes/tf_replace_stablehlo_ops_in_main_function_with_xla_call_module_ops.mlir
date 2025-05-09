// RUN: stablehlo-quant-opt %s -split-input-file \
// RUN:    -tf-stablehlo-replace-stablehlo-ops-in-main-function-with-xla-call-module-ops \
// RUN:    | FileCheck %s

// Modules with "main" or "serving_default" should properly run this pass and
// convert subgraphs into XLACallModuleOp.

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
    %2:4 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1x1024xf32>) -> (tensor<1x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %3 = "tf.XlaCallModule"(%2#0, %0, %1) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "{{.*}}", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64} : (tensor<1x1024xf32>, tensor<1024x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %4:4 = "tf.CustomAggregator"(%3) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %5 = stablehlo.constant dense<1.000000e+03> : tensor<3x64xf32>
    %6 = stablehlo.constant dense<1.000000e+03> : tensor<1x64xf32>
    %7:4 = "tf.CustomAggregator"(%4#0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %8 = "tf.XlaCallModule"(%7#0, %5, %6) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_with_relu_fn_1, _original_entry_function = "composite_dot_general_with_relu_fn_1", _stablehlo_version = "{{.*}}", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64} : (tensor<1x3xf32>, tensor<3x64xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
    %9:4 = "tf.CustomAggregator"(%6) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1x64xf32>) -> (tensor<1x64xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    return %9#0 : tensor<1x64xf32>
  }

  // CHECK: %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_0:.*]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<{{.*}}>, #tf_type.shape<{{.*}}>], {{.*}}, module = "", platforms = ["CPU", "TPU"], use_shardy_partitioner = false, version = 9 : i64}> {_entry_function = @_stablehlo_main_1
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE_0:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_0]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_0:.*]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_0:.*]]) <{Sout = [#tf_type.shape<1x3>], {{.*}}, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "{{.*}}", _tfl_quant_trait = "fully_quantizable"}
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_0]])
  // CHECK: %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_1:.*]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<{{.*}}>, #tf_type.shape<{{.*}}>], {{.*}}, module = "", platforms = ["CPU", "TPU"], use_shardy_partitioner = false, version = 9 : i64}> {_entry_function = @_stablehlo_main_0
  // CHECK: %[[CUSTOM_AGGREGATOR_2:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[CUSTOM_AGGREGATOR_1]]) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE_1:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_2]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_1:.*]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP_1:.*]]) <{Sout = [#tf_type.shape<1x3>], {{.*}}, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_with_relu_fn_1, _original_entry_function = "composite_dot_general_with_relu_fn_1", _stablehlo_version = "{{.*}}", _tfl_quant_trait = "fully_quantizable"}
  // CHECK: %[[CUSTOM_AGGREGATOR_3:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE_1:.*]])
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

// Tests that the subgraph in serving_default excluding the tf.Identity is
// converted to a single XlaCallModuleOp.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1654 : i32}, tf_saved_model.semantics} {

  // CHECK: func private @_stablehlo_main_0(%arg0: tensor<i32>, %arg1: tensor<1x1024xf32>)
  // CHECK: %[[CONSTANT_0:.*]] = stablehlo.constant dense<0.134728625> : tensor<1x3xf32>
  // CHECK: %[[CONSTANT_1:.*]] = stablehlo.constant dense<-1.280000e+02> : tensor<1x1024xf32>
  // CHECK: %[[CONSTANT_2:.*]] = stablehlo.constant dense<0.003921567> : tensor<1x1024xf32>
  // CHECK: %[[DIVIDE:.*]] = stablehlo.divide %arg1, %[[CONSTANT_2]]
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[DIVIDE]], %[[CONSTANT_1]]
  // CHECK return %[[ADD]]
  // CHECK: }

  // CHECK: @serving_default
  func.func @serving_default(%arg0: tensor<1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x1024xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<0.134728625> : tensor<1x3xf32>
    %1 = stablehlo.constant dense<-1.280000e+02> : tensor<1x1024xf32>
    %2 = stablehlo.constant dense<0.003921567> : tensor<1x1024xf32>
    %3 = stablehlo.divide %arg0, %2 : tensor<1x1024xf32>
    %4 = stablehlo.add %3, %1 : tensor<1x1024xf32>
    %5 = "tf.Identity"(%4) {device = ""} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    return %5 : tensor<1x1024xf32>
  }

 // CHECK: %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP:.*]] = "tf.XlaCallModule"(%arg0) <{Sout = [#tf_type.shape<1x1024>], {{.*}}, module = "", platforms = ["CPU", "TPU"], use_shardy_partitioner = false, version = 9 : i64}> {_entry_function = @_stablehlo_main_0, _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _stablehlo_version = "{{.*}}"} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
 // CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP]])
 // CHECK: return %[[IDENTITY]]
 // CHECK }

}

// -----

// Tests that the first stablehlo.constant is converted to XlaCallModuleOp.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK: func private @_stablehlo_main_0
  // CHECK: %[[CONSTANT:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: return %[[CONSTANT:.*]]
  // CHECK: }

  // CHECK: @serving_default
  func.func @serving_default(%arg0: tensor<1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %1:4 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1x1024xf32>) -> (tensor<1x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %2 = "tf.XlaCallModule"(%1#0, %0) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "{{.*}}", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    %3:4 = "tf.CustomAggregator"(%2) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    return %3#0 : tensor<1x3xf32>
  }

  // CHECK: %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP:.*]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<1024x3>], {{.*}}, module = "", platforms = ["CPU", "TPU"], use_shardy_partitioner = false, version = 9 : i64}> {_entry_function = @_stablehlo_main_0, _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _stablehlo_version = "{{.*}}"}
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR:.*]], %[[STABLEHLO_SUBGRAPH_TO_XLA_CALL_MODULE_OP:.*]]) <{Sout = [#tf_type.shape<1x3>], {{.*}}}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "{{.*}}"
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
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

// Tests to confirm that the StableHLO graph is not replaced if "main" or
// "serving_default" function is not in the module.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK-NOT: func private @_stablehlo_main_

  // CHECK-LABEL: @random_name
  func.func @random_name(%arg0: tensor<1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %1:4 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1x1024xf32>) -> (tensor<1x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %2 = "tf.XlaCallModule"(%1#0, %0) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    %3:4 = "tf.CustomAggregator"(%2) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    return %3#0 : tensor<1x3xf32>
  }

  // CHECK: %[[CONSTANT:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR:.*]], %[[XLA_CALL_MODULE_EXTRACTED_FROM_SUBGRAPH:.*]]) <{Sout = [#tf_type.shape<1x3>], {{.*}}, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0"
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
  // CHECK: return %[[CUSTOM_AGGREGATOR_1]]
  // CHECK: }

  // CHECK: @composite_dot_general_fn_1
  // CHECK: tf_quant.composite_function
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// -----

// Tests where StableHLO graph in main has a small constant to be duplicated.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK: func private @_stablehlo_main_1(%arg0: tensor<i32>) -> tensor<1024x3xf32> attributes {_from_xla_call_module}
  // CHECK: %[[CONSTANT1:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: return %[[CONSTANT1:.*]]
  // CHECK: }

  // CHECK: func private @_stablehlo_main_0(%arg0: tensor<i32>
  // CHECK-SAME: %[[INPUT1:.*]]: tensor<1024x3xf32>, %[[INPUT2:.*]]: tensor<1024x3xf32>
  // CHECK: %[[CONSTANT2:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[INPUT1]], %[[CONSTANT2]] : tensor<1024x3xf32>
  // CHECK: %[[MUL:.*]] = stablehlo.multiply %[[INPUT1]], %[[INPUT2]] : tensor<1024x3xf32>
  // CHECK: return %[[ADD]], %[[MUL]]
  // CHECK: }

  // CHECK: @serving_default
  func.func @serving_default(%arg0: tensor<1024x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1024x3xf32> {tf_saved_model.index_path = ["output1"]}, tensor<1024x3xf32> {tf_saved_model.index_path = ["output2"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %1:4 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %2 = "tf.XlaCallModule"(%1#0, %0) {Sout = [#tf_type.shape<1024x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1024x1024xf32>, tensor<1024x3xf32>) -> tensor<1024x3xf32>
    %3:4 = "tf.CustomAggregator"(%2) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1024x3xf32>) -> (tensor<1024x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %4 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %5 = stablehlo.add %3#0, %4 : tensor<1024x3xf32>
    %6 = stablehlo.multiply %3#0, %0 : tensor<1024x3xf32>
    return %5, %6 : tensor<1024x3xf32>, tensor<1024x3xf32>
  }

  // CHECK: %[[SUBGRAPH_1:.*]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<1024x3>], {{.*}} ["CPU", "TPU"], {{.*}}}> {_entry_function = @_stablehlo_main_1
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_1]], %[[SUBGRAPH_1]]) <{Sout = [#tf_type.shape<1024x3>], {{.*}}}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0"
  // CHECK: %[[CUSTOM_AGGREGATOR_2:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
  // CHECK: %[[SUBGRAPH_2:.*]]:2 = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_2]], %[[SUBGRAPH_1]]) <{Sout = [#tf_type.shape<1024x3>, #tf_type.shape<1024x3>], {{.*}}}> {_entry_function = @_stablehlo_main_0
  // CHECK: return %[[SUBGRAPH_2]]#0, %[[SUBGRAPH_2]]#1
  // CHECK: }

  // CHECK: @composite_dot_general_fn_1
  // CHECK-NOT: tf_quant.composite_function
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

// -----

// Tests where StableHLO graph in main has branches.
// This test makes sure tracing won't stop at op (%1) with multiple uses.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK: func private @_stablehlo_main_1(%arg0: tensor<i32>) -> tensor<3x11xf32>
  // CHECK: %[[CONSTANT_1:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<3x11xf32>
  // CHECK: return %[[CONSTANT_1:.*]]
  // CHECK: }

  // CHECK: func private @_stablehlo_main_0
  // CHECK-SAME: (%arg0: tensor<i32>, %[[INPUT_1:.*]]: tensor<3x11xf32>)
  // CHECK-SAME: -> tensor<3x11xf32>
  // CHECK: %[[CONSTANT_2:.*]] = stablehlo.constant dense<1.000000e+01> : tensor<3x11xf32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[INPUT_1]], %[[CONSTANT_2]] : tensor<3x11xf32>
  // CHECK: %[[MUL:.*]] = stablehlo.multiply %[[ADD]], %[[CONSTANT_2]] : tensor<3x11xf32>
  // CHECK: return %[[MUL]]
  // CHECK: }

  // CHECK: @serving_default
  func.func @serving_default(%arg0: tensor<3x3xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<3x11xf32> {tf_saved_model.index_path = ["output1"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<1.000000e+03> : tensor<3x11xf32>
    // %1 is large enough that it won't be duplicated.
    %1 = stablehlo.constant dense<1.000000e+01> : tensor<3x11xf32>
    %2:4 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %3 = "tf.XlaCallModule"(%2#0, %0) {Sout = [#tf_type.shape<3x11>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<3x3xf32>, tensor<3x11xf32>) -> tensor<3x11xf32>
    %4:4 = "tf.CustomAggregator"(%3) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<3x11xf32>) -> (tensor<3x11xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %5 = stablehlo.add %4#0, %1 : tensor<3x11xf32>
    %6 = stablehlo.multiply %5, %1 : tensor<3x11xf32>
    return %6 : tensor<3x11xf32>
  }

  // CHECK: %[[SUBGRAPH_1:.*]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<3x11>], {{.*}} ["CPU", "TPU"], {{.*}}}> {_entry_function = @_stablehlo_main_1
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_1]], %[[SUBGRAPH_1]]) <{Sout = [#tf_type.shape<3x11>], {{.*}}}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0"
  // CHECK: %[[CUSTOM_AGGREGATOR_2:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
  // CHECK: %[[SUBGRAPH_2:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_2]]) <{Sout = [#tf_type.shape<3x11>], {{.*}} ["CPU", "TPU"], {{.*}}}> {_entry_function = @_stablehlo_main_0
  // CHECK: return %[[SUBGRAPH_2]]
  // CHECK: }

  // CHECK: @composite_dot_general_fn_1
  // CHECK-NOT: tf_quant.composite_function
  func.func private @composite_dot_general_fn_1(%arg0: tensor<3x3xf32>, %arg1: tensor<3x11xf32>) -> tensor<3x11xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x3xf32>, tensor<3x11xf32>) -> tensor<3x11xf32>
    return %0 : tensor<3x11xf32>
  }
}

// -----

// Tests where StableHLO graph in main has dead end.
// This test makes sure tracing will include the dead end from the op in the
// same sub graph:
// stablehlo.add and %0 along with its dead end branch are in the same sub
// graph.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK: func.func private @_stablehlo_main_1(%arg0: tensor<i32>) -> tensor<1024x3xf32> attributes {_from_xla_call_module} {
  // CHECK: %[[CONSTANT_0:.*]] = stablehlo.constant dense<2.000000e+03> : tensor<1024x3xf32>
  // CHECK: return %[[CONSTANT_0]]
  // CHECK: }

  // CHECK: func.func private @_stablehlo_main_0(%arg0: tensor<i32>, %[[ARG_1:.*]]: tensor<1024x3xf32>) -> tensor<1024x3xf32> attributes {_from_xla_call_module} {
  // CHECK: %[[CONSTANT_1:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[CONSTANT_2:.*]] = stablehlo.constant dense<5.000000e+01> : tensor<1024x3xf32>
  // CHECK: %[[CONSTANT_3:.*]] = stablehlo.constant dense<4.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[REMAINDER:.*]] = stablehlo.remainder %[[CONSTANT_3]], %[[CONSTANT_1]] : tensor<1024x3xf32>
  // CHECK: %[[COMPARE:.*]] = stablehlo.compare  EQ, %[[REMAINDER]], %[[CONSTANT_2]],  NOTYPE : (tensor<1024x3xf32>, tensor<1024x3xf32>) -> tensor<1024x3xi1>
  // CHECK: stablehlo.custom_call @shape_assertion(%[[COMPARE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<1024x3xi1>) -> ()
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ARG_1]], %[[CONSTANT_3]]
  // CHECK: return %[[ADD]]
  // CHECK: }

  // CHECK: @serving_default
  func.func @serving_default(%arg0: tensor<1024x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1024x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<4.000000e+03> : tensor<1024x3xf32>
    %1 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %2 = stablehlo.constant dense<5.000000e+01> : tensor<1024x3xf32>
    %3 = stablehlo.remainder %0, %1 : tensor<1024x3xf32>
    %4 = stablehlo.compare  EQ, %3, %2,  NOTYPE : (tensor<1024x3xf32>, tensor<1024x3xf32>) -> tensor<1024x3xi1>
    stablehlo.custom_call @shape_assertion(%4) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<1024x3xi1>) -> ()
    %5 = stablehlo.constant dense<2.000000e+03> : tensor<1024x3xf32>
    %6:4 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %7 = "tf.XlaCallModule"(%6#0, %5) {Sout = [#tf_type.shape<1024x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1024x1024xf32>, tensor<1024x3xf32>) -> tensor<1024x3xf32>
    %8:4 = "tf.CustomAggregator"(%7) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1024x3xf32>) -> (tensor<1024x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %9 = stablehlo.add %8#0, %0 : tensor<1024x3xf32>
    return %9 : tensor<1024x3xf32>
  }
  // CHECK: %[[SUBGRAPH_0:.*]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<1024x3>], {{.*}} ["CPU", "TPU"], {{.*}}}> {_entry_function = @_stablehlo_main_1
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_0]], %[[SUBGRAPH_0]]) <{Sout = [#tf_type.shape<1024x3>], {{.*}}}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0"
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
  // CHECK: %[[SUBGRAPH_1:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_1]]) <{Sout = [#tf_type.shape<1024x3>], {{.*}}}> {_entry_function = @_stablehlo_main_0
  // CHECK: return %[[SUBGRAPH_1]] : tensor<1024x3xf32>
  // CHECK: }
}

// -----

// Tests where StableHLO graph in main has branch.
// This test makes sure the branch will not be added to subgraph when it reaches
// a tf op:
// stablehlo.add and %0 are not in the same subgraph.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK: func.func private @_stablehlo_main_2(%arg0: tensor<i32>) -> (tensor<1024x3xf32>, tensor<1024x3xf32>) attributes {_from_xla_call_module} {
  // CHECK: %[[CONSTANT_0:.*]] = stablehlo.constant dense<4.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[CONSTANT_1:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[REMAINDER:.*]] = stablehlo.remainder %[[CONSTANT_0]], %[[CONSTANT_1]] : tensor<1024x3xf32>
  // CHECK: return %[[CONSTANT_0]], %[[REMAINDER]]
  // CHECK: }

  // CHECK: func.func private @_stablehlo_main_1(%arg0: tensor<i32>) -> tensor<1024x3xf32> attributes {_from_xla_call_module} {
  // CHECK: %[[CONSTANT_2:.*]] = stablehlo.constant dense<2.000000e+03> : tensor<1024x3xf32>
  // CHECK: return %[[CONSTANT_2]] : tensor<1024x3xf32>
  // CHECK: }

  // CHECK: func.func private @_stablehlo_main_0(%arg0: tensor<i32>, %[[ARG_1:.*]]: tensor<1024x3xf32>, %[[ARG_2:.*]]: tensor<1024x3xf32>) -> tensor<1024x3xf32> attributes {_from_xla_call_module} {
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ARG_1]], %[[ARG_2]]
  // CHECK: return %[[ADD]]

  // CHECK: @serving_default
  func.func @serving_default(%arg0: tensor<1024x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1024x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<4.000000e+03> : tensor<1024x3xf32>
    %1 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %2 = stablehlo.remainder %0, %1 : tensor<1024x3xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<1024x3xf32>) -> tensor<1024x3xf32>
    %4 = stablehlo.constant dense<2.000000e+03> : tensor<1024x3xf32>
    %5:4 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %6 = "tf.XlaCallModule"(%5#0, %4) {Sout = [#tf_type.shape<1024x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1024x1024xf32>, tensor<1024x3xf32>) -> tensor<1024x3xf32>
    %7:4 = "tf.CustomAggregator"(%6) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1024x3xf32>) -> (tensor<1024x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %8 = stablehlo.add %7#0, %0 : tensor<1024x3xf32>
    return %8 : tensor<1024x3xf32>
  }
  // CHECK: %[[SUBGRAPH_0:.*]]:2 = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<1024x3>, #tf_type.shape<1024x3>], {{.*}} ["CPU", "TPU"], {{.*}}}> {_entry_function = @_stablehlo_main_2
  // CHECK: %[[IDENTIFY:.*]] = "tf.Identity"(%[[SUBGRAPH_0]]#1) {device = ""} : (tensor<1024x3xf32>) -> tensor<1024x3xf32>
  // CHECK: %[[SUBGRAPH_1:.*]] = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<1024x3>], {{.*}} ["CPU", "TPU"], {{.*}}}> {_entry_function = @_stablehlo_main_1
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_0]], %[[SUBGRAPH_1]]) <{Sout = [#tf_type.shape<1024x3>], {{.*}}}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "1.0.0"
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
  // CHECK: %[[SUBGRAPH_2:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_1]], %[[SUBGRAPH_0]]#0) <{Sout = [#tf_type.shape<1024x3>], {{.*}}}> {_entry_function = @_stablehlo_main_0
  // CHECK: return %[[SUBGRAPH_2]] : tensor<1024x3xf32>
  // CHECK: }
}

// -----

// Tests where StableHLO graph in main has dead end.
// This test checks tracing will stop if the dead end is too deep (>5):
// stablehlo.add and %0 are not in the same subgraph.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK: func.func private @_stablehlo_main_1(%arg0: tensor<i32>) -> (tensor<1024x3xf32>, tensor<1024x3xf32>) attributes {_from_xla_call_module} {
  // CHECK: %[[CONSTANT_0:.*]] = stablehlo.constant dense<4.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[CONSTANT_1:.*]] = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
  // CHECK: %[[CONSTANT_2:.*]] = stablehlo.constant dense<5.000000e+01> : tensor<1024x3xf32>
  // CHECK: %[[REMAINDER_0:.*]] = stablehlo.remainder %[[CONSTANT_0]], %[[CONSTANT_1]] : tensor<1024x3xf32>
  // CHECK: %[[REMAINDER_1:.*]] = stablehlo.remainder %[[REMAINDER_0]], %[[CONSTANT_1]] : tensor<1024x3xf32>
  // CHECK: %[[REMAINDER_2:.*]] = stablehlo.remainder %[[REMAINDER_1]], %[[CONSTANT_1]] : tensor<1024x3xf32>
  // CHECK: %[[REMAINDER_3:.*]] = stablehlo.remainder %[[REMAINDER_2]], %[[CONSTANT_1]] : tensor<1024x3xf32>
  // CHECK: %[[COMPARE:.*]] = stablehlo.compare  EQ, %[[REMAINDER_3]], %[[CONSTANT_2]],  NOTYPE : (tensor<1024x3xf32>, tensor<1024x3xf32>) -> tensor<1024x3xi1>
  // CHECK: stablehlo.custom_call @shape_assertion(%[[COMPARE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<1024x3xi1>) -> ()
  // CHECK: %[[CONSTANT_3:.*]] = stablehlo.constant dense<2.000000e+03> : tensor<1024x3xf32>
  // CHECK: return %[[CONSTANT_0]], %[[CONSTANT_3]]
  // CHECK: }

  // CHECK: func.func private @_stablehlo_main_0(%arg0: tensor<i32>, %[[ARG_1:.*]]: tensor<1024x3xf32>, %[[ARG_2:.*]]: tensor<1024x3xf32>) -> tensor<1024x3xf32> attributes {_from_xla_call_module} {
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ARG_1]], %[[ARG_2]]
  // CHECK: return %[[ADD]]
  // CHECK: }

  // CHECK: @serving_default
  func.func @serving_default(%arg0: tensor<1024x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1024x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<4.000000e+03> : tensor<1024x3xf32>
    %1 = stablehlo.constant dense<1.000000e+03> : tensor<1024x3xf32>
    %2 = stablehlo.constant dense<5.000000e+01> : tensor<1024x3xf32>
    %3 = stablehlo.remainder %0, %1 : tensor<1024x3xf32>
    %4 = stablehlo.remainder %3, %1 : tensor<1024x3xf32>
    %5 = stablehlo.remainder %4, %1 : tensor<1024x3xf32>
    %6 = stablehlo.remainder %5, %1 : tensor<1024x3xf32>
    %7 = stablehlo.compare  EQ, %6, %2,  NOTYPE : (tensor<1024x3xf32>, tensor<1024x3xf32>) -> tensor<1024x3xi1>
    stablehlo.custom_call @shape_assertion(%7) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<1024x3xi1>) -> ()
    %8 = stablehlo.constant dense<2.000000e+03> : tensor<1024x3xf32>
    %9:4 = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1024x1024xf32>) -> (tensor<1024x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %10 = "tf.XlaCallModule"(%9#0, %8) {Sout = [#tf_type.shape<1024x3>], _entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "{{.*}}", _tfl_quant_trait = "fully_quantizable", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1024x1024xf32>, tensor<1024x3xf32>) -> tensor<1024x3xf32>
    %11:4 = "tf.CustomAggregator"(%10) {calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<1024x3xf32>) -> (tensor<1024x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
    %12 = stablehlo.add %11#0, %0 : tensor<1024x3xf32>
    return %12 : tensor<1024x3xf32>
  }
  // CHECK: %[[SUBGRAPH_0:.*]]:2 = "tf.XlaCallModule"() <{Sout = [#tf_type.shape<1024x3>, #tf_type.shape<1024x3>], {{.*}} ["CPU", "TPU"], {{.*}}}> {_entry_function = @_stablehlo_main_1
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_0]], %[[SUBGRAPH_0]]#1) <{Sout = [#tf_type.shape<1024x3>], {{.*}}}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_version = "{{.*}}"
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[XLA_CALL_MODULE:.*]])
  // CHECK: %[[SUBGRAPH_1:.*]] = "tf.XlaCallModule"(%[[CUSTOM_AGGREGATOR_1]], %[[SUBGRAPH_0]]#0) <{Sout = [#tf_type.shape<1024x3>], {{.*}}}> {_entry_function = @_stablehlo_main_0
  // CHECK: return %[[SUBGRAPH_1]] : tensor<1024x3xf32>
  // CHECK: }
}

// -----

// main function contains PartitionedCall and StatefulPartitionedCall ops which
// is used to preserve aliased functions. This test make sure stablehlo ops in
// each PartitionedCall functions are lifted.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1629 : i32}, tf_saved_model.semantics} {
  // CHECK: func private @_stablehlo_main_2
  // CHECK: stablehlo.multiply %arg1, %arg2 : tensor<3x3xf32>
  // CHECK: return
  // CHECK: }

  // CHECK: func private @_stablehlo_main_1
  // CHECK: stablehlo.add %arg1, %arg2 : tensor<3x3xf32>
  // CHECK: return
  // CHECK: }

  // CHECK: func private @_stablehlo_main_0
  // CHECK: stablehlo.constant dense<1.000000e+03> : tensor<3x3xf32>
  // CHECK: stablehlo.constant dense<2.000000e+03> : tensor<3x3xf32>
  // CHECK: return
  // CHECK: }

  func.func @main() -> (tensor<3x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.constant dense<1.000000e+03> : tensor<3x3xf32>
    %1 = stablehlo.constant dense<2.000000e+03> : tensor<3x3xf32>
    %2 = "tf.StatefulPartitionedCall"(%0, %1) <{
      config = "", config_proto = "", executor_type = "", f = @some_func
    }> {
      _collective_manager_ids = [], device = ""
    } : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %3 = "tf.PartitionedCall"(%2, %1) <{
      config = "", config_proto = "", executor_type = "", f = @some_other_func
    }> {
      _collective_manager_ids = [], device = ""
    } : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    return %3 : tensor<3x3xf32>
  }
  // CHECK: func.func @main
  // CHECK: %[[INPUT:.*]]:3 = "tf.XlaCallModule"()
  // CHECK-SAME: _entry_function = @_stablehlo_main_0
  // CHECK: %[[ADD:.*]] = "tf.StatefulPartitionedCall"(%[[INPUT]]#1, %[[INPUT]]#2)
  // CHECK-SAME: f = @some_func
  // CHECK: "tf.PartitionedCall"(%[[ADD]], %[[INPUT]]#0)
  // CHECK-SAME: f = @some_other_func
  // CHECK: return

  func.func private @some_func(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes {tf._noinline = true} {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
  // CHECK: func.func private @some_func
  // CHECK: tf.XlaCallModule
  // CHECK-SAME: _entry_function = @_stablehlo_main_1
  // CHECK: return

  func.func private @some_other_func(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes {tf._noinline = true} {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
  // CHECK: func.func private @some_other_func
  // CHECK: tf.XlaCallModule
  // CHECK-SAME: _entry_function = @_stablehlo_main_2
  // CHECK: return
}
