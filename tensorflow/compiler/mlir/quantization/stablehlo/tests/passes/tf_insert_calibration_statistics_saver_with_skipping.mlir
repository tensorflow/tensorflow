// RUN: stablehlo-quant-opt %s -split-input-file -tf-stablehlo-insert-calibration-statistics-saver='aggregator-ops-to-ignore=skipping_id' | FileCheck %s

func.func @serving_default(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x2x2x2xf32>) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}} {
  %cst = "tf.Const"() <{value = dense<[[[[-0.891899645, 0.392044574], [0.77720493, 1.31188095], [0.255048186, 2.700150e+00]], [[-1.08111858, -0.406604826], [-0.298575521, -2.25356531], [-1.00201964, 2.54532099]], [[-1.34911358, 0.279911458], [-0.868258893, -1.36708188], [0.866317451, -2.05804896]]], [[[-0.591397941, 0.331505477], [0.715151429, 2.64073896], [1.27163255, 0.206143498]], [[0.474211812, 1.45044816], [0.119936548, 2.54149938], [-0.939900994, 0.438387245]], [[-1.12486279, -1.09022558], [0.82202208, 1.04652023], [1.30316162, 2.62054276]]]]> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
  %output, %min, %max, %histogram = "tf.CustomAggregator"(%arg0) <{calibration_method = 5 : i32, id = "skipping_id", num_bins = 32 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> : (tensor<1x3x4x3xf32>) -> (tensor<1x3x4x3xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
  %0 = "tf.Conv2D"(%output, %cst) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true}> {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
  %output_1, %min_2, %max_3, %histogram_4 = "tf.CustomAggregator"(%0) <{calibration_method = 5 : i32, id = "keeping_id", num_bins = 32 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> : (tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
  %1 = "tf.Identity"(%output_1) {device = ""} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %1 : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: @serving_default
// CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], %[[MIN_O:.*]], %[[MAX_O:.*]], %[[HISTOGRAM_0:.*]] = "tf.CustomAggregator"
// CKECK-SAME: <{calibration_method = 5 : i32, id = "skipping_id", num_bins = 32 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> : (tensor<1x3x4x3xf32>) -> (tensor<1x3x4x3xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], %[[MIN_1:.*]], %[[MAX_1:.*]], %[[HISTOGRAM_1:.*]] = "tf.CustomAggregator"
// CKECK-SAME: <{calibration_method = 5 : i32, id = "keeping_id", num_bins = 32 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> : (tensor<1x3x4x3xf32>) -> (tensor<1x3x4x3xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// CHECK: "tf.CalibrationStatisticsSaver"(%[[MIN_1]], %[[MAX_1]], %[[HISTOGRAM_1]])
// CHECK-SAME: <{calibration_methods = [5 : i32], ids = ["keeping_id"], output_file_path = "serving_default_0.pb"}>  : (tensor<f32>, tensor<f32>, tensor<512xi64>) -> ()
// CHECK: return

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1836 : i32}, tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<10x1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<10x1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = stablehlo.constant dense<0.000000e+00>: tensor<10x1024x3xf32>
    %output, %min, %max, %histogram = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "skipping_id", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<10x1x1024xf32>) -> (tensor<10x1x1024xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    %0 = "tf.XlaCallModule"(%output, %cst) <{Sout = [#tf_type.shape<10x1x3>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_with_relu_fn_1, _original_entry_function = "composite_dot_general_with_relu_fn_1", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<10x1x1024xf32>, tensor<10x1024x3xf32>) -> tensor<10x1x3xf32>
    %output_0, %min_1, %max_2, %histogram_3 = "tf.CustomAggregator"(%0) <{calibration_method = 1 : i32, id = "keeping_id", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<10x1x3xf32>) -> (tensor<10x1x3xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    return %output_0 : tensor<10x1x3xf32>
  }
  // CHECK-LABEL: @main
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], %[[MIN_O:.*]], %[[MAX_O:.*]], %[[HISTOGRAM_0:.*]] = "tf.CustomAggregator"
  // CKECK-SAME: <{calibration_method = 1 : i32, id = "skipping_id", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], %[[MIN_1:.*]], %[[MAX_1:.*]], %[[HISTOGRAM_1:.*]] = "tf.CustomAggregator"
  // CKECK-SAME: <{calibration_method = 1 : i32, id = "keeping_id", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: "tf.CalibrationStatisticsSaver"(%[[MIN_1]], %[[MAX_1]], %[[HISTOGRAM_1]])
  // CHECK-SAME: <{calibration_methods = [1 : i32], ids = ["keeping_id"], output_file_path = "main_0.pb"}> : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> ()
  // CHECK: return

  func.func private @composite_dot_general_with_relu_fn_1(%arg0: tensor<10x1x1024xf32>, %arg1: tensor<10x1024x3xf32>) -> tensor<10x1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x1x3xf32>
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<10x1x1024xf32>, tensor<10x1024x3xf32>) -> tensor<10x1x3xf32>
    %1 = stablehlo.maximum %0, %cst : tensor<10x1x3xf32>
    return %1 : tensor<10x1x3xf32>
  }
  // CHECK-LABEL: func.func private @composite_dot_general_with_relu_fn_1
  // CHECK-NOT: "tf.CalibrationStatisticsSaver"
}