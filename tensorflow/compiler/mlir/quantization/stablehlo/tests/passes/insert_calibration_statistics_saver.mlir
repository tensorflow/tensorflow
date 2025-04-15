// RUN: stablehlo-quant-opt %s -split-input-file -mlir-disable-threading -stablehlo-insert-calibration-statistics-saver | FileCheck %s

func.func @serving_default(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x2x2x2xf32>) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}} {
  %cst = "tf.Const"() <{value = dense<[[[[-0.891899645, 0.392044574], [0.77720493, 1.31188095], [0.255048186, 2.700150e+00]], [[-1.08111858, -0.406604826], [-0.298575521, -2.25356531], [-1.00201964, 2.54532099]], [[-1.34911358, 0.279911458], [-0.868258893, -1.36708188], [0.866317451, -2.05804896]]], [[[-0.591397941, 0.331505477], [0.715151429, 2.64073896], [1.27163255, 0.206143498]], [[0.474211812, 1.45044816], [0.119936548, 2.54149938], [-0.939900994, 0.438387245]], [[-1.12486279, -1.09022558], [0.82202208, 1.04652023], [1.30316162, 2.62054276]]]]> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2xf32>
  %output, %min, %max, %histogram = "tf.CustomAggregator"(%arg0) <{calibration_method = 5 : i32, id = "0", num_bins = 32 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> : (tensor<1x3x4x3xf32>) -> (tensor<1x3x4x3xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
  %0 = "tf.Conv2D"(%output, %cst) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true}> {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
  %output_1, %min_2, %max_3, %histogram_4 = "tf.CustomAggregator"(%0) <{calibration_method = 5 : i32, id = "1", num_bins = 32 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> : (tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
  %1 = "tf.Identity"(%output_1) {device = ""} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %1 : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: @serving_default
// CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], %[[MIN_O:.*]], %[[MAX_O:.*]], %[[HISTOGRAM_0:.*]] = "tf.CustomAggregator"
// CKECK-SAME: <{calibration_method = 5 : i32, id = "0", num_bins = 32 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> : (tensor<1x3x4x3xf32>) -> (tensor<1x3x4x3xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], %[[MIN_1:.*]], %[[MAX_1:.*]], %[[HISTOGRAM_1:.*]] = "tf.CustomAggregator"
// CKECK-SAME: <{calibration_method = 5 : i32, id = "1", num_bins = 32 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> : (tensor<1x3x4x3xf32>) -> (tensor<1x3x4x3xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// CHECK: "tf.CalibrationStatisticsSaver"(%[[MIN_O]], %[[MAX_O]], %[[HISTOGRAM_0]], %[[MIN_1]], %[[MAX_1]], %[[HISTOGRAM_1]])
// CHECK-SAME: <{calibration_methods = [5 : i32, 5 : i32], ids = ["0", "1"], output_file_path = "serving_default_0.pb"}>  : (tensor<f32>, tensor<f32>, tensor<512xi64>, tensor<f32>, tensor<f32>, tensor<512xi64>) -> ()
// CHECK: return

// -----

// No CustomAggregator ops exist.
func.func private @composite_conv2d_with_bias_and_relu6_fn_1(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<1x2x2x2xf32> attributes {tf_quant.composite_function} {
  %0 = "tf.Conv2D"(%arg0, %arg1) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true}> {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x2x2x2xf32>
  %1 = "tf.BiasAdd"(%0, %arg2) <{data_format = "NHWC"}> : (tensor<1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
  %2 = "tf.Relu6"(%1) {device = ""} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %2 : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: @composite_conv2d_with_bias_and_relu6_fn_1
// CHECK-NOT: "tf.CalibrationStatisticsSaver"

// -----

// Check the IfOp is set to stateful.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1833 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: func.func @serving_default
  // CHECK: "tf.If"
  // CHECK-SAME: is_stateless = false
  func.func @serving_default(%arg0: tensor<1x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() <{value = dense<[0, 1]> : tensor<2xi32>}> {device = ""} : () -> tensor<2xi32>
    %cst_0 = "tf.Const"() <{value = dense<1.000000e+01> : tensor<f32>}> {device = ""} : () -> tensor<f32>
    %0 = "tf.Sum"(%arg0, %cst) <{keep_dims = false}> {device = ""} : (tensor<1x4xf32>, tensor<2xi32>) -> tensor<f32>
    %1 = "tf.Greater"(%0, %cst_0) {device = ""} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2:2 = "tf.If"(%1, %arg0) <{else_branch = @cond_false_80, is_stateless = true, then_branch = @cond_true_70}> {Tcond = i1, Tin = [f32], Tout = [i1, f32], _lower_using_switch_merge = true, _read_only_resource_inputs = [], device = ""} : (tensor<i1>, tensor<1x4xf32>) -> (tensor<i1>, tensor<1x3xf32>)
    %3 = "tf.Identity"(%2#1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @cond_false_80
  // CHECK: "tf.CalibrationStatisticsSaver"
  // CHECK-SAME: output_file_path = "cond_false_80_0.pb"
  func.func private @cond_false_80(%arg0: tensor<1x4xf32> {tf._user_specified_name = "x"}) -> (tensor<i1>, tensor<1x3xf32>) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4>], tf._original_func_name = "cond_false_8"} {
    %cst = "tf.Const"() <{value = dense<true> : tensor<i1>}> {device = ""} : () -> tensor<i1>
    %cst_0 = "tf.Const"() <{value = dense<[0.117216609, 0.933735609, 0.0728900209]> : tensor<3xf32>}> {device = ""} : () -> tensor<3xf32>
    %cst_1 = "tf.Const"() <{value = dense<[[-0.795477629, 0.581315517, 0.921566545], [0.138622552, 0.463866323, 0.95474267], [-0.143770888, -0.796835303, 0.899996876], [0.0989735424, -0.483384758, -7.277030e-01]]> : tensor<4x3xf32>}> {device = ""} : () -> tensor<4x3xf32>
    %output, %min, %max, %histogram = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    %0 = "tf.Identity"(%cst) {device = ""} : (tensor<i1>) -> tensor<i1>
    %1 = "tf.PartitionedCall"(%output, %cst_1, %cst_0) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_with_bias_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    %output_2, %min_3, %max_4, %histogram_5 = "tf.CustomAggregator"(%1) <{calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    %2 = "tf.Identity"(%output_2) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0, %2 : tensor<i1>, tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @cond_true_70
  // CHECK: "tf.CalibrationStatisticsSaver"
  // CHECK-SAME: output_file_path = "cond_true_70_0.pb"
  func.func private @cond_true_70(%arg0: tensor<1x4xf32> {tf._user_specified_name = "x"}) -> (tensor<i1>, tensor<1x3xf32>) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4>], tf._original_func_name = "cond_true_7"} {
    %cst = "tf.Const"() <{value = dense<true> : tensor<i1>}> {device = ""} : () -> tensor<i1>
    %cst_0 = "tf.Const"() <{value = dense<[0.335351914, 0.084816426, -0.664676845]> : tensor<3xf32>}> {device = ""} : () -> tensor<3xf32>
    %cst_1 = "tf.Const"() <{value = dense<[[-0.630731344, 0.54962182, 0.180364341], [-0.764542698, -0.211145893, -0.708605706], [-0.954062759, -0.614013135, 0.612640202], [-0.418223292, 5.057390e-01, 0.899269938]]> : tensor<4x3xf32>}> {device = ""} : () -> tensor<4x3xf32>
    %output, %min, %max, %histogram = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "2", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    %0 = "tf.Identity"(%cst) {device = ""} : (tensor<i1>) -> tensor<i1>
    %1 = "tf.PartitionedCall"(%output, %cst_1, %cst_0) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_with_bias_fn_2}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    %output_2, %min_3, %max_4, %histogram_5 = "tf.CustomAggregator"(%1) <{calibration_method = 1 : i32, id = "3", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    %2 = "tf.Identity"(%output_2) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0, %2 : tensor<i1>, tensor<1x3xf32>
  }

  func.func private @composite_matmul_with_bias_fn_1(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<1x3xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_b", device = ""} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) <{data_format = "NHWC"}> {device = ""} : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }

  func.func private @composite_matmul_with_bias_fn_2(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<1x3xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_b", device = ""} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) <{data_format = "NHWC"}> {device = ""} : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
}

// -----

// Check the IfRegion is set to stateful.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1833 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: func.func @serving_default
  // CHECK: "tf.IfRegion"
  // CHECK-SAME: is_stateless = false

  // CHECK: "tf.CalibrationStatisticsSaver"
  // CHECK-SAME: output_file_path = "serving_default_0.pb"

  // CHECK: "tf.CalibrationStatisticsSaver"
  // CHECK-SAME: output_file_path = "serving_default_1.pb"

  // CHECK: "tf.CalibrationStatisticsSaver"
  // CHECK-SAME: output_file_path = "serving_default_2.pb"
  func.func @serving_default(%arg0: tensor<1x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() <{value = dense<1.000000e+01> : tensor<f32>}> {device = ""} : () -> tensor<f32>
    %cst_0 = "tf.Const"() <{value = dense<[0, 1]> : tensor<2xi32>}> {device = ""} : () -> tensor<2xi32>
    %cst_1 = "tf.Const"() <{value = dense<[[-0.630731344, 0.54962182, 0.180364341], [-0.764542698, -0.211145893, -0.708605706], [-0.954062759, -0.614013135, 0.612640202], [-0.418223292, 5.057390e-01, 0.899269938]]> : tensor<4x3xf32>}> {device = ""} : () -> tensor<4x3xf32>
    %cst_2 = "tf.Const"() <{value = dense<[0.335351914, 0.084816426, -0.664676845]> : tensor<3xf32>}> {device = ""} : () -> tensor<3xf32>
    %cst_3 = "tf.Const"() <{value = dense<true> : tensor<i1>}> {device = ""} : () -> tensor<i1>
    %cst_4 = "tf.Const"() <{value = dense<[[-0.795477629, 0.581315517, 0.921566545], [0.138622552, 0.463866323, 0.95474267], [-0.143770888, -0.796835303, 0.899996876], [0.0989735424, -0.483384758, -7.277030e-01]]> : tensor<4x3xf32>}> {device = ""} : () -> tensor<4x3xf32>
    %cst_5 = "tf.Const"() <{value = dense<[0.117216609, 0.933735609, 0.0728900209]> : tensor<3xf32>}> {device = ""} : () -> tensor<3xf32>
    %output, %min, %max, %histogram = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    %0 = "tf.Sum"(%output, %cst_0) <{keep_dims = false}> {device = ""} : (tensor<1x4xf32>, tensor<2xi32>) -> tensor<f32>
    %1 = "tf.Greater"(%0, %cst) {device = ""} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2:2 = "tf.IfRegion"(%1) <{_else_func_name = "cond_false_80", _then_func_name = "cond_true_70", is_stateless = true}> ({
      %4 = "tf.Identity"(%cst_3) {device = ""} : (tensor<i1>) -> tensor<i1>
      %5 = "tf.PartitionedCall"(%output, %cst_1, %cst_2) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_with_bias_fn_2}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
      %output_6, %min_7, %max_8, %histogram_9 = "tf.CustomAggregator"(%5) <{calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
      %6 = "tf.Identity"(%output_6) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
      "tf.Yield"(%4, %6) {device = ""} : (tensor<i1>, tensor<1x3xf32>) -> ()
    }, {
      %4 = "tf.Identity"(%cst_3) {device = ""} : (tensor<i1>) -> tensor<i1>
      %5 = "tf.PartitionedCall"(%output, %cst_4, %cst_5) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_with_bias_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
      %output_6, %min_7, %max_8, %histogram_9 = "tf.CustomAggregator"(%5) <{calibration_method = 1 : i32, id = "2", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
      %6 = "tf.Identity"(%output_6) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
      "tf.Yield"(%4, %6) {device = ""} : (tensor<i1>, tensor<1x3xf32>) -> ()
    }) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = ""} : (tensor<i1>) -> (tensor<i1>, tensor<1x3xf32>)
    %3 = "tf.Identity"(%2#1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }
  func.func private @composite_matmul_with_bias_fn_2(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<1x3xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_b", device = ""} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) <{data_format = "NHWC"}> {device = ""} : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
  func.func private @composite_matmul_with_bias_fn_1(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<1x3xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_b", device = ""} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) <{data_format = "NHWC"}> {device = ""} : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1836 : i32}, tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<10x1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<10x1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = stablehlo.constant dense<0.000000e+00>: tensor<10x1024x3xf32>
    %output, %min, %max, %histogram = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<10x1x1024xf32>) -> (tensor<10x1x1024xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    %0 = "tf.XlaCallModule"(%output, %cst) <{Sout = [#tf_type.shape<10x1x3>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_with_relu_fn_1, _original_entry_function = "composite_dot_general_with_relu_fn_1", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<10x1x1024xf32>, tensor<10x1024x3xf32>) -> tensor<10x1x3xf32>
    %output_0, %min_1, %max_2, %histogram_3 = "tf.CustomAggregator"(%0) <{calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<10x1x3xf32>) -> (tensor<10x1x3xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    return %output_0 : tensor<10x1x3xf32>
  }
  // CHECK-LABEL: @main
  // CHECK: %[[CUSTOM_AGGREGATOR_0:.*]], %[[MIN_O:.*]], %[[MAX_O:.*]], %[[HISTOGRAM_0:.*]] = "tf.CustomAggregator"
  // CKECK-SAME: <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: %[[CUSTOM_AGGREGATOR_1:.*]], %[[MIN_1:.*]], %[[MAX_1:.*]], %[[HISTOGRAM_1:.*]] = "tf.CustomAggregator"
  // CKECK-SAME: <{calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}>
  // CHECK: "tf.CalibrationStatisticsSaver"(%[[MIN_O]], %[[MAX_O]], %[[HISTOGRAM_0]], %[[MIN_1]], %[[MAX_1]], %[[HISTOGRAM_1]])
  // CHECK-SAME: <{calibration_methods = [1 : i32, 1 : i32], ids = ["0", "1"], output_file_path = "main_0.pb"}> : (tensor<f32>, tensor<f32>, tensor<0xi64>, tensor<f32>, tensor<f32>, tensor<0xi64>) -> ()
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

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1836 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: func.func @main
  // CHECK: "tf.CalibrationStatisticsSaver"
  // CHECK-SAME: output_file_path = "main_0.pb"
  // CHECK: "tf.CalibrationStatisticsSaver"
  // CHECK-SAME: output_file_path = "main_1.pb"
  // CHECK: "tf.CalibrationStatisticsSaver"
  // CHECK-SAME: output_file_path = "main_2.pb"
  func.func @main(%arg0: tensor<1x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %cst_0 = stablehlo.constant dense<[[-0.630731344, 0.54962182, 0.180364341], [-0.764542698, -0.211145893, -0.708605706], [-0.954062759, -0.614013135, 0.612640202], [-0.418223292, 5.057390e-01, 0.899269938]]> : tensor<4x3xf32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %cst_1 = stablehlo.constant dense<[[-0.795477629, 0.581315517, 0.921566545], [0.138622552, 0.463866323, 0.95474267], [-0.143770888, -0.796835303, 0.899996876], [0.0989735424, -0.483384758, -7.277030e-01]]> : tensor<4x3xf32>
    %cst_2 = stablehlo.constant dense<-0.000000e+00> : tensor<f32>
    %cst_3 = stablehlo.constant dense<[[0.335351914, 0.084816426, -0.664676845]]> : tensor<1x3xf32>
    %cst_4 = stablehlo.constant dense<[[0.117216609, 0.933735609, 0.0728900209]]> : tensor<1x3xf32>
    %output, %min, %max, %histogram = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
    %0 = stablehlo.reduce(%output init: %cst_2) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<f32>
    %1 = stablehlo.compare  GT, %0, %cst : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2:2 = "stablehlo.if"(%1) ({
      %3 = "tf.XlaCallModule"(%output, %cst_0, %cst_3) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_same_shape_fn_2, _original_entry_function = "composite_dot_general_with_bias_same_shape_fn_2", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
      %output_5, %min_6, %max_7, %histogram_8 = "tf.CustomAggregator"(%3) <{calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
      stablehlo.return %c, %output_5 : tensor<i1>, tensor<1x3xf32>
    }, {
      %3 = "tf.XlaCallModule"(%output, %cst_1, %cst_4) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_same_shape_fn_1, _original_entry_function = "composite_dot_general_with_bias_same_shape_fn_1", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
      %output_5, %min_6, %max_7, %histogram_8 = "tf.CustomAggregator"(%3) <{calibration_method = 1 : i32, id = "2", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
      stablehlo.return %c, %output_5 : tensor<i1>, tensor<1x3xf32>
    }) : (tensor<i1>) -> (tensor<i1>, tensor<1x3xf32>)
    return %2#1 : tensor<1x3xf32>
  }
  func.func private @composite_dot_general_with_bias_same_shape_fn_2(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }

  func.func private @composite_dot_general_with_bias_same_shape_fn_1(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
}