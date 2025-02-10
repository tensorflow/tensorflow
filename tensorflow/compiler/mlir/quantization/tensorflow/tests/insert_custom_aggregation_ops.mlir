// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=MIN_MAX' -split-input-file | FileCheck --check-prefix=MIN-MAX-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=AVERAGE_MIN_MAX'  -split-input-file | FileCheck --check-prefix=AVERAGE-MIN-MAX-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_PERCENTILE' -split-input-file | FileCheck --check-prefix=HISTOGRAM-PERCENTILE-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_MSE_BRUTEFORCE' -split-input-file | FileCheck --check-prefix=HISTOGRAM-MSE-BRUTEFORCE-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_MSE_MAX_FREQUENCY' -split-input-file | FileCheck --check-prefix=HISTOGRAM-MSE-MAX-FREQUENCY-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_MSE_SYMMETRIC' -split-input-file | FileCheck --check-prefix=HISTOGRAM-MSE-SYMMETRIC-CHECK %s

module {
  func.func @wrap_composite_func(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %0 = "tf.PartitionedCall"(%arg0, %arg1) <{f = @composite_conv2d_with_relu6_fn}> {_tfl_quant_trait = "fully_quantizable"}
          : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %0 : tensor<*xf32>
  }

  func.func @no_composite_func(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %add : tensor<*xf32>
  }

  func.func @composite_conv2d_with_relu6_fn(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.Relu6"(%0) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %1 : tensor<*xf32>
  }
}

// CalibrationOptions(calibration_method=CALIBRATION_METHOD_MIN_MAX)
// MIN-MAX-CHECK: func @wrap_composite_func
// MIN-MAX-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 1 : i32, id = "composite_conv2d_with_relu6_fn_arg_1_calibration_method_1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// MIN-MAX-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "composite_conv2d_with_relu6_fn_arg_0_calibration_method_1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// MIN-MAX-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// MIN-MAX-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 1 : i32, id = "composite_conv2d_with_relu6_fn_calibration_method_1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// MIN-MAX-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// MIN-MAX-CHECK: func @no_composite_func
// MIN-MAX-CHECK-NEXT:  "tf.AddV2"
// MIN-MAX-CHECK-NEXT:  return

// MIN-MAX-CHECK: func @composite_conv2d_with_relu6_fn
// MIN-MAX-CHECK-NEXT:  "tf.Conv2D"
// MIN-MAX-CHECK-NEXT:  "tf.Relu6"
// MIN-MAX-CHECK-NEXT:  return

// CalibrationOptions(calibration_method=CALIBRATION_METHOD_AVERAGE_MIN_MAX)
// AVERAGE-MIN-MAX-CHECK: func @wrap_composite_func
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 2 : i32, id = "composite_conv2d_with_relu6_fn_arg_1_calibration_method_2", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 2 : i32, id = "composite_conv2d_with_relu6_fn_arg_0_calibration_method_2", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 2 : i32, id = "composite_conv2d_with_relu6_fn_calibration_method_2", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// AVERAGE-MIN-MAX-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// AVERAGE-MIN-MAX-CHECK: func @no_composite_func
// AVERAGE-MIN-MAX-CHECK-NEXT:  "tf.AddV2"
// AVERAGE-MIN-MAX-CHECK-NEXT:  return

// AVERAGE-MIN-MAX-CHECK: func @composite_conv2d_with_relu6_fn
// AVERAGE-MIN-MAX-CHECK-NEXT:  "tf.Conv2D"
// AVERAGE-MIN-MAX-CHECK-NEXT:  "tf.Relu6"
// AVERAGE-MIN-MAX-CHECK-NEXT:  return

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
//   calibration_parameters=CalibrationParameters(num_bins=256, min_percentile=0.001, max_percentile=99.999)
// )
// HISTOGRAM-PERCENTILE-CHECK: func @wrap_composite_func
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 3 : i32, id = "composite_conv2d_with_relu6_fn_arg_1_calibration_method_3", max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 3 : i32, id = "composite_conv2d_with_relu6_fn_arg_0_calibration_method_3", max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 3 : i32, id = "composite_conv2d_with_relu6_fn_calibration_method_3", max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// HISTOGRAM-PERCENTILE-CHECK: func @no_composite_func
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  "tf.AddV2"
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  return

// HISTOGRAM-PERCENTILE-CHECK: func @composite_conv2d_with_relu6_fn
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  "tf.Conv2D"
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  "tf.Relu6"
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  return

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
//   calibration_parameters=CalibrationParameters(num_bins=256)
// )
// HISTOGRAM-MSE-BRUTEFORCE-CHECK: func @wrap_composite_func
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 4 : i32, id = "composite_conv2d_with_relu6_fn_arg_1_calibration_method_4", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 4 : i32, id = "composite_conv2d_with_relu6_fn_arg_0_calibration_method_4", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 4 : i32, id = "composite_conv2d_with_relu6_fn_calibration_method_4", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// HISTOGRAM-MSE-BRUTEFORCE-CHECK: func @no_composite_func
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  "tf.AddV2"
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  return

// HISTOGRAM-MSE-BRUTEFORCE-CHECK: func @composite_conv2d_with_relu6_fn
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  "tf.Conv2D"
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  "tf.Relu6"
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  return

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
//   calibration_parameters=CalibrationParameters(num_bins=256)
// )
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK: func @wrap_composite_func
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 5 : i32, id = "composite_conv2d_with_relu6_fn_arg_1_calibration_method_5", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 5 : i32, id = "composite_conv2d_with_relu6_fn_arg_0_calibration_method_5", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 5 : i32, id = "composite_conv2d_with_relu6_fn_calibration_method_5", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK: func @no_composite_func
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  "tf.AddV2"
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  return

// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK: func @composite_conv2d_with_relu6_fn
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  "tf.Conv2D"
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  "tf.Relu6"
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  return

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
//   calibration_parameters=CalibrationParameters(num_bins=256)
// )
// HISTOGRAM-MSE-SYMMETRIC-CHECK: func @wrap_composite_func
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 6 : i32, id = "composite_conv2d_with_relu6_fn_arg_1_calibration_method_6", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 6 : i32, id = "composite_conv2d_with_relu6_fn_arg_0_calibration_method_6", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 6 : i32, id = "composite_conv2d_with_relu6_fn_calibration_method_6", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// HISTOGRAM-MSE-SYMMETRIC-CHECK: func @no_composite_func
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  "tf.AddV2"
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  return

// HISTOGRAM-MSE-SYMMETRIC-CHECK: func @composite_conv2d_with_relu6_fn
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  "tf.Conv2D"
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  "tf.Relu6"
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  return


// -----

module {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<?x100352xf32>, %arg1: tensor<100352x10xf32>) -> tensor<?x10xf32> {
    // MIN-MAX-CHECK-DAG: %[[ARG0_ID:.*]] = "tf.Identity"(%arg0)
    // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[ARG0_ID]])
    // MIN-MAX-CHECK-SAME: id = "composite_dot_general_fn_1_arg_0_calibration_method_1"
    // MIN-MAX-CHECK-DAG: %[[ARG1_ID:.*]] = "tf.Identity"(%arg1)
    // MIN-MAX-CHECK: %[[ARG1_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[ARG1_ID]])
    // MIN-MAX-CHECK-SAME: id = "composite_dot_general_fn_1_arg_1_calibration_method_1"
    // MIN-MAX-CHECK: %[[RES:.*]] = "tf.XlaCallModule"(%[[ARG0_AGG]], %[[ARG1_AGG]])
    // MIN-MAX-CHECK: %[[RES_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%[[RES]])
    // MIN-MAX-CHECK-SAME: id = "composite_dot_general_fn_1_calibration_method_1"
    // MIN-MAX-CHECK: %[[RES_ID:.*]] = "tf.Identity"(%[[RES_AGG]])
    // MIN-MAX-CHECK: return %[[RES_ID]] : tensor<?x10xf32>
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<?x100352xf32>) -> tensor<?x100352xf32>
    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<100352x10xf32>) -> tensor<100352x10xf32>
    %2 = "tf.XlaCallModule"(%0, %1) <{
        Sout = [#tf_type.shape<?x10>], dim_args_spec = [],
        disabled_checks = [], function_list = [],
        has_token_input_output = false, module = "", platforms = [],
        version = 5 : i64
    }> {
        _entry_function = @composite_dot_general_fn_1,
        _stablehlo_version = "1.0.0",
        _original_entry_function = "composite_dot_general_fn_1",
        _tfl_quant_trait = "fully_quantizable",
        _quantization_method = "static_range_ptq { }"
    } : (tensor<?x100352xf32>, tensor<100352x10xf32>) -> tensor<?x10xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<?x10xf32>) -> tensor<?x10xf32>
    return %3 : tensor<?x10xf32>
  }

  // CHECK-LABEL: func.func private @composite_dot_general_fn_1
  func.func private @composite_dot_general_fn_1(%arg0: tensor<?x100352xf32>, %arg1: tensor<100352x10xf32>) -> tensor<?x10xf32> {
    // CHECK-NOT: tf.CustomAggregator
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x100352xf32>, tensor<100352x10xf32>) -> tensor<?x10xf32>
    return %0 : tensor<?x10xf32>
  }
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1833 : i32}, tf_saved_model.semantics} {
  func.func @serving_default(%arg0: tensor<1x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() <{value = dense<[0, 1]> : tensor<2xi32>}> {device = ""} : () -> tensor<2xi32>
    %cst_0 = "tf.Const"() <{value = dense<1.000000e+01> : tensor<f32>}> {device = ""} : () -> tensor<f32>
    %0 = "tf.Sum"(%arg0, %cst) <{keep_dims = false}> {device = ""} : (tensor<1x4xf32>, tensor<2xi32>) -> tensor<f32>
    %1 = "tf.Greater"(%0, %cst_0) {device = ""} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2:2 = "tf.If"(%1, %arg0) <{else_branch = @cond_false_80, is_stateless = true, then_branch = @cond_true_70}> {Tcond = i1, Tin = [f32], Tout = [i1, f32], _lower_using_switch_merge = true, _read_only_resource_inputs = [], device = ""} : (tensor<i1>, tensor<1x4xf32>) -> (tensor<i1>, tensor<1x3xf32>)
    %3 = "tf.Identity"(%2#1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }


  func.func private @cond_false_80(%arg0: tensor<1x4xf32> {tf._user_specified_name = "x"}) -> (tensor<i1>, tensor<1x3xf32>) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4>], tf._original_func_name = "cond_false_8"} {
    %cst = "tf.Const"() <{value = dense<true> : tensor<i1>}> {device = ""} : () -> tensor<i1>
    %cst_0 = "tf.Const"() <{value = dense<[0.117216609, 0.933735609, 0.0728900209]> : tensor<3xf32>}> {device = ""} : () -> tensor<3xf32>
    %cst_1 = "tf.Const"() <{value = dense<[[-0.795477629, 0.581315517, 0.921566545], [0.138622552, 0.463866323, 0.95474267], [-0.143770888, -0.796835303, 0.899996876], [0.0989735424, -0.483384758, -7.277030e-01]]> : tensor<4x3xf32>}> {device = ""} : () -> tensor<4x3xf32>
    %0 = "tf.Identity"(%cst) {device = ""} : (tensor<i1>) -> tensor<i1>
    %1 = "tf.PartitionedCall"(%arg0, %cst_1, %cst_0) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_with_bias_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    %2 = "tf.Identity"(%1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0, %2 : tensor<i1>, tensor<1x3xf32>
  }
  // MIN-MAX-CHECK: func.func private @cond_false_80
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_matmul_with_bias_fn_1_arg_0_calibration_method_1"
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_matmul_with_bias_fn_1_calibration_method_1"

  func.func private @cond_true_70(%arg0: tensor<1x4xf32> {tf._user_specified_name = "x"}) -> (tensor<i1>, tensor<1x3xf32>) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x4>], tf._original_func_name = "cond_true_7"} {
    %cst = "tf.Const"() <{value = dense<true> : tensor<i1>}> {device = ""} : () -> tensor<i1>
    %cst_0 = "tf.Const"() <{value = dense<[0.335351914, 0.084816426, -0.664676845]> : tensor<3xf32>}> {device = ""} : () -> tensor<3xf32>
    %cst_1 = "tf.Const"() <{value = dense<[[-0.630731344, 0.54962182, 0.180364341], [-0.764542698, -0.211145893, -0.708605706], [-0.954062759, -0.614013135, 0.612640202], [-0.418223292, 5.057390e-01, 0.899269938]]> : tensor<4x3xf32>}> {device = ""} : () -> tensor<4x3xf32>
    %0 = "tf.Identity"(%cst) {device = ""} : (tensor<i1>) -> tensor<i1>
    %1 = "tf.PartitionedCall"(%arg0, %cst_1, %cst_0) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_with_bias_fn_2}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    %2 = "tf.Identity"(%1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0, %2 : tensor<i1>, tensor<1x3xf32>
  }
  // MIN-MAX-CHECK: func.func private @cond_true_70
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_matmul_with_bias_fn_2_arg_0_calibration_method_1"
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_matmul_with_bias_fn_2_calibration_method_1"

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

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1833 : i32}, tf_saved_model.semantics} {
  func.func @serving_default(%arg0: tensor<1x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() <{value = dense<1.000000e+01> : tensor<f32>}> {device = ""} : () -> tensor<f32>
    %cst_0 = "tf.Const"() <{value = dense<[0, 1]> : tensor<2xi32>}> {device = ""} : () -> tensor<2xi32>
    %cst_1 = "tf.Const"() <{value = dense<[[-0.630731344, 0.54962182, 0.180364341], [-0.764542698, -0.211145893, -0.708605706], [-0.954062759, -0.614013135, 0.612640202], [-0.418223292, 5.057390e-01, 0.899269938]]> : tensor<4x3xf32>}> {device = ""} : () -> tensor<4x3xf32>
    %cst_2 = "tf.Const"() <{value = dense<[0.335351914, 0.084816426, -0.664676845]> : tensor<3xf32>}> {device = ""} : () -> tensor<3xf32>
    %cst_3 = "tf.Const"() <{value = dense<true> : tensor<i1>}> {device = ""} : () -> tensor<i1>
    %cst_4 = "tf.Const"() <{value = dense<[[-0.795477629, 0.581315517, 0.921566545], [0.138622552, 0.463866323, 0.95474267], [-0.143770888, -0.796835303, 0.899996876], [0.0989735424, -0.483384758, -7.277030e-01]]> : tensor<4x3xf32>}> {device = ""} : () -> tensor<4x3xf32>
    %cst_5 = "tf.Const"() <{value = dense<[0.117216609, 0.933735609, 0.0728900209]> : tensor<3xf32>}> {device = ""} : () -> tensor<3xf32>
    %0 = "tf.Sum"(%arg0, %cst_0) <{keep_dims = false}> {device = ""} : (tensor<1x4xf32>, tensor<2xi32>) -> tensor<f32>
    %1 = "tf.Greater"(%0, %cst) {device = ""} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2:2 = "tf.IfRegion"(%1) <{_else_func_name = "cond_false_80", _then_func_name = "cond_true_70", is_stateless = true}> ({
      %4 = "tf.Identity"(%cst_3) {device = ""} : (tensor<i1>) -> tensor<i1>
      %5 = "tf.PartitionedCall"(%arg0, %cst_1, %cst_2) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_with_bias_fn_2}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
      %6 = "tf.Identity"(%5) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
      "tf.Yield"(%4, %6) {device = ""} : (tensor<i1>, tensor<1x3xf32>) -> ()
    }, {
      %4 = "tf.Identity"(%cst_3) {device = ""} : (tensor<i1>) -> tensor<i1>
      %5 = "tf.PartitionedCall"(%arg0, %cst_4, %cst_5) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_with_bias_fn_1}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
      %6 = "tf.Identity"(%5) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
      "tf.Yield"(%4, %6) {device = ""} : (tensor<i1>, tensor<1x3xf32>) -> ()
    }) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = ""} : (tensor<i1>) -> (tensor<i1>, tensor<1x3xf32>)
    %3 = "tf.Identity"(%2#1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }
  // MIN-MAX-CHECK: func.func @serving_default
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_matmul_with_bias_fn_1_arg_0_calibration_method_1"
  // MIN-MAX-CHECK: "tf.IfRegion"
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_matmul_with_bias_fn_2_calibration_method_1"
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_matmul_with_bias_fn_1_calibration_method_1"

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
    %0 = "tf.XlaCallModule"(%arg0, %cst) <{Sout = [#tf_type.shape<10x1x3>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_with_relu_fn_1, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_with_relu_fn_1", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<10x1x1024xf32>, tensor<10x1024x3xf32>) -> tensor<10x1x3xf32>
    return %0 : tensor<10x1x3xf32>
  }
  // MIN-MAX-CHECK: func.func @main
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_dot_general_with_relu_fn_1_arg_0_calibration_method_1"
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_dot_general_with_relu_fn_1_calibration_method_1"

  func.func private @composite_dot_general_with_relu_fn_1(%arg0: tensor<10x1x1024xf32>, %arg1: tensor<10x1024x3xf32>) -> tensor<10x1x3xf32> attributes {_from_xla_call_module, tf_quant.composite_function} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x1x3xf32>
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<10x1x1024xf32>, tensor<10x1024x3xf32>) -> tensor<10x1x3xf32>
    %1 = stablehlo.maximum %0, %cst : tensor<10x1x3xf32>
    return %1 : tensor<10x1x3xf32>
  }
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1836 : i32}, tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %cst_0 = stablehlo.constant dense<[[-0.630731344, 0.54962182, 0.180364341], [-0.764542698, -0.211145893, -0.708605706], [-0.954062759, -0.614013135, 0.612640202], [-0.418223292, 5.057390e-01, 0.899269938]]> : tensor<4x3xf32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %cst_1 = stablehlo.constant dense<[[-0.795477629, 0.581315517, 0.921566545], [0.138622552, 0.463866323, 0.95474267], [-0.143770888, -0.796835303, 0.899996876], [0.0989735424, -0.483384758, -7.277030e-01]]> : tensor<4x3xf32>
    %cst_2 = stablehlo.constant dense<-0.000000e+00> : tensor<f32>
    %cst_3 = stablehlo.constant dense<[[0.335351914, 0.084816426, -0.664676845]]> : tensor<1x3xf32>
    %cst_4 = stablehlo.constant dense<[[0.117216609, 0.933735609, 0.0728900209]]> : tensor<1x3xf32>
    %0 = stablehlo.reduce(%arg0 init: %cst_2) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x4xf32>, tensor<f32>) -> tensor<f32>
    %1 = stablehlo.compare  GT, %0, %cst : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2:2 = "stablehlo.if"(%1) ({
      %3 = "tf.XlaCallModule"(%arg0, %cst_0, %cst_3) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_same_shape_fn_2, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_with_bias_same_shape_fn_2", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
      stablehlo.return %c, %3 : tensor<i1>, tensor<1x3xf32>
    }, {
      %3 = "tf.XlaCallModule"(%arg0, %cst_1, %cst_4) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @composite_dot_general_with_bias_same_shape_fn_1, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_with_bias_same_shape_fn_1", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {jax.uses_shape_polymorphism = true}, _tfl_quant_trait = "fully_quantizable"} : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
      stablehlo.return %c, %3 : tensor<i1>, tensor<1x3xf32>
    }) : (tensor<i1>) -> (tensor<i1>, tensor<1x3xf32>)
    return %2#1 : tensor<1x3xf32>
  }
  // MIN-MAX-CHECK: func.func @main
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_dot_general_with_bias_same_shape_fn_1_arg_0_calibration_method_1"
  // MIN-MAX-CHECK: "stablehlo.if"
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_dot_general_with_bias_same_shape_fn_2_calibration_method_1"
  // MIN-MAX-CHECK: %[[ARG0_AGG:.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"
  // MIN-MAX-CHECK-SAME: id = "composite_dot_general_with_bias_same_shape_fn_1_calibration_method_1"

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
