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
// MIN-MAX-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 1 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// MIN-MAX-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// MIN-MAX-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// MIN-MAX-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 1 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
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
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 2 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 2 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 2 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<0xi64>)
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
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 3 : i32, id = "", max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 3 : i32, id = "", max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 3 : i32, id = "", max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
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
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 4 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 4 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 4 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
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
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 5 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 5 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 5 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
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
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 6 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 6 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[add:%.*]] = "tf.PartitionedCall"([[lhs]], [[rhs]])
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 6 : i32, id = "", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 512 : i32}> : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<512xi64>)
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
    // CHECK-DAG: %[[ARG0_ID:.*]] = "tf.Identity"(%arg0)
    // CHECK-DAG: %[[ARG1_ID:.*]] = "tf.Identity"(%arg1)
    // CHECK-DAG: %[[ARG0_AGG:.*]] = "tf.CustomAggregator"(%[[ARG0_ID]])
    // CHECK-DAG: %[[ARG1_AGG:.*]] = "tf.CustomAggregator"(%[[ARG1_ID]])
    // CHECK: %[[RES:.*]] = "tf.XlaCallModule"(%[[ARG0_AGG]], %[[ARG1_AGG]])
    // CHECK: %[[RES_AGG:.*]] = "tf.CustomAggregator"(%[[RES]])
    // CHECK-DAG: %[[RES_ID:.*]] = "tf.Identity"(%[[RES_AGG]])
    // CHECK: return %[[RES_ID]] : tensor<?x10xf32>
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<?x100352xf32>) -> tensor<?x100352xf32>
    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<100352x10xf32>) -> tensor<100352x10xf32>
    %2 = "tf.XlaCallModule"(%0, %1) <{
        Sout = [#tf_type.shape<?x10>], dim_args_spec = [],
        disabled_checks = [], function_list = [],
        has_token_input_output = false, module = "", platforms = [],
        version = 5 : i64
    }> {
        _entry_function = @composite_dot_general_fn_1,
        _original_entry_function = "composite_dot_general_fn_1",
        _tfl_quant_trait = "fully_quantizable"
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
