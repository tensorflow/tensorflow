// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=MIN_MAX'  | FileCheck --check-prefix=MIN-MAX-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=AVERAGE_MIN_MAX'  | FileCheck --check-prefix=AVERAGE-MIN-MAX-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_PERCENTILE'  | FileCheck --check-prefix=HISTOGRAM-PERCENTILE-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_MSE_BRUTEFORCE'  | FileCheck --check-prefix=HISTOGRAM-MSE-BRUTEFORCE-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_MSE_MAX_FREQUENCY'  | FileCheck --check-prefix=HISTOGRAM-MSE-MAX-FREQUENCY-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_MSE_SYMMETRIC'  | FileCheck --check-prefix=HISTOGRAM-MSE-SYMMETRIC-CHECK %s

module {
  func.func @add_custom_ops(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %add : tensor<*xf32>
  }

  func.func @no_custom_ops_on_non_f32_type(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %add : tensor<*xi32>
  }

  func.func @composite_conv2d_with_bias_and_relu6_fn(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.Conv2D"(%arg0, %arg1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.BiasAdd"(%0, %arg2) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "tf.Relu6"(%1) : (tensor<*xf32>) -> tensor<*xf32>
    func.return %2 : tensor<*xf32>
  }
}

// CalibrationOptions(calibration_method=CALIBRATION_METHOD_MIN_MAX)
// MIN-MAX-CHECK: func @add_custom_ops
// MIN-MAX-CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {calibration_method = 1 : i32, id = "", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// MIN-MAX-CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 1 : i32, id = "", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// MIN-MAX-CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// MIN-MAX-CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {calibration_method = 1 : i32, id = "", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// MIN-MAX-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// MIN-MAX-CHECK: func @no_custom_ops_on_non_f32_type
// MIN-MAX-CHECK-NEXT:  "tf.AddV2"
// MIN-MAX-CHECK-NEXT:  return

// MIN-MAX-CHECK: func @composite_conv2d_with_bias_and_relu6_fn
// MIN-MAX-CHECK-NEXT:  "tf.Conv2D"
// MIN-MAX-CHECK-NEXT:  "tf.BiasAdd"
// MIN-MAX-CHECK-NEXT:  "tf.Relu6"
// MIN-MAX-CHECK-NEXT:  return

// CalibrationOptions(calibration_method=CALIBRATION_METHOD_AVERAGE_MIN_MAX)
// AVERAGE-MIN-MAX-CHECK: func @add_custom_ops
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {calibration_method = 2 : i32, id = "", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 2 : i32, id = "", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {calibration_method = 2 : i32, id = "", initial_num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// AVERAGE-MIN-MAX-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// AVERAGE-MIN-MAX-CHECK: func @no_custom_ops_on_non_f32_type
// AVERAGE-MIN-MAX-CHECK-NEXT:  "tf.AddV2"
// AVERAGE-MIN-MAX-CHECK-NEXT:  return

// AVERAGE-MIN-MAX-CHECK: func @composite_conv2d_with_bias_and_relu6_fn
// AVERAGE-MIN-MAX-CHECK-NEXT:  "tf.Conv2D"
// AVERAGE-MIN-MAX-CHECK-NEXT:  "tf.BiasAdd"
// AVERAGE-MIN-MAX-CHECK-NEXT:  "tf.Relu6"
// AVERAGE-MIN-MAX-CHECK-NEXT:  return

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
//   calibration_parameters=CalibrationParameters(initial_num_bins=256, min_percentile=0.001, max_percentile=99.999)
// )
// HISTOGRAM-PERCENTILE-CHECK: func @add_custom_ops
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {calibration_method = 3 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 3 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {calibration_method = 3 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 9.999900e+01 : f32, min_percentile = 1.000000e-03 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// HISTOGRAM-PERCENTILE-CHECK: func @no_custom_ops_on_non_f32_type
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  "tf.AddV2"
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  return

// HISTOGRAM-PERCENTILE-CHECK: func @composite_conv2d_with_bias_and_relu6_fn
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  "tf.Conv2D"
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  "tf.BiasAdd"
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  "tf.Relu6"
// HISTOGRAM-PERCENTILE-CHECK-NEXT:  return

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
//   calibration_parameters=CalibrationParameters(initial_num_bins=256)
// )
// HISTOGRAM-MSE-BRUTEFORCE-CHECK: func @add_custom_ops
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {calibration_method = 4 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 4 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {calibration_method = 4 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// HISTOGRAM-MSE-BRUTEFORCE-CHECK: func @no_custom_ops_on_non_f32_type
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  "tf.AddV2"
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  return

// HISTOGRAM-MSE-BRUTEFORCE-CHECK: func @composite_conv2d_with_bias_and_relu6_fn
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  "tf.Conv2D"
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  "tf.BiasAdd"
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  "tf.Relu6"
// HISTOGRAM-MSE-BRUTEFORCE-CHECK-NEXT:  return

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
//   calibration_parameters=CalibrationParameters(initial_num_bins=256)
// )
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK: func @add_custom_ops
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {calibration_method = 5 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 5 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {calibration_method = 5 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK: func @no_custom_ops_on_non_f32_type
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  "tf.AddV2"
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  return

// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK: func @composite_conv2d_with_bias_and_relu6_fn
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  "tf.Conv2D"
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  "tf.BiasAdd"
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  "tf.Relu6"
// HISTOGRAM-MSE-MAX-FREQUENCY-CHECK-NEXT:  return

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
//   calibration_parameters=CalibrationParameters(initial_num_bins=256)
// )
// HISTOGRAM-MSE-SYMMETRIC-CHECK: func @add_custom_ops
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {calibration_method = 6 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {calibration_method = 6 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {calibration_method = 6 : i32, id = "", initial_num_bins = 256 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// HISTOGRAM-MSE-SYMMETRIC-CHECK: func @no_custom_ops_on_non_f32_type
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  "tf.AddV2"
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  return

// HISTOGRAM-MSE-SYMMETRIC-CHECK: func @composite_conv2d_with_bias_and_relu6_fn
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  "tf.Conv2D"
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  "tf.BiasAdd"
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  "tf.Relu6"
// HISTOGRAM-MSE-SYMMETRIC-CHECK-NEXT:  return

