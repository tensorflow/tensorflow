// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=MIN_MAX'  | FileCheck --check-prefix=MIN-MAX-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=AVERAGE_MIN_MAX'  | FileCheck --check-prefix=AVERAGE-MIN-MAX-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=HISTOGRAM_PERCENTILE'  | FileCheck --check-prefix=HISTOGRAM-PERCENTILE-CHECK %s

module {
  func.func @add_custom_ops(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %add : tensor<*xf32>
  }

  func.func @no_custom_ops_on_non_f32_type(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %add : tensor<*xi32>
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

// CalibrationOptions(
//   calibration_method=CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
//   calibration_parameter=CalibrationParameter(num_bins=256, min_percentile=0.001, max_percentile=99.999)
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
