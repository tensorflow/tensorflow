// RUN: tf-quant-opt %s -quant-issues-ids-of-custom-aggregation-ops | FileCheck %s

func.func @issue_ids(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0:4 = "tf.CustomAggregator"(%arg1) {id = "", calibration_method = 1 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  %1:4 = "tf.CustomAggregator"(%arg0) {id = "", calibration_method = 1 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  %2 = "tf.AddV2"(%1, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3:4 = "tf.CustomAggregator"(%2) {id = "", calibration_method = 1 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  func.return %3 : tensor<*xf32>
}


// CHECK: func @issue_ids
// CHECK-NEXT:  [[rhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg1) <{calibration_method = 1 : i32, id = "0", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>)
// CHECK-NEXT:  [[lhs:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"(%arg0) <{calibration_method = 1 : i32, id = "1", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>)
// CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[res:%.*]], {{.*}}, {{.*}}, {{.*}} = "tf.CustomAggregator"([[add]]) <{calibration_method = 1 : i32, id = "2", max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32, num_bins = 0 : i32}> : (tensor<*xf32>)
// CHECK-NEXT:  return [[res]] : tensor<*xf32>
