// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tf-quant-opt %s -quant-convert-tf-custom-aggregator-op-to-quant-stats | FileCheck %s

func.func @customAggregator(%arg0: tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) {
  %0:4 = "tf.CustomAggregator"(%arg0) {min = -0.1 : f32, max = 0.2 : f32, id = "0", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  %1:4 = "tf.CustomAggregator"(%arg0) {id = "1", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  func.return %0#0, %1#0 : tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>
}
// CHECK: func @customAggregator
// CHECK-NEXT: %[[stats:.*]] = "quantization.stats"(%arg0) <{layerStats = dense<[-1.000000e-01, 2.000000e-01]> : tensor<2xf32>}> : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
// CHECK-NEXT: return %[[stats]], %arg0

func.func @doNotHandleNoMinMaxCases(%arg0: tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) {
  %0:4 = "tf.CustomAggregator"(%arg0) {min = -0.1 : f32, id = "1", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  %1:4 = "tf.CustomAggregator"(%arg0) {max = 0.2 : f32, id = "2", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  %2:4 = "tf.CustomAggregator"(%arg0) {id = "3", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32} : (tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  func.return %0#0, %1#0, %2#0 : tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>
}
// CHECK: func @doNotHandleNoMinMaxCases
// CHECK-NOT: "quantization.stats"
