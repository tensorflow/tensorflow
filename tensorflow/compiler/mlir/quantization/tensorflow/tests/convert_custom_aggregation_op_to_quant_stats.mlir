// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: tf-quant-opt %s -quant-convert-tf-custom-aggregator-op-to-quant-stats | FileCheck %s

func.func @customAggregator(%arg0: tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) {
  %0 = "tf.CustomAggregator"(%arg0) {min = -0.1 : f32, max = 0.2 : f32, id = "0"} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %1 = "tf.CustomAggregator"(%arg0) {id = "1"} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %0, %1 : tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>
}
// CHECK: func @customAggregator
// CHECK-NEXT: %[[stats:.*]] = "quant.stats"(%arg0) {layerStats = dense<[-1.000000e-01, 2.000000e-01]> : tensor<2xf32>} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
// CHECK-NEXT: return %[[stats]], %arg0

func.func @doNotHandleNoMinMaxCases(%arg0: tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) {
  %0 = "tf.CustomAggregator"(%arg0) {min = -0.1 : f32, id = "1"} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %1 = "tf.CustomAggregator"(%arg0) {max = 0.2 : f32, id = "2"} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %2 = "tf.CustomAggregator"(%arg0) {id = "3"} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %0, %1, %2 : tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>
}
// CHECK: func @doNotHandleNoMinMaxCases
// CHECK-NOT: "quant.stats"
