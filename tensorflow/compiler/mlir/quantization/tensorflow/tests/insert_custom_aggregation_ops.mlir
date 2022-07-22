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

// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops | FileCheck %s

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

// CHECK: func @add_custom_ops
// CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  return [[res]] : tensor<*xf32>

// CHECK: func @no_custom_ops_on_non_f32_type
// CHECK-NEXT:  "tf.AddV2"
// CHECK-NEXT:  return
