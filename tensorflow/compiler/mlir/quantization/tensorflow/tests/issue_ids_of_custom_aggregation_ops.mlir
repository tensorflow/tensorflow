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

// RUN: tf-quant-opt %s -quant-issues-ids-of-custom-aggregation-ops | FileCheck %s

func.func @issue_ids(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.CustomAggregator"(%arg1) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "tf.CustomAggregator"(%arg0) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
  %2 = "tf.AddV2"(%1, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = "tf.CustomAggregator"(%2) {id = ""} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
}


// CHECK: func @issue_ids
// CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {id = "0"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {id = "1"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {id = "2"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:  return [[res]] : tensor<*xf32>
