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
// RUN: tf-opt -tf-to-hlo-pipeline %s | FileCheck %s

// Verifies that constants generated post shape inference are propagated.
// get_shape result in this test.
module attributes {tf.versions = {producer = 179 : i32}} {

  // CHECK-LABEL: func @main
  func.func @main(%arg0: tensor<10x19xf32>, %arg1: tensor<19x10xf32> {mhlo.is_same_data_across_replicas = true}) -> tensor<?xi64> {
    %0 = "tf.Shape"(%arg0) : (tensor<10x19xf32>) -> tensor<2xi64>
    %1 = "tf.Reshape"(%arg1, %0) : (tensor<19x10xf32>, tensor<2xi64>) -> tensor<?x?xf32>

    // CHECK: %[[RESULT:.*]] = mhlo.constant dense<[10, 19]>
    %2 = "tf.PartitionedCall"(%1) {config = "", config_proto = "", executor_type = "", f = @get_shape} : (tensor<?x?xf32>) -> (tensor<?xi64>)

    // CHECK: return %[[RESULT]]
    func.return %2 : tensor<?xi64>
  }

  // CHECK-LABEL: func @get_shape
  func.func @get_shape(%arg0 : tensor<*xi64>) -> tensor<?xi64> {
    %0 = "tf.Shape"(%arg0) : (tensor<*xi64>) -> tensor<?xi64>
    func.return %0 : tensor<?xi64>
  }

}

