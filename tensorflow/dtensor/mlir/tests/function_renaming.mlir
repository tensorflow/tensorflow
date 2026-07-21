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
// RUN: dtensor-opt %s -split-input-file -dtensor-function-renaming -verify-diagnostics | FileCheck %s

module attributes {dtensor.cache_key = "_abc_def"}  {
  // CHECK-LABEL: func @main
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
    // CHECK:       "tf.StatefulPartitionedCall"
    // CHECK-SAME:  f = @_func_0_abc_def
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {f = @_func_0, config = "", config_proto = "", executor_type = ""} : (tensor<f32>, tensor<4xf32>) -> (tensor<4xf32>)
    func.return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: func private @_func_0_abc_def
  func.func private @_func_0(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<f32>, tensor<4xf32>) -> (tensor<4xf32>)
    func.return %0 : tensor<4xf32>
  }
}
