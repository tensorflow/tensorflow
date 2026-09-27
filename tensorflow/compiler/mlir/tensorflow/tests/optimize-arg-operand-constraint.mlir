// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt -tf-optimize %s | FileCheck %s

// Check passing an argument into DefinedByConv2D constraint does not crash.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32>
attributes  {tf.entry_function = {inputs = "input", outputs = "output_node"}} {
  %0 = arith.constant dense<2.000000e+00> : tensor<f32>
  %1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %2 = "tf.AddV2"(%arg0, %1) {T = "tfdtype$DT_FLOAT", device = "", name = "StatefulPartitionedCall/add"} : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  %3 = "tf.Mul"(%2, %0) {T = "tfdtype$DT_FLOAT", device = "", name = "output_node"} : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %3 : tensor<1xf32>
}
