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
// RUN: tf-opt %s -tf-standard-pipeline | FileCheck %s

// CHECK-LABEL: removeDeadReadVariableOp
func.func @removeDeadReadVariableOp(%arg0: tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32> {
  %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  func.return %0: tensor<f32>

 // CHECK: %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
 // CHECK: return %0
}
