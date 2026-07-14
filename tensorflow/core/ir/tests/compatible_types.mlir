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
// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// CHECK: tfg.func @case
// CHECK-SAME: tensor<*x!tf_type.resource<tensor<32xf32>>>
tfg.func @case(%arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<i32>) {
  %A, %ctl = A : () -> (tensor<i32>)
  return(%A) : tensor<i32>
}

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  %index, %arg, %ctl = Args : () -> (tensor<i32>, tensor<*x!tf_type.resource>)
  // CHECK: Case
  // CHECK-SAME: @case
  // CHECK-SAME: tensor<*x!tf_type.resource>
  %Case, %ctl_0 = Case(%index, %arg) {
    Tin = [!tf_type.resource], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case, {}>]
  } : (tensor<i32>, tensor<*x!tf_type.resource>) -> (tensor<i32>)
}
