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
// RUN: tfg-transforms-opt --tfg-remapper=enable-onednn-patterns %s | FileCheck %s

// -----

// CHECK-LABEL: tfg.func @fusedmish_test
tfg.func @fusedmish_test() {
  // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input_tensor")
  %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input_tensor") {dtype = f32, shape = #tf_type.shape<64x64>} : () -> (tensor<64x64xf32>)
  // CHECK: %[[SOFTPLUS:.*]], {{.*}} name("Softplus")
  %Softplus, %ctl_0 = Softplus(%Placeholder) device("/device:CPU:0") name("Softplus") {T = f32} : (tensor<64x64xf32>) -> (tensor<64x64xf32>)
  // CHECK: %[[TANH:.*]], {{.*}} name("Tanh")
  %Tanh, %ctl_1 = Tanh(%Softplus) device("/device:CPU:0") name("Tanh") {T = f32} : (tensor<64x64xf32>) -> (tensor<64x64xf32>)
  // CHECK: _MklFusedMish(%[[PLACEHOLDER:.*]]) {{.*}} name("Mul")
  %Mul, %ctl_2 = Mul(%Placeholder, %Tanh) device("/device:CPU:0") name("Mul") {T = f32} : (tensor<64x64xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>)
  return
}
