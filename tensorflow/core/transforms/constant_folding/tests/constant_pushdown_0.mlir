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
// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module  {
  tfg.func @test() {
    %Const, %ctl = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_0, %ctl_1 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("x")
    %Placeholder, %ctl_2 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[CONST:.*]], {{.*}} = Const {{.*}} name("child")
    %Add, %ctl_3 = Add(%Const, %Placeholder) name("child") {T = f32} : (tensor<2xf32>, tensor<2x2xf32>) -> (tensor<*xf32>)
    // CHECK: Add(%[[PLACEHOLDER]], %[[CONST]]) name("parent")
    %Add_4, %ctl_5 = Add(%Const_0, %Add) name("parent") {T = f32} : (tensor<2xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    return
  }
}
