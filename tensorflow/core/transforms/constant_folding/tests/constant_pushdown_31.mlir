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

module {
  tfg.func @test() {
    %Const, %ctl = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_0, %ctl_1 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Placeholder, %ctl_2 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: Const{{.*}} name("child")
    // CHECK-SAME: value = dense<6.000000e+00>
    // CHECK: Const{{.*}} name("parent/child/_recip")
    // CHECK-SAME: value = dense<0.166666672>
    %Div, %ctl_3 = Div(%Placeholder, %Const) name("child") {T = f32} : (tensor<2x2xf32>, tensor<2xf32>) -> (tensor<2x2xf32>)
    // CHECK: Mul{{.*}} name("parent")
    %Div_4, %ctl_5 = Div(%Div, %Const_0) name("parent") {T = f32} : (tensor<2x2xf32>, tensor<2xf32>) -> (tensor<2x2xf32>)
    return
  }
}
