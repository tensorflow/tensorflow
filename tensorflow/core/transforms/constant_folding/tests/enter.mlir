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
  tfg.func @test() -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
    %Placeholder, %ctl = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    // CHECK: , %[[CTRL_C1:.*]] = Const {{.*}} name("c1")
    %Const, %ctl_0 = Const [%ctl] name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<f32>} : () -> (tensor<f32>)
    %Enter, %ctl_1 = Enter(%Placeholder) name("enter1") {T = f32, frame_name = "foo", is_constant = true, parallel_iterations = 10 : i64} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: %[[CONST_0:.*]], %[[CTRL:.*]] = Const [%[[ENTER_CTRL:.*]]] name("enter2/_enter")
    // CHECK: , %[[ENTER_CTRL]] = Enter{{.*}} name("enter2")
    %Enter_2, %ctl_3 = Enter(%Const) name("enter2") {T = f32, frame_name = "foo", is_constant = true, parallel_iterations = 10 : i64} : (tensor<f32>) -> (tensor<*xf32>)
    %Enter_4, %ctl_5 = Enter(%Const) name("enter3") {T = f32, frame_name = "foo", is_constant = false, parallel_iterations = 10 : i64} : (tensor<f32>) -> (tensor<*xf32>)
    // CHECK: Identity{{.*}} name("id1")
    %Identity, %ctl_6 = Identity(%Enter) name("id1") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Const [%[[CTRL]]] name("id2")
    %Identity_7, %ctl_8 = Identity(%Enter_2) name("id2") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Const [%[[CTRL]]] name("id3")
    %Identity_9, %ctl_10 = Identity(%Enter_2) name("id3") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Identity{{.*}} name("id4")
    %Identity_11, %ctl_12 = Identity(%Enter_4) name("id4") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    return (%Identity, %Identity_7, %Identity_9, %Identity_11) : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
  }
}
