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
    // CHECK: %[[VAR:.*]], {{.*}} name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<2>, shared_name = ""} : () -> (tensor<2x!tf_type.f32ref>)
    %VariableV2_0, %ctl_1 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<4>, shared_name = ""} : () -> (tensor<4x!tf_type.f32ref>)
    // CHECK: , %[[CTRL:.*]] = Const name("split_dim")
    %Const, %ctl_2 = Const name("split_dim") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    // CHECK: Identity(%[[VAR]]) [%[[CTRL]]] name("s1")
    %Split, %ctl_3 = Split(%Const, %VariableV2) name("s1") {T = f32, num_split = 1 : i64} : (tensor<i32>, tensor<2x!tf_type.f32ref>) -> (tensor<*xf32>)
    %Split_4:2, %ctl_5 = Split(%Const, %VariableV2_0) name("s2") {T = f32, num_split = 2 : i64} : (tensor<i32>, tensor<4x!tf_type.f32ref>) -> (tensor<*xf32>, tensor<*xf32>)
    %Add, %ctl_6 = Add(%Split, %Split_4#0) name("out") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    return
  }
}
