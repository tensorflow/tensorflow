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
    // CHECK: , %[[CTRL:.*]] = Const name("if")
    %Const, %ctl = Const name("if") {dtype = i1, value = dense<false> : tensor<2x2xi1>} : () -> (tensor<2x2xi1>)
    // CHECK: , %[[CTRL0:.*]] = Placeholder name("then")
    %Placeholder, %ctl_0 = Placeholder name("then") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[PLACEHOLDER1:.*]], {{.*}} = Placeholder name("else")
    %Placeholder_1, %ctl_2 = Placeholder name("else") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: Identity(%[[PLACEHOLDER1]]) [%[[CTRL]], %[[CTRL0]]] name("select")
    %SelectV2, %ctl_3 = SelectV2(%Const, %Placeholder, %Placeholder_1) name("select") {T = f32} : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    %Identity, %ctl_4 = Identity(%SelectV2) name("id") {T = f32} : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
    return
  }
}
