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
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<4x6>, shared_name = ""} : () -> (tensor<4x6x!tf_type.f32ref>)
    %VariableV2_0, %ctl_1 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<4x3>, shared_name = ""} : () -> (tensor<4x3x!tf_type.f32ref>)
    // CHECK: , %[[CTRL:.*]] = Const name("multiplies1")
    %Const, %ctl_2 = Const name("multiplies1") {dtype = i32, value = dense<1> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %Const_3, %ctl_4 = Const name("multiplies2") {dtype = i32, value = dense<[1, 2]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK: Identity(%[[VAR]]) [%[[CTRL]]] name("t1")
    %Tile, %ctl_5 = Tile(%VariableV2, %Const) name("t1") {T = f32, Tmultiples = i32} : (tensor<4x6x!tf_type.f32ref>, tensor<2xi32>) -> (tensor<*xf32>)
    %Tile_6, %ctl_7 = Tile(%VariableV2_0, %Const_3) name("t2") {T = f32, Tmultiples = i32} : (tensor<4x3x!tf_type.f32ref>, tensor<2xi32>) -> (tensor<*xf32>)
    %Add, %ctl_8 = Add(%Tile, %Tile_6) name("out") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    return
  }
}
