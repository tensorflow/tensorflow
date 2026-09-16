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
    %Const, %ctl_0 = Const name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    // CHECK: Const {{.*}} name("id
    %Identity, %ctl_1 = Identity(%Const) name("id") {T = f32} : (tensor<2xf32>) -> (tensor<2xf32>)
    // CHECK: Identity{{.*}} name("id_1")
    %Identity_1, %ctl_2 = Identity(%Identity) name("id_1") {T = f32} : (tensor<2xf32>) -> (tensor<2xf32>)
    // CHECK: Identity{{.*}} name("id_2")
    %Identity_2, %ctl_3 = Identity(%Const) name("id_2") {T = f32} : (tensor<2xf32>) -> (tensor<2xf32>)
    return [%ctl_2, %ctl_3]
  }
}
