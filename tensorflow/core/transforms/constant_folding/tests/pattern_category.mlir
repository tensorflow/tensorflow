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
// RUN: tfg-transforms-opt -tfg-constant-folding=pattern-category=1 %s | FileCheck %s --check-prefix=CHECK-CAT1
// RUN: tfg-transforms-opt -tfg-constant-folding=pattern-category=2 %s | FileCheck %s --check-prefix=CHECK-CAT2

module {
  tfg.func @test() -> (tensor<*xf32>, tensor<*xf32>) {
    // CHECK-CAT1: , %[[CTRL:.*]] = Const name("c")
    // CHECK-CAT2: , %[[CTRL:.*]] = Const name("c")
    %Const, %ctl = Const name("c") {dtype = f32, value = dense<3.140000e+00> : tensor<1000xf32>} : () -> (tensor<1000xf32>)
    // CHECK-CAT1: Const [%[[CTRL]]] name("i1")
    // CHECK-CAT2: Identity{{.*}} name("i1")
    %Identity, %ctl_0 = Identity(%Const) name("i1") {T = f32} : (tensor<1000xf32>) -> (tensor<*xf32>)
    // CHECK-CAT1: Const [%[[CTRL]]] name("i2")
    // CHECK-CAT2: Identity{{.*}} name("i2")
    %Identity_1, %ctl_2 = Identity(%Const) name("i2") {T = f32} : (tensor<1000xf32>) -> (tensor<*xf32>)
    return (%Identity, %Identity_1) : tensor<*xf32>, tensor<*xf32>
  }
}