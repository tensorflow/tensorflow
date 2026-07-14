// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
// RUN: hlo-translate -mlir-to-hlo -emit-proto %s | FileCheck %s

// CHECK: name: "foobar
// CHECK: entry_computation_name: "main
// CHECK: computations {
// CHECK: name: "main
// CHECK: instructions {
// CHECK: name: "Arg_
// CHECK: opcode: "parameter"
// CHECK: name: "add
// CHECK: opcode: "add"
// CHECK: name: "dot
// CHECK: opcode: "dot"
module @foobar {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = stablehlo.dot %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
}
