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
// RUN: hlo-translate -mlir-to-hlo %s.bc | FileCheck %s

// File `vhlo_input.mlir.bc` is created by running the following command:
//  $ stablehlo-translate --serialize --target=1.0.0 --strip-debuginfo vhlo_input.mlir > vhlo_input.mlir.bc
//
// The `.mlir.bc` file is used in the above RUN command, along with the
// filechecks specified in this file.

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[4], Arg_1.1: f32[4]) -> f32[]
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  // CHECK: %Arg_0.1 = f32[4] parameter(0)
  // CHECK: %Arg_1.1 = f32[4] parameter(1)
  // CHECK: %add.1 = f32[4] add(%Arg_0.1, %Arg_1.1)
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  // CHECK: ROOT %dot.1 = f32[] dot(%add.1, %Arg_1.1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  %1 = stablehlo.dot %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}