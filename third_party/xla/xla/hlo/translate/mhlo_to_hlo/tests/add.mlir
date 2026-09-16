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
// RUN: xla-translate -mlir-hlo-to-hlo-text %s | FileCheck %s
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args %s | FileCheck %s --check-prefix=TUPLE-ARG
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLE-RET
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLES

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[4], Arg_1.1: f32[4]) -> f32[4]
// TUPLE-ARG-LABEL: ENTRY %main.{{.*}} (arg_tuple.1: (f32[4], f32[4])) -> f32[4]
// TUPLE-RET-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[4], Arg_1.1: f32[4]) -> (f32[4])
// TUPLES-LABEL: ENTRY %main.{{.*}} (arg_tuple.1: (f32[4], f32[4])) -> (f32[4])

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %Arg_0.1 = f32[4] parameter(0)
  // CHECK-NEXT: %Arg_1.1 = f32[4] parameter(1)

  // CHECK-NEXT: %add.2 = f32[4] add(%Arg_0.1, %Arg_1.1)
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT: ROOT %add.3 = f32[4] add(%add.2, %Arg_1.1)
  %1 = "mhlo.add"(%0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}
