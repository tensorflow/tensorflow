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
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLE

// Test to verify that multiple result function with always emit return tuple
// does not result in nested tuples.

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: s32[4]) -> (s32[4], s32[1,2,3,4])
// TUPLE-LABEL: ENTRY %main.{{.*}} (arg_tuple.1: (s32[4])) -> (s32[4], s32[1,2,3,4])
func.func @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<1x2x3x4xi32>) {
  // CHECK-NEXT: %Arg_0.1 = s32[4] parameter(0)
  // CHECK-NEXT: %broadcast.1 = s32[1,2,3,4] broadcast(%Arg_0.1), dimensions={3}
  %0 = "mhlo.broadcast"(%arg0) <{broadcast_sizes = dense<[1,2,3]> : tensor<3xi64>}> : (tensor<4xi32>) -> tensor<1x2x3x4xi32>
  func.return %arg0, %0 : tensor<4xi32>, tensor<1x2x3x4xi32>
}
