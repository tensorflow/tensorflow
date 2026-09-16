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
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Test int4 constants and conversions.

// CHECK-LABEL: ENTRY %main.{{.*}} () -> s4[6]
func.func @main() -> tensor<6xi4> {
  // CHECK-NEXT: %[[CONSTANT:.*]] = s4[6] constant({1, -2, -3, 4, -8, 7})
  %0 = mhlo.constant dense<[1, -2, -3, 4, -8, 7]> : tensor<6xi4>
  // CHECK-NEXT: %[[CONVERT1:.*]] = s8[6] convert(%[[CONSTANT]])
  %1 = "mhlo.convert"(%0) : (tensor<6xi4>) -> tensor<6xi8>
  // CHECK-NEXT: ROOT %[[CONVERT2:.*]] = s4[6] convert(%[[CONVERT1]])
  %2 = "mhlo.convert"(%1) : (tensor<6xi8>) -> tensor<6xi4>
  func.return %2 : tensor<6xi4>
}

// -----

// CHECK-LABEL: ENTRY %main.{{.*}} () -> u4[4]
func.func @main() -> tensor<4xui4> {
  // CHECK-NEXT: %[[CONSTANT:.*]] = u4[4] constant({1, 2, 3, 15})
  %0 = mhlo.constant dense<[1, 2, 3, 15]> : tensor<4xui4>
  // CHECK-NEXT: %[[CONVERT1:.*]] = u8[4] convert(%[[CONSTANT]])
  %1 = "mhlo.convert"(%0) : (tensor<4xui4>) -> tensor<4xui8>
  // CHECK-NEXT: ROOT %[[CONVERT2:.*]] = u4[4] convert(%[[CONVERT1]])
  %2 = "mhlo.convert"(%1) : (tensor<4xui8>) -> tensor<4xui4>
  func.return %2 : tensor<4xui4>
}
