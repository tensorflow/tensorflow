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

// Tests that the exported HLO module keeps parameter replication annotation.

// CHECK:  HloModule
func.func @main(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32> {mhlo.is_same_data_across_replicas = true}) -> tensor<16x16xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[16,16] parameter(0)
// CHECK-NOT: parameter_replication={true}
// CHECK:  %[[ARG1:.*]] = f32[16,16] parameter(1), parameter_replication={true}
// CHECK:  ROOT %[[RESULT:.*]] = f32[16,16] add(%[[ARG0]], %[[ARG1]])
