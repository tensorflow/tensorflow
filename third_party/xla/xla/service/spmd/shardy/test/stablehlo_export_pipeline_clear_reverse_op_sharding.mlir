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
// =============================================================================

// RUN: sdy_opt %s -split-input-file -xla-sdy-stablehlo-export-pipeline="clear-reverse-op-sharding=true" 2>&1 | FileCheck %s

sdy.mesh @mesh_1 = <["axis_0"=16]>

// CHECK-LABEL: func @reverse_blocked(
// CHECK-SAME:      %arg0: tensor<8x16xf32> {mhlo.sharding = "{devices=[1,16]<=[16]}"})
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @reverse_blocked(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {"axis_0"}]>}) -> tensor<8x16xf32> {
  // CHECK-NOT: mhlo.sharding
  // CHECK: %0 = stablehlo.reverse %arg0, dims = [1] : tensor<8x16xf32>
  %0 = stablehlo.reverse %arg0, dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"axis_0"}]>]>} : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
