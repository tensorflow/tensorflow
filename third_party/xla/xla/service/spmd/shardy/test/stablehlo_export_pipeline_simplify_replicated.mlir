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
// RUN: sdy_opt %s -split-input-file -xla-sdy-stablehlo-export-pipeline='enable-hlo-sharding-v3=true simplify-replicated-shardings=true' 2>&1 | FileCheck %s

sdy.mesh @mesh_1 = <["axis_0"=16]>

// CHECK-LABEL: func @simplify_replicated_sharding(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{mesh[], replicated}"},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {mhlo.sharding = "{mesh['axis_0'=16], [{?}, {}]}"})
func.func @simplify_replicated_sharding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}]>},
                                        %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{?}, {}]>}) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh_1 = <["axis_0"=16]>

// CHECK-LABEL: func @simplify_explicit_replicated_sharding(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{mesh[], replicated}"})
func.func @simplify_explicit_replicated_sharding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}], replicated={"axis_0"}>}) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh_1 = <["axis_0"=16]>

// CHECK-LABEL: func @do_not_simplify_unreduced(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{mesh['axis_0'=16], [{}, {}], unreduced={'axis_0'}}"})
func.func @do_not_simplify_unreduced(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}], unreduced={"axis_0"}>}) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
