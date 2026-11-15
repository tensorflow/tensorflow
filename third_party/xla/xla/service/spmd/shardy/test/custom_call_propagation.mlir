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
// RUN: sdy_opt %s -sdy-basic-propagate -split-input-file | FileCheck %s

sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>

// CHECK-LABEL: func @cx64_combine_used_then_rng_bit_generator
func.func @cx64_combine_used_then_rng_bit_generator(
    %arg0: tensor<2xui32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>},
    %arg1: tensor<2xui32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>})
    -> (tensor<4x1000xui32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}, tensor<2xui64>) {
  // CHECK-NEXT: %[[X64_COMBINE:.*]] = stablehlo.custom_call @X64Combine(%arg0, %arg1)
  // CHECK-NOT:  sdy.sharding
  // CHECK-NEXT: %output_state, %output = stablehlo.rng_bit_generator %0, algorithm =  DEFAULT
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{?}]>, <@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui64>
  %output_state, %output = stablehlo.rng_bit_generator %0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4x1000xui32>)
  return %output, %0 : tensor<4x1000xui32>, tensor<2xui64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>

// CHECK-LABEL: func @x64_combine_propagates_without_rng_bit_generator
func.func @x64_combine_propagates_without_rng_bit_generator(
    %arg0: tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
    %arg1: tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
    -> tensor<8x2xui64> {
  // CHECK-NEXT: %[[X64_COMBINE:.*]] = stablehlo.custom_call @X64Combine(%arg0, %arg1)
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  return %0 : tensor<8x2xui64>
}
